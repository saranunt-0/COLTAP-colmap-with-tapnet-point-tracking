# SPDX-License-Identifier: BSD-3-Clause
"""TAPNext++ model loading and single-frame online inference.

Frames are resized to a square ``resolution x resolution`` input, but query and
output coordinates always live in the network's native 256x256 space (also for
the 512 checkpoint). That space follows the COLMAP keypoint convention:
``(0, 0)`` is the top-left corner of the top-left pixel, so a pixel center sits
at ``+0.5`` and converting to image coordinates is a pure per-axis scale.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import urllib.request
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from .third_party.tapnext import (
    TAPNext,
    TAPNextTrackingState,
    tracker_certainty,
)

LOGGER = logging.getLogger(__name__)

# Raised by the vendored certainty helper (torch.vmap(torch.meshgrid)); the
# default indexing it relies on is the one upstream was validated with.
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)

# The network is always built for 256x256 (its learned positional embedding is
# 32x32 patches); the 512 checkpoint interpolates that embedding.
MODEL_COORD_SIZE = 256

CHECKPOINT_URLS = {
    256: "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
    512: "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
}


def default_cache_dir() -> Path:
    return Path(
        os.environ.get("COLTAP_CACHE_DIR", Path.home() / ".cache" / "coltap")
    )


def resolve_checkpoint(path: str | Path | None, resolution: int) -> Path:
    """Return a local checkpoint path, downloading the official one if needed.

    The downloaded Lightning checkpoint (~2.5 GB, includes optimizer state) is
    slimmed to a plain fp32 ``state_dict`` (~0.8 GB) on first use so that later
    runs load faster.
    """
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TAPNext++ checkpoint not found: {path}")
        return path
    if resolution not in CHECKPOINT_URLS:
        raise ValueError(
            f"No official TAPNext++ checkpoint for resolution {resolution}; "
            f"choose one of {sorted(CHECKPOINT_URLS)} or pass a checkpoint."
        )
    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = CHECKPOINT_URLS[resolution]
    raw_path = cache_dir / Path(url).name
    slim_path = cache_dir / f"tapnextpp_{resolution}_weights.pt"
    if slim_path.exists():
        return slim_path
    if not raw_path.exists():
        LOGGER.info("Downloading TAPNext++ checkpoint %s", url)
        tmp_path = raw_path.with_suffix(raw_path.suffix + ".part")
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f, length=1 << 22)
        tmp_path.rename(raw_path)
    state_dict = _read_state_dict(raw_path)
    torch.save(state_dict, slim_path)
    return slim_path


def _read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip the Lightning-style "tapnext." prefix if present.
    return {k.removeprefix("tapnext."): v for k, v in state_dict.items()}


def resolve_device(device: str) -> torch.device:
    if device in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclasses.dataclass
class StepOutput:
    """Per-query predictions for one frame, in model coordinates."""

    xy: np.ndarray  # [Q, 2] float32, (x, y) in [0, MODEL_COORD_SIZE]
    visibility: np.ndarray  # [Q] float32 in [0, 1], sigmoid(visible_logit)
    certainty: np.ndarray  # [Q] float32 in [0, 1], mass within radius


class TapNextPP:
    """Thin wrapper around TAPNext++ for frame-by-frame (online) tracking."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        resolution: int = 256,
        device: str = "auto",
        num_threads: int = 0,
        certainty_radius: float = 8.0,
        precision: str = "auto",
    ):
        if resolution % 8 != 0:
            raise ValueError("TAPNext++ resolution must be a multiple of 8.")
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        self.resolution = resolution
        self.device = resolve_device(device)
        self.certainty_radius = certainty_radius
        ckpt_path = resolve_checkpoint(checkpoint, resolution)
        net = TAPNext(image_size=(MODEL_COORD_SIZE, MODEL_COORD_SIZE))
        net.load_state_dict(_read_state_dict(ckpt_path))
        self.net = net.to(self.device).eval()
        if precision == "auto":
            precision = "fp16" if self.device.type == "cuda" else "fp32"
        dtypes = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
        if precision not in dtypes:
            raise ValueError(f"Unknown precision {precision!r}")
        self.precision = precision
        self._autocast_dtype = dtypes[precision]

    def preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """uint8 [H, W, 3] RGB -> float [1, 1, S, S, 3] in [-1, 1].

        INTER_AREA is used for downsampling to avoid aliasing on high
        resolution inputs (the upstream demo uses plain bilinear).
        """
        s = self.resolution
        h, w = image_rgb.shape[:2]
        interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_LINEAR
        resized = cv2.resize(image_rgb, (s, s), interpolation=interp)
        tensor = torch.from_numpy(resized).to(self.device).float()
        tensor = tensor / 127.5 - 1.0
        return tensor[None, None]

    @torch.no_grad()
    def step(
        self,
        frame: torch.Tensor,
        query_xy: np.ndarray | None = None,
        state: TAPNextTrackingState | None = None,
    ) -> tuple[StepOutput, TAPNextTrackingState]:
        """Advance the tracker by one frame.

        Pass ``query_xy`` ([Q, 2] (x, y) in MODEL_COORD_SIZE space) on the
        first frame of a new tracker instance and ``state`` on every later
        frame.
        """
        queries = None
        if state is None:
            if query_xy is None:
                raise ValueError("query_xy is required to start tracking.")
            q = np.zeros((len(query_xy), 3), dtype=np.float32)
            q[:, 1] = query_xy[:, 1]  # TAPNext expects (t, y, x).
            q[:, 2] = query_xy[:, 0]
            queries = torch.from_numpy(q)[None].to(self.device)
        ctx = torch.autocast(
            self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )
        with ctx:
            tracks_yx, track_logits, vis_logits, state = self.net(
                video=frame, query_points=queries, state=state
            )
        tracks_yx = tracks_yx.float()
        certainty = tracker_certainty(
            tracks_yx, track_logits.float(), self.certainty_radius
        )
        xy = tracks_yx[0, 0].cpu().numpy()[:, ::-1]
        out = StepOutput(
            xy=np.ascontiguousarray(xy, dtype=np.float32),
            visibility=torch.sigmoid(vis_logits.float())[0, 0, :, 0]
            .cpu()
            .numpy(),
            certainty=certainty[0, 0, :, 0].cpu().numpy(),
        )
        return out, state


def image_to_model(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array(
        [MODEL_COORD_SIZE / width, MODEL_COORD_SIZE / height], np.float32
    )
    return (np.asarray(xy, np.float32) * scale).astype(np.float32)


def model_to_image(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array(
        [width / MODEL_COORD_SIZE, height / MODEL_COORD_SIZE], np.float32
    )
    return (np.asarray(xy, np.float32) * scale).astype(np.float32)
