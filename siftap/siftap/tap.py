# SPDX-License-Identifier: BSD-3-Clause
"""TAPNext++ measurements with heatmap-based covariances.

Every target is tracked through TAPNext++ as a set of query points: the
target itself plus support points around it (all targets share one model
pass). Per point and frame the model gives a position, P(visible) and a
coordinate distribution (softmax over 256 bins per axis); its spread is the
point's sigma (see coltap.model.heatmap_sigma).

Per target this yields one Measurement:

* ``point_mode="direct"`` (point targets): the target point's own estimate
  with its own sigma. Support points are only used while the target point is
  reported occluded.
* ``point_mode="fit"`` / area targets: a robust weighted similarity fit over
  all visible points (see geometry.fit_similarity).

TAPNext++'s error is not white: on weak texture it slowly wanders off (the
benchmark's low-texture sky and ground drift by several pixels over 60
frames while the heatmap stays fairly peaked). The heatmap describes the
per-frame noise; the drift rate is predicted from the image itself with the
Lucas-Kanade / Shi-Tomasi texture information around the tracked point
(texture_sigma, the Cramer-Rao bound of patch alignment), which predicts
TAPNext++'s late error much better than its own heatmap (log-correlation
0.86 vs 0.74 per target in the benchmark).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from coltap.model import (
    MODEL_COORD_SIZE,
    TapNextPP,
    image_to_model,
    model_to_image,
)

from .geometry import Measurement, fit_similarity, texture_sigma
from .targets import Target


@dataclasses.dataclass
class TapCalibration:
    sigma_mode: str = "local"  # "local", "global" or "max" of the two
    # Defaults fitted on the synthetic calibration sequences
    # (benchmark/evaluate.py --stage calibrate).
    sigma_scale: float = 0.190  # multiplies the heatmap sigma (image px)
    sigma_floor: float = 0.20  # pixels, added in quadrature
    min_visibility: float = 0.5
    correlation: float = 0.3  # error correlation between support points
    point_mode: str = "direct"  # "direct" or "fit" for point targets
    # Drift: per-frame std of the random-walk offset is drift_scale *
    # texture_sigma (see texture_sigma). Fitted in benchmark/evaluate.py.
    drift_scale: float = 10.91
    texture_half: int = 12  # half-size of the texture window (px)


@dataclasses.dataclass
class TapRaw:
    """Raw per-point TAPNext++ output for one frame (image pixels)."""

    xy: np.ndarray  # [Q, 2]
    visibility: np.ndarray  # [Q]
    sigma_local: np.ndarray  # [Q, 2]
    sigma_global: np.ndarray  # [Q, 2]


class TapFrontEnd:
    """Runs TAPNext++ for all targets and converts outputs to measurements."""

    def __init__(
        self,
        model: TapNextPP,
        targets: list[Target],
        calibration: TapCalibration | None = None,
    ):
        self.model = model
        self.targets = targets
        self.calibration = calibration or TapCalibration()
        self.offsets = [t.support_offsets() for t in targets]
        counts = np.cumsum([0] + [len(o) for o in self.offsets])
        self.slices = [
            slice(a, b) for a, b in zip(counts[:-1], counts[1:], strict=True)
        ]
        self.queries = np.concatenate(
            [t.center + o for t, o in zip(targets, self.offsets, strict=True)]
        )
        self.state = None
        self.size = None

    def step(self, image_rgb: np.ndarray) -> TapRaw:
        """Advance one frame; the first call starts tracking."""
        height, width = image_rgb.shape[:2]
        self.size = (width, height)
        frame = self.model.preprocess(image_rgb)
        if self.state is None:
            out, self.state = self.model.step(
                frame, query_xy=image_to_model(self.queries, width, height)
            )
            q = len(self.queries)
            return TapRaw(
                xy=self.queries.astype(np.float32),
                visibility=np.ones(q, np.float32),
                sigma_local=np.full((q, 2), 0.25, np.float32),
                sigma_global=np.full((q, 2), 0.25, np.float32),
            )
        out, self.state = self.model.step(frame, state=self.state)
        scale = np.array(
            [width / MODEL_COORD_SIZE, height / MODEL_COORD_SIZE], np.float32
        )
        return TapRaw(
            xy=model_to_image(out.xy, width, height),
            visibility=out.visibility,
            sigma_local=out.sigma_local * scale,
            sigma_global=out.sigma_global * scale,
        )

    def measurements(
        self, raw: TapRaw, gray: np.ndarray | None = None
    ) -> list[Measurement | None]:
        """Per-target measurements; ``gray`` (the current frame) enables the
        texture-driven drift model."""
        meas = [
            tap_measurement(
                raw,
                self.slices[k],
                self.offsets[k],
                self.targets[k],
                self.calibration,
            )
            for k in range(len(self.targets))
        ]
        if gray is not None:
            add_texture_drift(meas, gray, self.calibration)
        return meas


def add_texture_drift(meas, gray, calib: TapCalibration) -> None:
    """Set each TAPNext++ measurement's drift rate from the local texture."""
    pts = [m.position for m in meas if m is not None]
    if not pts:
        return
    tex = iter(texture_sigma(gray, np.array(pts), calib.texture_half))
    for m in meas:
        if m is None:
            continue
        sigma = min(next(tex), 1e3)
        m.info["texture_sigma"] = float(sigma)
        m.drift_sigma = calib.drift_scale * sigma


def point_sigma(raw: TapRaw, sel, calib: TapCalibration) -> np.ndarray:
    if calib.sigma_mode == "local":
        sig = raw.sigma_local[sel]
    elif calib.sigma_mode == "global":
        sig = raw.sigma_global[sel]
    elif calib.sigma_mode == "max":
        sig = np.maximum(raw.sigma_local[sel], raw.sigma_global[sel])
    else:
        raise ValueError(f"Unknown sigma_mode {calib.sigma_mode!r}")
    return np.sqrt((calib.sigma_scale * sig) ** 2 + calib.sigma_floor**2)


def tap_measurement(
    raw: TapRaw,
    sel: slice,
    offsets: np.ndarray,
    target: Target,
    calib: TapCalibration,
) -> Measurement | None:
    """One target's TAPNext++ measurement, or None if it is occluded."""
    xy = raw.xy[sel]
    vis = raw.visibility[sel]
    sig = point_sigma(raw, sel, calib)
    visible = vis >= calib.min_visibility
    info = {
        "center_visibility": float(vis[0]),
        "center_sigma": float(np.max(sig[0])),
        "visible_points": int(visible.sum()),
    }
    if not target.is_area and calib.point_mode == "direct" and visible[0]:
        cov = np.diag(sig[0] ** 2)
        fit = fit_similarity(
            offsets[visible],
            xy[visible],
            sig[visible],
            "tap",
            correlation=calib.correlation,
        )
        return Measurement(
            position=xy[0].astype(float),
            cov=cov,
            source="tap",
            transform=None if fit is None else _recenter(fit, xy[0]),
            num_points=1,
            info=info,
        )
    if visible.sum() < (1 if visible[0] else 3):
        return None
    fit = fit_similarity(
        offsets[visible],
        xy[visible],
        sig[visible],
        "tap",
        correlation=calib.correlation,
    )
    if fit is not None:
        fit.info.update(info)
    return fit


def _recenter(fit: Measurement, center: np.ndarray):
    """Keep the fitted rotation/scale but move it to ``center``."""
    if fit.transform is None:
        return None
    transform = fit.transform.copy()
    transform[2:] = center
    return transform
