# SPDX-License-Identifier: BSD-3-Clause
"""Per-frame SIFTAP loop: measure with both trackers, fuse, update refs."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterable

import cv2
import numpy as np

from .fusion import FusedEstimate, make_fusion
from .geometry import Measurement
from .sift import SiftFrontEnd, SiftParams
from .tap import TapCalibration, TapFrontEnd, TapRaw
from .targets import Target

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class TrackerOptions:
    fusion: str = "kalman"
    accel_sigma: float = 1.5  # px / frame^2, constant-velocity process noise
    use_tap: bool = True
    use_sift: bool = True
    initial_sigma: float = 0.5  # px, uncertainty of the user's selection
    # Adaptive SIFT keyframe: re-anchor when the first-frame reference has
    # degraded, the fused estimate is precise, and enough time has passed.
    keyframe_interval: int = 10
    keyframe_max_sigma: float = 1.0
    keyframe_min_inlier_fraction: float = 0.5


@dataclasses.dataclass
class FrameRecord:
    frame: int
    prior: np.ndarray
    tap: Measurement | None
    sift: Measurement | None
    fused: FusedEstimate
    # Similarity (a, b, tx, ty) mapping first-frame offsets around the target
    # to this frame, centred on the fused position (for drawing areas).
    transform: np.ndarray | None


class SiftapTracker:
    def __init__(
        self,
        targets: list[Target],
        tap_model=None,
        options: TrackerOptions | None = None,
        tap_calibration: TapCalibration | None = None,
        sift_params: SiftParams | None = None,
    ):
        self.targets = targets
        self.options = options or TrackerOptions()
        self.tap = TapFrontEnd(tap_model, targets, tap_calibration)
        self.sift = [SiftFrontEnd(t, sift_params) for t in targets]
        self.fusions = []
        self.initial_inliers = [0] * len(targets)
        self.last_keyframe = [0] * len(targets)
        self.records: list[list[FrameRecord]] = [[] for _ in targets]
        self.tap_raw: list[TapRaw] = []

    def run(
        self,
        frames: Iterable[np.ndarray],
        tap_raw: list[TapRaw] | None = None,
    ) -> list[list[FrameRecord]]:
        """Track all targets through ``frames`` (RGB uint8).

        ``tap_raw`` replays cached TAPNext++ outputs (the model is not run),
        which lets several fusion strategies share one network pass.
        """
        tic = time.time()
        for t, image in enumerate(frames):
            raw = None
            if self.options.use_tap:
                raw = (
                    tap_raw[t] if tap_raw is not None else self.tap.step(image)
                )
                self.tap_raw.append(raw)
            self._frame(t, image, raw)
            if (t + 1) % 20 == 0:
                LOGGER.info(
                    "tracked %d frames (%.1fs)", t + 1, time.time() - tic
                )
        return self.records

    def _frame(self, t: int, image: np.ndarray, raw: TapRaw | None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        opts = self.options
        tap_meas = (
            self.tap.measurements(raw, gray)
            if raw is not None and t > 0
            else [None] * len(self.targets)
        )
        for k, target in enumerate(self.targets):
            if t == 0:
                fusion = make_fusion(opts.fusion, **self._fusion_kwargs())
                cov0 = opts.initial_sigma**2 * np.eye(2)
                fusion.reset(target.center, cov0)
                self.fusions.append(fusion)
                if opts.use_sift:
                    self.initial_inliers[k] = self.sift[k].add_reference(
                        gray, target.center, 0
                    )
                fused = FusedEstimate(target.center.copy(), cov0)
                self.records[k].append(
                    FrameRecord(
                        0,
                        target.center.copy(),
                        None,
                        None,
                        fused,
                        np.array([1.0, 0.0, *target.center]),
                    )
                )
                continue
            fusion = self.fusions[k]
            prior, prior_cov = fusion.predict()
            prior_sigma = float(np.sqrt(np.linalg.eigvalsh(prior_cov)[-1]))
            sift_meas = None
            if opts.use_sift:
                sift_meas = self.sift[k].measure(
                    gray, prior, min(prior_sigma, max(gray.shape))
                )
            fused = fusion.update([tap_meas[k], sift_meas])
            transform = _region_transform(tap_meas[k], sift_meas, fused)
            self.records[k].append(
                FrameRecord(t, prior, tap_meas[k], sift_meas, fused, transform)
            )
            if opts.use_sift:
                self._maybe_keyframe(k, t, gray, sift_meas, fused, transform)

    def _fusion_kwargs(self):
        if self.options.fusion.startswith("kalman"):
            return {"accel_sigma": self.options.accel_sigma}
        return {}

    def _maybe_keyframe(self, k, t, gray, sift_meas, fused, transform):
        opts = self.options
        if (
            t - self.last_keyframe[k] < opts.keyframe_interval
            or fused.lost
            or fused.sigma > opts.keyframe_max_sigma
        ):
            return
        initial = max(self.initial_inliers[k], 1)
        degraded = (
            sift_meas is None
            or sift_meas.info.get("reference_frame", 0) != 0
            or sift_meas.num_points
            < opts.keyframe_min_inlier_fraction * min(initial, 40)
        )
        if not degraded:
            return
        scale = 1.0
        if transform is not None:
            scale = float(
                np.clip(np.hypot(transform[0], transform[1]), 0.25, 4)
            )
        added = self.sift[k].add_reference(
            gray, fused.position, t, scale=scale, anchor_cov=fused.cov
        )
        if added:
            self.last_keyframe[k] = t


def _region_transform(tap, sift, fused):
    """Rotation/scale from the more certain source, centred on the fusion."""
    candidates = [
        m for m in (tap, sift) if m is not None and m.transform is not None
    ]
    # Only TAPNext++ transforms are relative to the first frame (SIFT may
    # use an adaptive keyframe), so prefer them for drawing.
    candidates.sort(key=lambda m: (m.source != "tap", np.trace(m.cov)))
    a, b = (1.0, 0.0)
    if candidates:
        a, b = candidates[0].transform[:2]
    return np.array([a, b, *fused.position])
