# SPDX-License-Identifier: BSD-3-Clause
"""SIFT measurements with least-squares covariances.

SIFT cannot measure an arbitrary pixel: it measures where the *keypoints
around* the target went. For each reference frame (the first frame, plus one
adaptive keyframe) the keypoints inside the target neighbourhood are stored
relative to the target. In a new frame:

1. SIFT is run in a search window around the filter's predicted position
   (half-size grows with the predicted uncertainty);
2. reference descriptors are matched with Lowe's ratio test + mutual check;
3. RANSAC estimates a similarity transform, and a weighted LS fit on the
   inliers gives the target position T(0) and its covariance. Uncertainty
   grows with fewer inliers, larger residuals and extrapolation away from
   the keypoints;
4. a mismatch model turns the inlier count and inlier ratio into a
   probability that the whole match is wrong (few inliers, or ambiguity such
   as repetitive windows). It is *not* folded into the covariance: SIFT
   errors are heavy-tailed (sub-pixel when right, tens of pixels when
   wrong), which no single Gaussian describes. The measurement carries
   ``p_outlier`` instead, used by the fusion's mixture update and by the
   displayed 95 % radius.

Adaptive keyframes carry the uncertainty of the position they were anchored
at, which is added to their measurements.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .geometry import Measurement, fit_similarity, texture_sigma
from .targets import Target


@dataclasses.dataclass
class SiftParams:
    contrast_threshold: float = 0.02  # lower than SIFT's 0.04: weak texture
    max_ratio: float = 0.8
    ransac_threshold: float = 3.0
    min_inliers: int = 5
    search_scale: float = 2.0  # window half-size: scale * radius + 3 sigma
    search_margin: float = 16.0
    keypoint_sigma: float = 0.35  # px, per-keypoint localization floor
    keypoint_sigma_per_size: float = 0.02  # + per pixel of keypoint size
    # Calibrated (benchmark/evaluate.py): with the texture term available the
    # LS covariance gets almost no weight; the floor and the texture term
    # predict SIFT's localization error better than the fit residuals.
    sigma_scale: float = 0.10  # multiplies the LS covariance std
    sigma_floor: float = 0.50  # px, added in quadrature
    # + (texture_scale * texture_sigma)^2: keypoints on weak texture
    # localize worse than the LS covariance assumes (benchmark: ground).
    texture_scale: float = 35.2
    texture_half: int = 12
    # Mismatch model: P(gross error) = sigmoid(bias + w_inliers *
    # log(inliers) + w_ratio * inlier_ratio); gross errors have std
    # gross_sigma per axis. Defaults are fitted on the synthetic calibration
    # sequence (benchmark/evaluate.py --stage calibrate).
    gross_bias: float = 10.885
    gross_w_inliers: float = -4.389
    gross_w_ratio: float = -5.338
    gross_sigma: float = 37.5
    min_reference_keypoints: int = 6


@dataclasses.dataclass
class _Reference:
    offsets: np.ndarray  # [K, 2] keypoints relative to the target
    descriptors: np.ndarray  # [K, 128] RootSIFT
    sizes: np.ndarray  # [K]
    anchor_cov: np.ndarray  # [2, 2] uncertainty of the anchor position
    radius: float
    frame: int


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    d = descriptors / np.maximum(descriptors.sum(1, keepdims=True), 1e-8)
    return np.sqrt(d).astype(np.float32)


class SiftFrontEnd:
    """SIFT measurement model for one target."""

    def __init__(self, target: Target, params: SiftParams | None = None):
        self.target = target
        self.params = params or SiftParams()
        self.sift = cv2.SIFT_create(
            contrastThreshold=self.params.contrast_threshold
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references: list[_Reference] = []

    # -- references -------------------------------------------------------
    def add_reference(
        self,
        gray: np.ndarray,
        center: np.ndarray,
        frame: int,
        scale: float = 1.0,
        anchor_cov: np.ndarray | None = None,
        keep_first: bool = True,
    ) -> int:
        """Store the keypoints around ``center``; returns their number.

        The first reference is kept forever (no drift); later calls replace
        the adaptive keyframe.
        """
        target = dataclasses.replace(self.target, center=np.asarray(center))
        if target.box is not None:
            half = 0.5 * (target.box[2:] - target.box[:2])
            target.box = np.r_[center - half, center + half]
        radius = target.radius * scale
        x0, y0, x1, y1 = _window(gray.shape, center, radius + 2)
        crop = gray[y0:y1, x0:x1]
        mask = target.region_mask(gray.shape, scale)[y0:y1, x0:x1]
        keypoints, desc = self.sift.detectAndCompute(
            crop, mask.astype(np.uint8) * 255
        )
        if desc is None or len(keypoints) < self.params.min_reference_keypoints:
            return 0 if desc is None else len(keypoints)
        xy = np.array([kp.pt for kp in keypoints]) + [x0 + 0.5, y0 + 0.5]
        ref = _Reference(
            offsets=xy - center,
            descriptors=root_sift(desc),
            sizes=np.array([kp.size for kp in keypoints]),
            anchor_cov=np.zeros((2, 2)) if anchor_cov is None else anchor_cov,
            radius=radius,
            frame=frame,
        )
        if keep_first and self.references:
            self.references = [self.references[0], ref]
        else:
            self.references = [ref]
        return len(keypoints)

    @property
    def has_reference(self) -> bool:
        return bool(self.references)

    # -- measurement ------------------------------------------------------
    def measure(
        self,
        gray: np.ndarray,
        predicted: np.ndarray,
        predicted_sigma: float,
    ) -> Measurement | None:
        if not self.references:
            return None
        p = self.params
        best = None
        for ref in self.references:
            half = p.search_scale * ref.radius + 3 * predicted_sigma
            half = min(half + p.search_margin, max(gray.shape))
            x0, y0, x1, y1 = _window(gray.shape, predicted, half)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            keypoints, desc = self.sift.detectAndCompute(
                gray[y0:y1, x0:x1], None
            )
            if desc is None or len(keypoints) < p.min_inliers:
                continue
            xy = np.array([kp.pt for kp in keypoints]) + [x0 + 0.5, y0 + 0.5]
            meas = self._match(ref, root_sift(desc), xy)
            if meas is not None and (
                best is None or meas.num_points > best.num_points
            ):
                best = meas
        if best is not None:
            tex = float(texture_sigma(gray, best.position, p.texture_half)[0])
            tex = min(tex, 1e3)
            best.cov = best.cov + (p.texture_scale * tex) ** 2 * np.eye(2)
            best.info["texture_sigma"] = tex
        return best

    def _match(self, ref: _Reference, desc, xy) -> Measurement | None:
        p = self.params
        if len(desc) < 2 or len(ref.descriptors) < 2:
            return None
        knn = self.matcher.knnMatch(ref.descriptors, desc, k=2)
        back = self.matcher.match(desc, ref.descriptors)
        back_of = {m.queryIdx: m.trainIdx for m in back}
        pairs = [
            (m.queryIdx, m.trainIdx)
            for m, n in (k for k in knn if len(k) == 2)
            if m.distance < p.max_ratio * n.distance
            and back_of.get(m.trainIdx) == m.queryIdx
        ]
        if len(pairs) < p.min_inliers:
            return None
        ri, ci = np.array(pairs).T
        src, dst = ref.offsets[ri], xy[ci]
        _, inliers = cv2.estimateAffinePartial2D(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=p.ransac_threshold,
            confidence=0.995,
        )
        if inliers is None:
            return None
        inl = inliers.ravel().astype(bool)
        if inl.sum() < p.min_inliers:
            return None
        sigmas = (
            p.keypoint_sigma + p.keypoint_sigma_per_size * ref.sizes[ri][inl]
        )
        fit = fit_similarity(src[inl], dst[inl], sigmas, "sift")
        if fit is None:
            return None
        ratio = inl.sum() / len(pairs)
        cov = (p.sigma_scale**2) * fit.cov + (p.sigma_floor**2) * np.eye(2)
        p_gross = gross_probability(int(inl.sum()), float(ratio), p)
        fit.cov = cov + ref.anchor_cov
        fit.p_outlier = p_gross
        fit.outlier_sigma = p.gross_sigma
        fit.info.update(
            {
                "inliers": int(inl.sum()),
                "matches": len(pairs),
                "inlier_ratio": float(ratio),
                "p_gross": float(p_gross),
                "reference_frame": ref.frame,
            }
        )
        fit.num_points = int(inl.sum())
        return fit


def gross_probability(inliers: int, ratio: float, p: SiftParams) -> float:
    z = (
        p.gross_bias
        + p.gross_w_inliers * np.log(max(inliers, 1))
        + (p.gross_w_ratio * ratio)
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def _window(shape, center, half):
    height, width = shape[:2]
    x0 = int(max(0, np.floor(center[0] - half)))
    y0 = int(max(0, np.floor(center[1] - half)))
    x1 = int(min(width, np.ceil(center[0] + half)))
    y1 = int(min(height, np.ceil(center[1] + half)))
    return x0, y0, max(x0, x1), max(y0, y1)
