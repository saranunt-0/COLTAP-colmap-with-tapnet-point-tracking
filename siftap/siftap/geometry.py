# SPDX-License-Identifier: BSD-3-Clause
"""Gaussian measurements and robust local similarity fits.

Both trackers observe several points around the target (SIFT keypoints in the
target region, TAPNext++ support points). A 2D similarity transform
(scale, rotation, translation) is fitted from their reference positions to
their current positions, expressed relative to the target, so the fitted
translation *is* the target position and its covariance follows directly
from the weighted least-squares fit.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

# Chi-square quantiles with 2 degrees of freedom.
CHI2_2DOF_95 = 5.991
CHI2_2DOF_99 = 9.210
CHI2_2DOF_997 = 11.618


@dataclasses.dataclass
class Measurement:
    """A 2D position estimate with a covariance (image pixels)."""

    position: np.ndarray  # [2]
    cov: np.ndarray  # [2, 2]
    source: str = ""
    # Similarity transform from the reference frame, relative to the target
    # (a, b, tx, ty): x' = [[a, -b], [b, a]] x + t. None if not estimated.
    transform: np.ndarray | None = None
    num_points: int = 0
    info: dict = dataclasses.field(default_factory=dict)
    # Heavy tail: with probability p_outlier the measurement is a gross error
    # (e.g. SIFT matched the wrong window), modelled as an isotropic Gaussian
    # of std outlier_sigma around the truth. ``cov`` is the inlier
    # (core) covariance.
    p_outlier: float = 0.0
    outlier_sigma: float = 0.0
    # A drifting source (TAPNext++) measures position + drift; the drift
    # does a random walk with std drift_sigma per frame.
    drifts: bool = False
    drift_sigma: float = 0.0
    # The target itself is hidden (TAPNext++ occlusion head); the position
    # is the tracker's estimate of where it is behind the occluder.
    occluded: bool = False

    @property
    def sigma(self) -> float:
        """Core standard deviation along the most uncertain direction."""
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov)[-1], 0.0)))

    def radius(self, level: float = 0.95) -> float:
        """Radius containing the true position with probability ``level``
        under the core + outlier mixture (the "GPS" radius)."""
        return mixture_radius(
            self.sigma, self.p_outlier, self.outlier_sigma, level
        )

    def confidence(self, tolerance: float) -> float:
        """P(|error| <= tolerance) under the core + outlier mixture."""
        return mixture_cdf(
            tolerance, self.sigma, self.p_outlier, self.outlier_sigma
        )

    def moment_cov(self) -> np.ndarray:
        """Single-Gaussian (moment-matched) covariance of the mixture."""
        p = self.p_outlier
        return (1 - p) * self.cov + p * self.outlier_sigma**2 * np.eye(2)

    def map(self, offsets: np.ndarray) -> np.ndarray:
        """Map reference offsets (relative to the target) to this frame."""
        offsets = np.asarray(offsets, float).reshape(-1, 2)
        if self.transform is None:
            return offsets + self.position
        a, b, tx, ty = self.transform
        rot = np.array([[a, -b], [b, a]])
        return offsets @ rot.T + np.array([tx, ty])


def mixture_cdf(
    r: float, sigma: float, p_outlier: float = 0.0, outlier_sigma: float = 0.0
) -> float:
    """P(|error| <= r) for an isotropic 2D Gaussian core (std ``sigma``)
    mixed with an outlier Gaussian (largest core std: conservative)."""

    def inside(s):
        return 1.0 - np.exp(-0.5 * (r / max(s, 1e-9)) ** 2)

    if p_outlier <= 0.0 or outlier_sigma <= 0.0:
        return float(inside(sigma))
    return float(
        (1 - p_outlier) * inside(sigma) + p_outlier * inside(outlier_sigma)
    )


def mixture_radius(
    sigma: float,
    p_outlier: float = 0.0,
    outlier_sigma: float = 0.0,
    level: float = 0.95,
) -> float:
    """Radius r with P(|error| <= r) = level for an isotropic 2D Gaussian
    core (std ``sigma``) mixed with an outlier Gaussian. Uses the core's
    largest std, so it is conservative for elongated ellipses."""

    def inside(r, s):
        return 1.0 - np.exp(-0.5 * (r / max(s, 1e-9)) ** 2)

    if p_outlier <= 0.0 or outlier_sigma <= 0.0:
        return float(sigma * np.sqrt(-2.0 * np.log(1.0 - level)))
    lo, hi = 0.0, 10.0 * max(sigma, outlier_sigma)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mass = (1 - p_outlier) * inside(mid, sigma) + p_outlier * inside(
            mid, outlier_sigma
        )
        lo, hi = (mid, hi) if mass < level else (lo, mid)
    return float(hi)


def inlier_posterior(
    innovation: np.ndarray,
    S: np.ndarray,
    p_outlier: float,
    outlier_sigma: float,
) -> float:
    """P(measurement is an inlier | innovation) for the mixture model."""
    if p_outlier <= 0.0:
        return 1.0
    S_out = S + outlier_sigma**2 * np.eye(2)

    def loglik(cov):
        return -0.5 * (
            mahalanobis2(innovation, cov) + np.log(np.linalg.det(cov))
        )

    a = np.log1p(-p_outlier) + loglik(S)
    b = np.log(p_outlier) + loglik(S_out)
    return float(1.0 / (1.0 + np.exp(np.clip(b - a, -50, 50))))


def texture_sigma(
    gray: np.ndarray, xy: np.ndarray, half: int = 12
) -> np.ndarray:
    """1 / sqrt(smallest eigenvalue of the structure tensor) around each
    point ((2 half + 1)^2 window, Sobel gradients in grey levels / px).

    For unit image noise this is the Cramer-Rao bound (px) of aligning the
    patch along its weakest direction: tiny on corners and text, large on
    sky, haze and smooth walls. Points outside the image get inf.
    """
    height, width = gray.shape[:2]
    out = np.full(len(xy), np.inf)
    for i, (x, y) in enumerate(np.asarray(xy, float).reshape(-1, 2)):
        col, row = int(np.floor(x)), int(np.floor(y))
        if not (0 <= col < width and 0 <= row < height):
            continue
        r0, r1 = max(row - half - 1, 0), min(row + half + 2, height)
        c0, c1 = max(col - half - 1, 0), min(col + half + 2, width)
        patch = gray[r0:r1, c0:c1].astype(np.float32)
        if min(patch.shape) < 3:
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8
        a, b, c = (gx * gx).sum(), (gx * gy).sum(), (gy * gy).sum()
        lmin = 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b * b))
        out[i] = 1.0 / np.sqrt(max(float(lmin), 1e-6))
    return out


def mahalanobis2(delta: np.ndarray, cov: np.ndarray) -> float:
    return float(delta @ np.linalg.solve(cov, delta))


def fit_similarity(
    ref_offsets: np.ndarray,
    observed: np.ndarray,
    sigmas: np.ndarray,
    source: str = "",
    min_points_similarity: int = 3,
    min_spread: float = 4.0,
    huber: float = 2.0,
    iterations: int = 5,
    correlation: float = 0.0,
) -> Measurement | None:
    """Robust weighted similarity (or translation) fit.

    Args:
        ref_offsets: [N, 2] reference positions relative to the target.
        observed: [N, 2] current positions (image pixels).
        sigmas: [N] or [N, 2] per-point standard deviations.
        min_points_similarity / min_spread: below these, fall back to a
            translation-only model (rotation and scale are not observable).
        huber: Huber threshold in sigmas for iteratively reweighted LS.
        correlation: average error correlation between the points. The
            covariance is inflated by 1 + correlation * (N_eff - 1) because
            errors of nearby points are not independent (measured for
            TAPNext++ support points).

    Returns the target position T(0) and its covariance, scaled up by the
    reduced chi-square of the fit when the points disagree more than their
    sigmas claim. None if no point is given.
    """
    ref = np.asarray(ref_offsets, float).reshape(-1, 2)
    obs = np.asarray(observed, float).reshape(-1, 2)
    n = len(ref)
    if n == 0:
        return None
    sig = np.asarray(sigmas, float)
    sig = np.broadcast_to(sig.reshape(n, -1), (n, 2)).copy()
    sig = np.maximum(sig, 1e-3)
    spread = np.sqrt(((ref - ref.mean(0)) ** 2).sum(1).mean())
    similarity = n >= min_points_similarity and spread >= min_spread

    if similarity:
        design = np.zeros((2 * n, 4))
        design[0::2] = np.c_[ref[:, 0], -ref[:, 1], np.ones(n), np.zeros(n)]
        design[1::2] = np.c_[ref[:, 1], ref[:, 0], np.zeros(n), np.ones(n)]
        target = obs.reshape(-1)
    else:
        design = np.tile(np.eye(2), (n, 1))
        target = (obs - ref).reshape(-1)
    base_w = 1.0 / sig.reshape(-1) ** 2
    robust_w = np.ones(2 * n)
    for _ in range(iterations):
        w = base_w * robust_w
        normal = design.T @ (design * w[:, None])
        params = np.linalg.solve(normal, design.T @ (w * target))
        resid = (target - design @ params).reshape(n, 2) / sig
        norm = np.sqrt((resid**2).sum(1))
        point_w = np.where(norm <= huber, 1.0, huber / np.maximum(norm, 1e-9))
        robust_w = np.repeat(point_w, 2)
    w = base_w * robust_w
    normal = design.T @ (design * w[:, None])
    cov_params = np.linalg.inv(normal)
    dof = 2 * n - len(params)
    chi2 = float((w * (target - design @ params) ** 2).sum())
    scale = max(1.0, chi2 / dof) if dof > 0 else 1.0
    n_eff = float(robust_w.reshape(n, 2)[:, 0].sum())
    scale *= 1.0 + correlation * max(n_eff - 1.0, 0.0)

    if similarity:
        a, b, tx, ty = params
        position = np.array([tx, ty])
        cov = cov_params[2:, 2:] * scale
        transform = params.copy()
    else:
        position = params.copy()
        cov = cov_params * scale
        transform = None
    return Measurement(
        position=position,
        cov=0.5 * (cov + cov.T),
        source=source,
        transform=transform,
        num_points=n,
        info={"chi2_reduced": chi2 / dof if dof > 0 else 0.0, "n_eff": n_eff},
    )


def inflate(cov: np.ndarray, factor: float = 1.0, floor: float = 0.0):
    """factor * cov + floor^2 * I."""
    return factor * cov + (floor**2) * np.eye(2)
