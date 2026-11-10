# SPDX-License-Identifier: BSD-3-Clause
"""Fusing SIFT and TAPNext++ measurements.

All strategies consume, per frame, a list of Gaussian measurements (position
+ covariance, possibly missing) and return a fused position + covariance.

* ``tap`` / ``sift``: a single source, no temporal model (baselines).
* ``best``: per frame, the source with the smaller covariance.
* ``ivw``: inverse-variance weighting (product of Gaussians), i.e. the
  static Bayesian fusion of independent measurements.
  This naive product uses each source's core covariance and trusts it fully.
* ``bayes``: robust static Bayesian fusion, no motion model. The most
  reliable measurement (lowest outlier probability, then smallest
  covariance) is the prior; the others are combined with a chi-square gate
  and the outlier-mixture update described below.
* ``kalman`` (default): constant-velocity Kalman filter. Each measurement is
  checked against the prediction with a chi-square gate on its Mahalanobis
  distance before it is used (rejecting e.g. a SIFT match on the wrong
  window of a repetitive facade). A measurement that passes is combined with
  the outlier mixture: the posterior probability w that it is an inlier
  (from its own p_outlier and how well it agrees with the prediction) mixes
  the Kalman update and the prior, P = w P_upd + (1 - w) P_prior +
  w (1 - w) d d^T (the one-measurement PDA filter). Accepted measurements
  update the state with their own covariance, so the more certain source
  dominates. TAPNext++'s slowly wandering error is a drift state that SIFT
  (no drift: it matches back to the reference frame) makes observable. If
  every available measurement is rejected for ``reacquire_after`` frames in
  a row (the filter lost the target, e.g. after an occlusion), the filter is
  re-initialized from them.
* ``kalman_nodrift``: as ``kalman`` but TAPNext++ errors are treated as
  white noise (no drift state).
* ``kalman_ci``: as ``kalman``, but SIFT and TAPNext++ are first combined by
  covariance intersection, which stays consistent if their errors are
  correlated (both look at the same pixels).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import (
    CHI2_2DOF_997,
    Measurement,
    inlier_posterior,
    mahalanobis2,
    mixture_cdf,
    mixture_radius,
)

LOST_COV = 1e8


@dataclasses.dataclass
class FusedEstimate:
    position: np.ndarray
    cov: np.ndarray
    accepted: dict = dataclasses.field(default_factory=dict)
    nis: dict = dataclasses.field(default_factory=dict)
    reacquired: bool = False
    weights: dict = dataclasses.field(default_factory=dict)  # inlier posterior
    p_outlier: float = 0.0  # only for single-source estimates
    outlier_sigma: float = 0.0
    drift: np.ndarray | None = None  # estimated TAPNext++ drift (px)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov)[-1], 0.0)))

    def radius(self, level: float = 0.95) -> float:
        return mixture_radius(
            self.sigma, self.p_outlier, self.outlier_sigma, level
        )

    def confidence(self, tolerance: float) -> float:
        """P(|error| <= tolerance): how likely the estimate is within
        ``tolerance`` px of the truth (0 when lost)."""
        if self.lost:
            return 0.0
        return mixture_cdf(
            tolerance, self.sigma, self.p_outlier, self.outlier_sigma
        )

    @property
    def lost(self) -> bool:
        return self.sigma**2 >= LOST_COV


def inverse_variance(measurements: list[Measurement]):
    info = sum(np.linalg.inv(m.cov) for m in measurements)
    cov = np.linalg.inv(info)
    pos = cov @ sum(np.linalg.solve(m.cov, m.position) for m in measurements)
    return pos, cov


def covariance_intersection(a: Measurement, b: Measurement, steps: int = 21):
    """CI fusion: consistent for any (unknown) correlation between a and b."""
    ia, ib = np.linalg.inv(a.cov), np.linalg.inv(b.cov)
    best = None
    for w in np.linspace(0.0, 1.0, steps):
        cov = np.linalg.inv(w * ia + (1 - w) * ib + 1e-12 * np.eye(2))
        if best is None or np.trace(cov) < best[0]:
            pos = cov @ (w * ia @ a.position + (1 - w) * ib @ b.position)
            best = (np.trace(cov), pos, cov)
    return best[1], best[2]


class Fusion:
    """Base class: keeps the last estimate, no motion model."""

    name = "base"

    def reset(self, position: np.ndarray, cov: np.ndarray) -> None:
        self.position = np.asarray(position, float).copy()
        self.cov = np.asarray(cov, float).copy()

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prior for the next frame (used for SIFT's search window)."""
        return self.position, self.cov

    def update(self, measurements: list[Measurement | None]) -> FusedEstimate:
        raise NotImplementedError

    def _hold(self) -> FusedEstimate:
        # No measurement: keep the position, report it as lost, and let the
        # next SIFT search cover the whole image.
        self.cov = LOST_COV * np.eye(2)
        return FusedEstimate(self.position.copy(), self.cov.copy())


class SingleSource(Fusion):
    def __init__(self, source: str):
        self.source = source
        self.name = source

    def update(self, measurements):
        meas = [
            m for m in measurements if m is not None and m.source == self.source
        ]
        if not meas:
            return self._hold()
        m = meas[0]
        self.position, self.cov = m.position, m.cov
        return FusedEstimate(
            self.position.copy(),
            self.cov.copy(),
            {self.source: True},
            p_outlier=m.p_outlier,
            outlier_sigma=m.outlier_sigma,
        )


class BestOf(Fusion):
    name = "best"

    def update(self, measurements):
        meas = [m for m in measurements if m is not None]
        if not meas:
            return self._hold()
        m = min(meas, key=lambda m: m.radius())
        self.position, self.cov = m.position, m.cov
        return FusedEstimate(
            self.position.copy(),
            self.cov.copy(),
            {x.source: x is m for x in meas},
            p_outlier=m.p_outlier,
            outlier_sigma=m.outlier_sigma,
        )


class InverseVariance(Fusion):
    name = "ivw"

    def update(self, measurements):
        meas = [m for m in measurements if m is not None]
        if not meas:
            return self._hold()
        self.position, self.cov = inverse_variance(meas)
        return FusedEstimate(
            self.position.copy(),
            self.cov.copy(),
            {m.source: True for m in meas},
        )


class KalmanFusion(Fusion):
    """Constant-velocity Kalman filter with gated sequential updates.

    State: position (2), velocity (2) and, with ``drift=True``, the drift
    (bias) of drifting sources (2). TAPNext++ tracks frame to frame, so its
    error is mostly a slowly wandering offset (lag-5 autocorrelation
    0.9+ on sky and ground in the benchmark) rather than white noise; it is
    modelled as a random walk b_t = b_{t-1} + w, w ~ N(0, drift_sigma^2 I),
    and TAPNext++ measures position + b. SIFT matches back to a fixed
    reference frame and does not drift, so whenever SIFT is available the
    drift is observable and corrected (like GPS correcting an IMU); without
    SIFT, the position uncertainty grows with the accumulated drift.
    """

    name = "kalman"

    def __init__(
        self,
        accel_sigma: float = 1.5,
        gate: float = CHI2_2DOF_997,
        reacquire_after: int = 2,
        use_gate: bool = True,
        covariance_intersection: bool = False,
        initial_velocity_sigma: float = 20.0,
        reject_inflation: float = 0.25,
        drift: bool = True,
    ):
        self.accel_sigma = accel_sigma
        self.gate = gate
        self.reacquire_after = reacquire_after
        self.reject_inflation = reject_inflation
        self.use_gate = use_gate
        self.ci = covariance_intersection
        self.drift = drift and not covariance_intersection
        self.initial_velocity_sigma = initial_velocity_sigma
        if covariance_intersection:
            self.name = "kalman_ci"
        elif not use_gate:
            self.name = "kalman_nogate"
        elif not drift:
            self.name = "kalman_nodrift"
        self.dim = 6 if self.drift else 4
        self.drift_sigma = 0.0

    def reset(self, position, cov, velocity=None, keep_drift=False):
        old_x, old_P = getattr(self, "x", None), getattr(self, "P", None)
        self.x = np.zeros(self.dim)
        self.x[:2] = position
        if velocity is not None:
            self.x[2:4] = velocity
        self.P = np.zeros((self.dim, self.dim))
        self.P[:2, :2] = cov
        self.P[2:4, 2:4] = self.initial_velocity_sigma**2 * np.eye(2)
        if keep_drift and self.drift and old_x is not None:
            self.x[4:], self.P[4:, 4:] = old_x[4:], old_P[4:, 4:]
        self.rejected_streak = 0
        self.position, self.cov = self.x[:2].copy(), self.P[:2, :2].copy()

    def predict(self):
        F = np.eye(self.dim)
        F[0, 2] = F[1, 3] = 1.0
        q = self.accel_sigma**2
        Q = np.zeros((self.dim, self.dim))
        for i in range(2):
            Q[i, i], Q[i, i + 2] = q / 4, q / 2
            Q[i + 2, i], Q[i + 2, i + 2] = q / 2, q
        if self.drift:
            Q[4:, 4:] = self.drift_sigma**2 * np.eye(2)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy(), self.P[:2, :2].copy()

    def _H(self, m: Measurement) -> np.ndarray:
        H = np.zeros((2, self.dim))
        H[0, 0] = H[1, 1] = 1.0
        if self.drift and m.drifts:
            H[0, 4] = H[1, 5] = 1.0
        return H

    def _update_one(self, m: Measurement) -> tuple[bool, float, float]:
        H = self._H(m)
        innovation = m.position - H @ self.x
        S = H @ self.P @ H.T + m.cov
        nis = mahalanobis2(innovation, S)
        w = inlier_posterior(innovation, S, m.p_outlier, m.outlier_sigma)
        if self.use_gate and (nis > self.gate or w < 0.5):
            # A rejected measurement still says the filter may be wrong:
            # widen the position uncertainty by part of the disagreement, so
            # the reported radius grows while the sources conflict, and
            # consistent disagreement eventually passes the gate.
            self.P[:2, :2] += self.reject_inflation * np.outer(
                innovation, innovation
            )
            return False, nis, w
        K = self.P @ H.T @ np.linalg.inv(S)
        x_upd = self.x + K @ innovation
        # Joseph form keeps P symmetric positive definite.
        IKH = np.eye(self.dim) - K @ H
        P_upd = IKH @ self.P @ IKH.T + K @ m.cov @ K.T
        if self.use_gate and w < 1.0:
            # Collapse the inlier/outlier mixture to one Gaussian.
            d = x_upd - self.x
            P_upd = w * P_upd + (1 - w) * self.P + w * (1 - w) * np.outer(d, d)
            x_upd = self.x + w * d
        self.x, self.P = x_upd, P_upd
        return True, nis, w

    def update(self, measurements):
        meas = [m for m in measurements if m is not None]
        result = FusedEstimate(self.x[:2].copy(), self.P[:2, :2].copy())
        if self.ci and len(meas) == 2:
            pos, cov = covariance_intersection(meas[0], meas[1])
            worst = max(meas, key=lambda m: m.p_outlier)
            meas = [
                Measurement(
                    pos,
                    cov,
                    "fused",
                    p_outlier=worst.p_outlier,
                    outlier_sigma=worst.outlier_sigma,
                )
            ]
        for m in meas:
            if m.drifts and not m.occluded:
                # Texture under an occluded point is the occluder's: keep
                # the last rate. Used by the next predict.
                self.drift_sigma = m.drift_sigma
        # Most reliable first: it best constrains the gate for the others.
        meas.sort(key=lambda m: np.trace(m.moment_cov()))
        for m in meas:
            ok, nis, w = self._update_one(m)
            result.accepted[m.source] = ok
            result.nis[m.source] = nis
            result.weights[m.source] = w
        if meas and not any(result.accepted.values()):
            self.rejected_streak += 1
        else:
            self.rejected_streak = 0
        if self.rejected_streak >= self.reacquire_after:
            # Re-acquire: trust the measurements again (e.g. TAPNext++
            # re-detecting the point after an occlusion). A drifting source
            # is de-biased with the current drift estimate and its
            # uncertainty.
            fixed = []
            for m in meas:
                if self.drift and m.drifts:
                    m = dataclasses.replace(
                        m,
                        position=m.position - self.x[4:],
                        cov=m.cov + self.P[4:, 4:],
                    )
                fixed.append(m)
            pos, cov = inverse_variance(fixed)
            self.reset(pos, cov, keep_drift=True)
            result.reacquired = True
            result.accepted = {m.source: True for m in meas}
        self.position, self.cov = self.x[:2].copy(), self.P[:2, :2].copy()
        result.position, result.cov = self.position.copy(), self.cov.copy()
        if self.drift:
            result.drift = self.x[4:].copy()
        return result


class RobustBayes(KalmanFusion):
    """Per-frame robust Bayesian fusion: KalmanFusion without memory."""

    name = "bayes"

    def __init__(self, **kwargs):
        super().__init__(drift=False, **kwargs)
        self.name = "bayes"

    def predict(self):
        return self.position, self.cov

    def update(self, measurements):
        meas = [m for m in measurements if m is not None]
        if not meas:
            return self._hold()
        meas.sort(key=lambda m: (m.p_outlier, np.trace(m.cov)))
        first = meas[0]
        self.reset(first.position, first.cov)
        result = FusedEstimate(
            first.position.copy(),
            first.cov.copy(),
            {first.source: True},
            weights={first.source: 1.0 - first.p_outlier},
        )
        for m in meas[1:]:
            ok, nis, w = self._update_one(m)
            result.accepted[m.source] = ok
            result.nis[m.source] = nis
            result.weights[m.source] = w
        self.position, self.cov = self.x[:2].copy(), self.P[:2, :2].copy()
        result.position, result.cov = self.position.copy(), self.cov.copy()
        if len(meas) == 1:
            result.p_outlier = first.p_outlier
            result.outlier_sigma = first.outlier_sigma
        return result


def make_fusion(name: str, **kwargs) -> Fusion:
    if name in ("tap", "sift"):
        return SingleSource(name)
    if name == "best":
        return BestOf()
    if name == "ivw":
        return InverseVariance()
    if name == "bayes":
        return RobustBayes(
            **{k: v for k, v in kwargs.items() if k != "accel_sigma"}
        )
    if name == "kalman":
        return KalmanFusion(**kwargs)
    if name == "kalman_nogate":
        return KalmanFusion(use_gate=False, **kwargs)
    if name == "kalman_nodrift":
        return KalmanFusion(drift=False, **kwargs)
    if name == "kalman_ci":
        return KalmanFusion(covariance_intersection=True, **kwargs)
    raise ValueError(f"Unknown fusion {name!r}")


FUSIONS = (
    "tap",
    "sift",
    "best",
    "ivw",
    "bayes",
    "kalman_nodrift",
    "kalman",
    "kalman_nogate",
    "kalman_ci",
)
