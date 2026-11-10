# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from siftap.fusion import KalmanFusion, inverse_variance, make_fusion
from siftap.geometry import (
    CHI2_2DOF_95,
    Measurement,
    fit_similarity,
    inlier_posterior,
    mixture_radius,
)
from siftap.targets import Target, parse_target


def _similarity(points, scale=1.1, angle=0.2, shift=(5.0, -3.0)):
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return points @ np.array([[c, -s], [s, c]]).T + np.array(shift)


def test_fit_recovers_target_position_and_shrinks_with_points():
    rng = np.random.default_rng(0)
    ref = rng.uniform(-20, 20, (40, 2))
    obs = _similarity(ref) + rng.normal(0, 0.5, ref.shape)
    fit = fit_similarity(ref, obs, np.full(40, 0.5))
    np.testing.assert_allclose(fit.position, [5.0, -3.0], atol=0.5)
    few = fit_similarity(ref[:5], obs[:5], np.full(5, 0.5))
    assert np.trace(few.cov) > np.trace(fit.cov)
    # The fitted transform maps the reference offsets onto the observations.
    np.testing.assert_allclose(fit.map(ref), obs, atol=2.0)


def test_fit_downweights_outliers_and_inflates_on_misfit():
    rng = np.random.default_rng(1)
    ref = rng.uniform(-20, 20, (30, 2))
    obs = _similarity(ref) + rng.normal(0, 0.3, ref.shape)
    clean = fit_similarity(ref, obs, np.full(30, 0.3))
    obs[:4] += 40.0  # gross outliers
    robust = fit_similarity(ref, obs, np.full(30, 0.3))
    np.testing.assert_allclose(robust.position, clean.position, atol=1.0)
    assert np.trace(robust.cov) >= np.trace(clean.cov)


def test_fit_falls_back_to_translation_for_one_point():
    fit = fit_similarity(np.zeros((1, 2)), np.array([[3.0, 4.0]]), [1.0])
    np.testing.assert_allclose(fit.position, [3.0, 4.0])
    np.testing.assert_allclose(fit.cov, np.eye(2))
    assert fit.transform is None


def test_correlation_inflates_covariance():
    rng = np.random.default_rng(2)
    ref = rng.uniform(-20, 20, (17, 2))
    obs = ref + 2.0
    a = fit_similarity(ref, obs, np.full(17, 1.0), correlation=0.0)
    b = fit_similarity(ref, obs, np.full(17, 1.0), correlation=0.3)
    assert np.trace(b.cov) / np.trace(a.cov) == pytest.approx(1 + 0.3 * 16)


def test_inverse_variance_trusts_the_certain_source():
    a = Measurement(np.array([0.0, 0.0]), np.eye(2) * 0.01, "sift")
    b = Measurement(np.array([10.0, 0.0]), np.eye(2) * 100.0, "tap")
    pos, cov = inverse_variance([a, b])
    assert pos[0] < 0.01
    assert np.trace(cov) < np.trace(a.cov)


def test_kalman_gate_rejects_outlier_and_follows_motion():
    kf = KalmanFusion(accel_sigma=0.5)
    kf.reset(np.zeros(2), np.eye(2) * 0.25)
    for t in range(1, 20):
        kf.predict()
        z = Measurement(np.array([2.0 * t, 0.0]), np.eye(2) * 0.25, "tap")
        est = kf.update([z])
    assert est.accepted["tap"]
    np.testing.assert_allclose(est.position, [38.0, 0.0], atol=0.5)
    prior, _ = kf.predict()
    np.testing.assert_allclose(prior, [40.0, 0.0], atol=0.6)  # velocity
    wrong = Measurement(np.array([40.0, 60.0]), np.eye(2) * 0.25, "sift")
    good = Measurement(np.array([40.0, 0.0]), np.eye(2) * 0.25, "tap")
    est = kf.update([wrong, good])
    assert est.accepted == {"tap": True, "sift": False}
    assert est.nis["sift"] > CHI2_2DOF_95


def test_kalman_converges_after_a_jump_and_reports_the_conflict():
    """A confident source suddenly disagrees (e.g. re-detection after an
    occlusion): the radius must cover the disagreement while it lasts, and
    the filter must settle on the new position within a few frames."""
    kf = KalmanFusion(accel_sigma=0.5, reacquire_after=2)
    kf.reset(np.zeros(2), np.eye(2) * 0.25)
    z = Measurement(np.array([80.0, 0.0]), np.eye(2), "tap")
    for frame in range(6):
        kf.predict()
        est = kf.update([z])
        error = np.linalg.norm(est.position - z.position)
        if error > 5.0:
            # Honest: the 95 % radius includes the conflicting measurement
            # or keeps growing towards it.
            assert est.sigma * np.sqrt(CHI2_2DOF_95) > 0.3 * error, frame
    np.testing.assert_allclose(kf.position, [80.0, 0.0], atol=1.0)
    assert est.sigma < 2.0


@pytest.mark.parametrize(
    "name",
    ["tap", "sift", "best", "ivw", "kalman", "kalman_nogate", "kalman_ci"],
)
def test_all_fusions_hold_when_no_measurement(name):
    f = make_fusion(name)
    f.reset(np.array([5.0, 5.0]), np.eye(2))
    f.predict()
    est = f.update([None, None])
    np.testing.assert_allclose(est.position, [5.0, 5.0], atol=1e-6)


def test_mixture_radius_covers_95_percent():
    rng = np.random.default_rng(3)
    for p in (0.0, 0.02, 0.2):
        n = 200_000
        out = rng.random(n) < p
        sig = np.where(out, 40.0, 0.5)[:, None]
        err = np.linalg.norm(rng.normal(0, 1, (n, 2)) * sig, axis=1)
        r = mixture_radius(0.5, p, 40.0)
        assert abs(np.mean(err <= r) - 0.95) < 0.005
    # Tiny outlier probability: the circle stays at the core size.
    assert mixture_radius(0.5, 0.001, 40.0) < 2.0
    # Above 5 % it must reach into the outlier component.
    assert mixture_radius(0.5, 0.2, 40.0) > 20.0


def test_inlier_posterior_uses_agreement_and_prior():
    S = np.eye(2)
    assert inlier_posterior(np.zeros(2), S, 0.0, 40.0) == 1.0
    near = inlier_posterior(np.array([0.5, 0.0]), S, 0.3, 40.0)
    far = inlier_posterior(np.array([8.0, 0.0]), S, 0.3, 40.0)
    assert near > 0.99 and far < 1e-6
    # With a vague prediction the prior probability dominates.
    vague = inlier_posterior(np.array([8.0, 0.0]), 1e4 * S, 0.3, 40.0)
    assert 0.5 < vague < 0.9


def _tap(xy, sigma=1.0, drift=0.3):
    return Measurement(
        np.array(xy, float), sigma**2 * np.eye(2), "tap", drift_sigma=drift
    )


def _sift(xy, sigma=0.3, p=0.001):
    return Measurement(
        np.array(xy, float),
        sigma**2 * np.eye(2),
        "sift",
        p_outlier=p,
        outlier_sigma=40.0,
    )


def test_drift_grows_without_sift_and_is_corrected_by_sift():
    f = KalmanFusion(accel_sigma=0.1)
    f.reset(np.zeros(2), 0.01 * np.eye(2), velocity=np.zeros(2))
    radii = []
    for t in range(1, 41):
        f.predict()
        radii.append(f.update([_tap([0.05 * t, 0.0])]).radius())
    # TAP alone: its drift is unobservable, the radius keeps growing.
    assert radii[-1] > 2 * radii[5]
    # The truth is (0, 0): SIFT says so and reveals the TAP drift.
    for _ in range(5):
        f.predict()
        est = f.update([_tap([2.0, 0.0]), _sift([0.0, 0.0])])
    assert np.linalg.norm(est.position) < 0.5
    np.testing.assert_allclose(est.drift, [2.0, 0.0], atol=0.6)
    assert est.radius() < radii[-1] / 2
    # SIFT gone again: the learned drift is still subtracted from TAP.
    f.predict()
    est = f.update([_tap([2.0, 0.0])])
    assert np.linalg.norm(est.position) < 0.7


def test_unreliable_sift_does_not_pull_the_estimate():
    f = KalmanFusion(accel_sigma=0.5)
    f.reset(np.zeros(2), 0.25 * np.eye(2), velocity=np.zeros(2))
    for _ in range(3):
        f.predict()
        f.update([_tap([0.0, 0.0], drift=0.0)])
    f.predict()
    # Few-inlier SIFT 3 px away: inside the gate, but likely a mismatch.
    est = f.update([_tap([0.0, 0.0], drift=0.0), _sift([3.0, 0.0], p=0.6)])
    assert est.position[0] < 0.5


def test_targets():
    p = parse_target("10,20", radius=12)
    assert not p.is_area and p.radius == 12
    assert len(p.support_offsets()) == 17
    a = parse_target("10,20,40,30")
    assert a.is_area
    np.testing.assert_allclose(a.center, [30, 35])
    assert len(a.support_offsets()) == 17
    np.testing.assert_allclose(
        a.corner_offsets(), [[-20, -15], [20, -15], [20, 15], [-20, 15]]
    )
    mask = Target.point(50.5, 50.5, 10).region_mask((100, 100))
    assert mask[50, 50] and not mask[50, 70]


def test_selection_click_drag_undo_and_scaling():
    import cv2

    from siftap.select import Selection

    sel = Selection(radius=10)
    sel.mouse(cv2.EVENT_LBUTTONDOWN, 50, 40)
    sel.mouse(cv2.EVENT_LBUTTONUP, 51, 40)  # a click: point at pixel centre
    sel.mouse(cv2.EVENT_LBUTTONDOWN, 90, 80)
    sel.mouse(cv2.EVENT_MOUSEMOVE, 70, 70)
    sel.mouse(cv2.EVENT_LBUTTONUP, 60, 60)  # dragged up-left: an area
    point, area = sel.targets
    np.testing.assert_allclose(point.center, [50.5, 40.5])
    assert not point.is_area and area.is_area
    np.testing.assert_allclose(area.box, [60.5, 60.5, 90.5, 80.5])
    assert not sel.key(ord("u")) and len(sel.targets) == 1
    assert sel.key(13)  # enter starts tracking
    assert Selection().key(13) is False  # nothing selected yet
    half = area.scaled(0.5)
    np.testing.assert_allclose(half.box, area.box * 0.5)
    np.testing.assert_allclose(half.center, area.center * 0.5)
