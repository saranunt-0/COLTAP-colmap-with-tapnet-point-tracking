# SPDX-License-Identifier: BSD-3-Clause
import cv2
import numpy as np

from coltap import synthetic
from siftap.geometry import Measurement
from siftap.sift import SiftFrontEnd, SiftParams
from siftap.tap import TapCalibration, TapRaw, tap_measurement
from siftap.targets import Target


def _shifted_pair(dx=7, dy=-4):
    tex = synthetic.procedural_texture(400, seed=3)
    gray = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY)
    a = gray[50:290, 50:370]
    b = gray[
        50 - dy : 290 - dy, 50 - dx : 370 - dx
    ]  # content moves by (dx, dy)
    return a, b


def test_sift_measures_translation_with_small_uncertainty():
    a, b = _shifted_pair()
    target = Target.point(160.5, 120.5, radius=40)
    front = SiftFrontEnd(target, SiftParams())
    assert front.add_reference(a, target.center, 0) >= 6
    m = front.measure(b, target.center, 2.0)
    assert m is not None
    np.testing.assert_allclose(m.position, target.center + [7, -4], atol=0.6)
    assert m.radius() < 3.0
    assert m.info["inliers"] >= 6


def test_sift_returns_none_on_textureless_region():
    flat = np.full((240, 320), 128, np.uint8)
    target = Target.point(160.5, 120.5, radius=30)
    front = SiftFrontEnd(target)
    assert front.add_reference(flat, target.center, 0) == 0
    assert front.measure(flat, target.center, 2.0) is None


def _raw(xy, vis, sigma):
    q = len(xy)
    sig = np.full((q, 2), sigma, np.float32)
    return TapRaw(
        np.asarray(xy, np.float32), np.asarray(vis, np.float32), sig, sig
    )


def test_tap_direct_and_fit_modes():
    target = Target.point(100.5, 100.5, radius=24)
    offsets = target.support_offsets()
    xy = target.center + offsets + [3.0, 1.0]
    raw = _raw(xy, np.ones(len(xy)), 0.8)
    sel = slice(0, len(xy))
    unit = TapCalibration(sigma_scale=1.0, sigma_floor=0.0)
    direct = tap_measurement(raw, sel, offsets, target, unit)
    np.testing.assert_allclose(
        direct.position, target.center + [3, 1], atol=1e-4
    )
    np.testing.assert_allclose(direct.cov, np.eye(2) * 0.64, atol=1e-4)
    calibrated = tap_measurement(raw, sel, offsets, target, TapCalibration())
    assert calibrated.sigma < direct.sigma  # the raw heatmap is too cautious
    fit = tap_measurement(
        raw,
        sel,
        offsets,
        target,
        TapCalibration(sigma_scale=1.0, sigma_floor=0.0, point_mode="fit"),
    )
    np.testing.assert_allclose(fit.position, target.center + [3, 1], atol=1e-3)


def test_tap_occluded_target_uses_support_or_reports_none():
    target = Target.point(100.5, 100.5, radius=24)
    offsets = target.support_offsets()
    xy = target.center + offsets
    vis = np.ones(len(xy))
    vis[0] = 0.1  # the target point itself is hidden
    sel = slice(0, len(xy))
    m = tap_measurement(
        _raw(xy, vis, 1.0), sel, offsets, target, TapCalibration()
    )
    assert m is not None  # recovered from the visible support points
    np.testing.assert_allclose(m.position, target.center, atol=1e-3)
    vis[:] = 0.1
    assert (
        tap_measurement(
            _raw(xy, vis, 1.0), sel, offsets, target, TapCalibration()
        )
        is None
    )


def test_texture_sigma_ranks_texture_and_sets_drift():
    from siftap.tap import add_texture_drift, texture_sigma

    rng = np.random.default_rng(0)
    gray = np.full((120, 240), 128, np.uint8)
    gray[:, 120:] = rng.integers(0, 255, (120, 120), dtype=np.uint8)
    flat, busy = texture_sigma(gray, np.array([[60.5, 60.5], [180.5, 60.5]]))
    assert flat > 100 * busy
    meas = [
        Measurement(np.array([60.5, 60.5]), np.eye(2), "tap"),
        None,
        Measurement(np.array([180.5, 60.5]), np.eye(2), "tap"),
    ]
    add_texture_drift(meas, gray, TapCalibration(drift_scale=2.0))
    assert meas[0].drift_sigma > 100 * meas[2].drift_sigma
