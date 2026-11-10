# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from siftap.focus import jitter_series, stabilized_crop, to_panel
from siftap.fusion import FusedEstimate
from siftap.targets import Target
from siftap.tracker import FrameRecord


def _scene(center, size=(160, 200)):
    """A textured image whose content is shifted so that the pattern's
    reference point lies at ``center`` (sub-pixel, COLMAP convention)."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64)).astype(np.float32)
    base = np.kron(base, np.ones((4, 4), np.float32))  # 256x256, smooth-ish
    import cv2

    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    # Pattern point (128, 128) (COLMAP) is moved to ``center``.
    M = np.float32([[1, 0, center[0] - 128], [0, 1, center[1] - 128]])
    img = cv2.warpAffine(base, M, size[::-1], flags=cv2.INTER_CUBIC)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, 2)


def _rec(t, pos):
    fused = FusedEstimate(np.asarray(pos, float), 0.1 * np.eye(2))
    return FrameRecord(t, np.asarray(pos, float), None, None, fused, None)


def test_stabilized_crop_centres_the_tracked_point_subpixel():
    target = Target.point(100.0, 80.0)
    centers = [(100.0, 80.0), (103.3, 81.7), (98.6, 84.2)]
    crops = [
        stabilized_crop(_scene(c)[..., 0], np.eye(2), np.array(c), 20.0, 80)
        for c in centers
    ]
    for crop in crops[1:]:
        diff = np.abs(
            crop[10:-10, 10:-10].astype(float) - crops[0][10:-10, 10:-10]
        )
        assert diff.mean() < 1.5  # same content, only interpolation noise
    # The panel centre maps back to the tracked point.
    np.testing.assert_allclose(
        to_panel(
            np.array(centers[1]), np.eye(2), np.array(centers[1]), 20.0, 80
        )[0],
        [39.5, 39.5],
    )
    assert target.radius > 0


def test_jitter_is_zero_when_tracking_is_exact_and_sees_a_pop():
    target = Target.point(100.0, 80.0)
    truth = [(100.0 + 0.7 * t, 80.0 + 0.3 * t) for t in range(6)]
    images = [_scene(c) for c in truth]
    exact = [_rec(t, c) for t, c in enumerate(truth)]
    j = jitter_series(images, exact, target, half=20.0, size=80)
    assert np.nanmax(j[1:]) < 0.15
    popped = list(exact)
    popped[3] = _rec(3, (truth[3][0] + 1.5, truth[3][1]))  # 1.5 px pop
    j = jitter_series(images, popped, target, half=20.0, size=80)
    assert 1.2 < j[3] < 1.8 and 1.2 < j[4] < 1.8  # in and out of the pop
    assert j[2] < 0.15 and j[5] < 0.15
