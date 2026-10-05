# SPDX-License-Identifier: BSD-3-Clause
import cv2
import numpy as np

from coltap import synthetic
from coltap.bridge import (
    bridge_pairs,
    extract_sift,
    match_descriptors,
    weak_links,
)


def test_weak_links_and_bridge_window():
    observed = np.zeros((10, 8), bool)
    observed[:5, :4] = True  # tracks on frames 0-3
    observed[5:, 4:] = True  # new tracks on frames 4-7 -> break at 3|4
    assert weak_links(observed, min_shared_tracks=3) == [3]
    pairs = bridge_pairs([3], num_frames=8, window=2)
    assert pairs == [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    # Long-range schedule pairs that cross the break are added as well.
    schedule = [(0, 4), (0, 6), (1, 5), (2, 3), (4, 7)]
    pairs = bridge_pairs([3], num_frames=8, window=2, schedule=schedule)
    assert (0, 4) in pairs and (1, 5) in pairs
    # Bridge frames on the same side are matched among themselves.
    assert (0, 1) in pairs and (0, 2) in pairs
    assert (0, 6) not in pairs  # ends outside the window after the break
    assert (4, 7) not in pairs  # does not cross the break


def test_sift_bridge_matches_shifted_image():
    texture = synthetic.procedural_texture(320, seed=7)
    img1 = texture[:240, :280]
    img2 = texture[10:250, 25:305]  # content shifted by (-25, -10)
    xy1, d1 = extract_sift(img1, 2000)
    xy2, d2 = extract_sift(img2, 2000)
    matches = match_descriptors(d1, d2, max_ratio=0.8)
    assert len(matches) > 50
    flow = xy2[matches[:, 1]] - xy1[matches[:, 0]]
    inliers = np.linalg.norm(flow - [-25, -10], axis=1) < 1.0
    assert inliers.mean() > 0.9
    # Keypoints use COLMAP's +0.5 pixel convention.
    kp = cv2.SIFT_create(nfeatures=2000).detect(
        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None
    )
    np.testing.assert_allclose(xy1[0], np.array(kp[0].pt) + 0.5, atol=1e-4)
