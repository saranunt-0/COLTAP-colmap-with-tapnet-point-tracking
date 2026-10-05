# SPDX-License-Identifier: BSD-3-Clause
import cv2
import numpy as np

from coltap import synthetic
from coltap.queries import CoverageGrid, detect_corners, select_queries


def test_grid_cell_index_and_occupancy():
    grid = CoverageGrid.create(width=100, height=50, num_cells_long_side=10)
    assert grid.shape == (5, 10)
    xy = np.array([[0.5, 0.5], [99.9, 49.9], [-1, 3], [np.nan, 2], [55, 25]])
    index = grid.cell_index(xy)
    assert index.tolist() == [0, 49, -1, -1, 25]
    occupied = grid.occupancy(xy)
    assert occupied.sum() == 3


def test_valid_cells_respects_mask():
    grid = CoverageGrid.create(width=100, height=100, num_cells_long_side=4)
    mask = np.ones((100, 100), bool)
    mask[:, :50] = False
    valid = grid.valid_cells(mask).reshape(grid.shape)
    assert not valid[:, :2].any()
    assert valid[:, 2:].all()


def test_detect_corners_uses_colmap_pixel_convention():
    gray = synthetic.procedural_texture(64, seed=1)[..., 0]
    corners, scores = detect_corners(gray, max_corners=20, min_distance=3)
    reference = cv2.goodFeaturesToTrack(
        gray, maxCorners=20, qualityLevel=0.001, minDistance=3, blockSize=7
    ).reshape(-1, 2)
    assert len(corners) == len(reference) and (scores > 0).all()
    # OpenCV puts pixel centers at integers, COLMAP at +0.5.
    np.testing.assert_allclose(corners, reference + 0.5)


def _cell_centers(grid, cells):
    rows, cols = grid.shape
    return np.stack(
        [
            (np.asarray(cells) % cols + 0.5) * grid.cell_size,
            (np.asarray(cells) // cols + 0.5) * grid.cell_size,
        ],
        axis=1,
    ).astype(np.float32)


def test_select_queries_fills_free_cells_uniformly():
    image = synthetic.procedural_texture(256, seed=3)
    grid = CoverageGrid.create(256, 256, num_cells_long_side=8)
    live = _cell_centers(grid, range(32))  # top half already tracked
    queries = select_queries(image, grid, live, max_queries=1000)
    cells = grid.cell_index(queries)
    assert len(queries) > 0
    assert (cells >= 32).all(), "queries must avoid cells at their quota"
    # One query per cell by default -> no duplicate cells.
    assert len(np.unique(cells)) == len(cells)
    # Bottom half should be (nearly) fully covered.
    assert len(np.unique(cells)) >= 28


def test_select_queries_respects_mask_and_budget():
    image = synthetic.procedural_texture(128, seed=4)
    grid = CoverageGrid.create(128, 128, num_cells_long_side=8)
    mask = np.zeros((128, 128), bool)
    mask[:, 64:] = True
    queries = select_queries(
        image, grid, np.zeros((0, 2)), max_queries=5, mask=mask
    )
    assert len(queries) <= 5
    assert (queries[:, 0] >= 64).all()


def _half_textured_image(size=256):
    """Left half: strong texture. Right half: the same pattern at 10 %
    contrast (weak texture, e.g. hazy terrain)."""
    image = synthetic.procedural_texture(size, seed=5).astype(np.float32)
    left = image[:, : size // 2]
    image[:, size // 2 :] = left.mean() + 0.1 * (left - left.mean())
    return image.astype(np.uint8)


def test_adaptive_sampling_densifies_textured_cells_only():
    image = _half_textured_image()
    grid = CoverageGrid.create(256, 256, num_cells_long_side=8)
    uniform = select_queries(image, grid, np.zeros((0, 2)), 10000)
    adaptive = select_queries(
        image,
        grid,
        np.zeros((0, 2)),
        10000,
        max_per_cell=4,
        texture_threshold=0.05,
    )
    left = adaptive[:, 0] < 128
    # Textured cells get several queries, faint cells keep coverage only.
    assert left.sum() > 2.5 * (uniform[:, 0] < 128).sum()
    right_cells = grid.cell_index(adaptive[~left])
    assert np.bincount(right_cells).max() <= 1
    assert len(np.unique(right_cells)) >= 0.75 * 32  # still covered


def test_new_queries_keep_distance_from_live_tracks():
    image = synthetic.procedural_texture(256, seed=6)
    grid = CoverageGrid.create(256, 256, num_cells_long_side=8)
    live = np.array([[50.0, 50.0], [150.0, 120.0]], np.float32)
    queries = select_queries(
        image, grid, live, 10000, max_per_cell=4, min_distance=10.0
    )
    dist = np.linalg.norm(queries[:, None] - live[None], axis=-1)
    assert dist.min() >= 9.0
    pairwise = np.linalg.norm(queries[:, None] - queries[None], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() >= 8.0


def test_sift_detector_queries():
    image = synthetic.procedural_texture(256, seed=7)
    grid = CoverageGrid.create(256, 256, num_cells_long_side=8)
    queries = select_queries(
        image, grid, np.zeros((0, 2)), 10000, max_per_cell=3, detector="sift"
    )
    assert len(queries) > 64
    assert np.bincount(grid.cell_index(queries)).max() <= 3
