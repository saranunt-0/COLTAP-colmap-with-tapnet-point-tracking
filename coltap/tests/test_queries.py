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


def test_select_queries_fills_free_cells_uniformly():
    image = synthetic.procedural_texture(256, seed=3)
    grid = CoverageGrid.create(256, 256, num_cells_long_side=8)
    occupied = np.zeros(grid.shape[0] * grid.shape[1], bool)
    occupied[:32] = True  # top half already covered
    queries = select_queries(image, grid, occupied, max_queries=1000)
    cells = grid.cell_index(queries)
    assert len(queries) > 0
    assert (cells >= 32).all(), "queries must avoid occupied cells"
    # One query per cell by default -> no duplicate cells.
    assert len(np.unique(cells)) == len(cells)
    # Bottom half should be (nearly) fully covered.
    assert len(np.unique(cells)) >= 28


def test_select_queries_respects_mask_and_budget():
    image = synthetic.procedural_texture(128, seed=4)
    grid = CoverageGrid.create(128, 128, num_cells_long_side=8)
    occupied = np.zeros(64, bool)
    mask = np.zeros((128, 128), bool)
    mask[:, 64:] = True
    queries = select_queries(image, grid, occupied, max_queries=5, mask=mask)
    assert len(queries) <= 5
    assert (queries[:, 0] >= 64).all()
