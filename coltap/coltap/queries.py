# SPDX-License-Identifier: BSD-3-Clause
"""Query point selection: where to start new tracks.

Queries are placed on textured, corner-like pixels (Shi-Tomasi), spread over a
coarse grid so that every image region is covered, and only in grid cells not
already covered by a live, confident track.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass
class CoverageGrid:
    """Coarse grid over the image used for coverage and spatial bucketing."""

    width: int
    height: int
    cell_size: float

    @classmethod
    def create(cls, width: int, height: int, num_cells_long_side: int):
        return cls(width, height, max(width, height) / num_cells_long_side)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols)."""
        return (
            int(np.ceil(self.height / self.cell_size)),
            int(np.ceil(self.width / self.cell_size)),
        )

    def cell_index(self, xy: np.ndarray) -> np.ndarray:
        """Flat cell index for each (x, y); -1 for points outside the image."""
        rows, cols = self.shape
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        inside = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < self.width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < self.height)
        )
        col = np.clip((np.nan_to_num(xy[:, 0]) // self.cell_size), 0, cols - 1)
        row = np.clip((np.nan_to_num(xy[:, 1]) // self.cell_size), 0, rows - 1)
        index = (row * cols + col).astype(np.int64)
        index[~inside] = -1
        return index

    def occupancy(self, xy: np.ndarray) -> np.ndarray:
        """[rows * cols] bool, True for cells containing at least one point."""
        rows, cols = self.shape
        occupied = np.zeros(rows * cols, dtype=bool)
        index = self.cell_index(xy)
        occupied[index[index >= 0]] = True
        return occupied

    def valid_cells(self, mask: np.ndarray | None) -> np.ndarray:
        """Cells with at least some unmasked pixels."""
        rows, cols = self.shape
        if mask is None:
            return np.ones(rows * cols, dtype=bool)
        small = cv2.resize(
            mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA
        )
        return (small > 0.25).reshape(-1)


def detect_corners(
    gray: np.ndarray,
    max_corners: int,
    min_distance: float,
    mask: np.ndarray | None = None,
    quality_level: float = 0.001,
    block_size: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Shi-Tomasi corners in COLMAP coordinates (+0.5) with their scores."""
    cv_mask = None if mask is None else mask.astype(np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=cv_mask,
        blockSize=block_size,
    )
    if corners is None:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    corners = corners.reshape(-1, 2)
    eig = cv2.cornerMinEigenVal(gray, block_size)
    xi = np.clip(np.round(corners[:, 0]).astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(np.round(corners[:, 1]).astype(int), 0, gray.shape[0] - 1)
    scores = eig[yi, xi]
    # OpenCV puts pixel centers at integers, COLMAP at +0.5.
    return (corners + 0.5).astype(np.float32), scores.astype(np.float32)


def select_queries(
    image_rgb: np.ndarray,
    grid: CoverageGrid,
    occupied: np.ndarray,
    max_queries: int,
    per_cell: int = 1,
    mask: np.ndarray | None = None,
    border: int = 4,
) -> np.ndarray:
    """Pick up to ``max_queries`` corners in unoccupied grid cells.

    Corners are taken round-robin over cells (best corner of every free cell
    first) so that the result is spatially uniform rather than clustered on
    the most textured region.
    """
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    valid = np.zeros((height, width), dtype=bool)
    valid[border : height - border, border : width - border] = True
    if mask is not None:
        valid &= mask
    min_distance = max(2.0, 0.25 * grid.cell_size)
    corners, scores = detect_corners(
        gray,
        max_corners=8 * occupied.size * per_cell,
        min_distance=min_distance,
        mask=valid,
    )
    if len(corners) == 0:
        return corners
    cells = grid.cell_index(corners)
    free = (cells >= 0) & ~occupied[np.maximum(cells, 0)]
    corners, scores, cells = corners[free], scores[free], cells[free]
    # Rank of each corner within its cell (0 = strongest).
    order = np.lexsort((-scores, cells))
    corners, scores, cells = corners[order], scores[order], cells[order]
    first = np.r_[True, cells[1:] != cells[:-1]]
    group_start = np.maximum.accumulate(
        np.where(first, np.arange(len(cells)), 0)
    )
    rank = np.arange(len(cells)) - group_start
    keep = rank < per_cell
    corners, scores, rank = corners[keep], scores[keep], rank[keep]
    # Round-robin over cells: all rank-0 corners (by score), then rank-1, ...
    order = np.lexsort((-scores, rank))
    return corners[order[:max_queries]]
