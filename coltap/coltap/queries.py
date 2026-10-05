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


def detect_sift(
    gray: np.ndarray, max_keypoints: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """SIFT keypoint locations (COLMAP convention) with their responses.

    Duplicates (SIFT returns one keypoint per dominant orientation) are
    removed; the strongest response per location is kept.
    """
    cv_mask = None if mask is None else mask.astype(np.uint8) * 255
    keypoints = cv2.SIFT_create(nfeatures=max_keypoints).detect(gray, cv_mask)
    if not keypoints:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    xy = np.array([kp.pt for kp in keypoints], np.float32)
    scores = np.array([kp.response for kp in keypoints], np.float32)
    order = np.argsort(-scores)
    xy, scores = xy[order], scores[order]
    _, first = np.unique(
        np.round(xy).astype(np.int64), axis=0, return_index=True
    )
    first = np.sort(first)
    return (xy[first] + 0.5).astype(np.float32), scores[first]


def _suppress(xy, scores, blocked, radius):
    """Greedy non-maximum suppression on a raster, strongest first.

    ``blocked`` (uint8 image, modified in place) already contains discs
    around points that must not be duplicated (live tracks).
    """
    height, width = blocked.shape
    keep = np.zeros(len(xy), bool)
    # Slightly smaller than ``radius``: rasterized discs would otherwise
    # reject candidates that the detector already spaced by ``radius``.
    r = max(int(0.9 * radius), 1)
    for k in np.argsort(-scores):
        x, y = int(xy[k, 0]), int(xy[k, 1])
        if 0 <= x < width and 0 <= y < height and not blocked[y, x]:
            keep[k] = True
            cv2.circle(blocked, (x, y), r, 1, -1)
    return keep


def select_queries(
    image_rgb: np.ndarray,
    grid: CoverageGrid,
    live_xy: np.ndarray,
    max_queries: int,
    valid_cells: np.ndarray | None = None,
    per_cell: int = 1,
    max_per_cell: int = 1,
    texture_threshold: float = 0.05,
    detector: str = "shi_tomasi",
    min_distance: float = 0.0,
    min_corner_quality: float = 1e-4,
    mask: np.ndarray | None = None,
    border: int = 4,
) -> np.ndarray:
    """Pick up to ``max_queries`` new query points.

    Each grid cell has a quota. Live tracks already in a cell count towards
    it, and new queries keep ``min_distance`` from live tracks and from each
    other. Queries are chosen in two passes:

    1. coverage: every cell is filled up to ``per_cell`` with its strongest
       corners, wherever there is any texture at all (flat walls, ground,
       distant terrain) -- this gives uniform coverage;
    2. texture: cells are filled further, up to ``max_per_cell``, with
       corners whose score is at least ``texture_threshold`` times the
       image's 95th-percentile score (buildings, windows, ...).

    ``detector`` is ``"shi_tomasi"`` (corners, what KLT-style trackers use)
    or ``"sift"`` (SIFT keypoint locations, i.e. where COLMAP's SIFT would
    put features). ``min_corner_quality`` is the Shi-Tomasi floor relative
    to the strongest corner in the image; it is kept low so that weakly
    textured cells next to high-contrast structures still get candidates.
    """
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    rows, cols = grid.shape
    if valid_cells is None:
        valid_cells = np.ones(rows * cols, bool)
    allowed = np.zeros((height, width), dtype=bool)
    allowed[border : height - border, border : width - border] = True
    if mask is not None:
        allowed &= mask
    if min_distance <= 0:
        min_distance = max(2.0, 0.25 * grid.cell_size)
    max_per_cell = max(max_per_cell, per_cell)
    # No global candidate cap (0 = all): a cap filled by the strongest
    # corners image-wide would starve weakly textured cells of candidates.
    if detector == "sift":
        corners, scores = detect_sift(gray, 0, allowed)
    elif detector == "shi_tomasi":
        corners, scores = detect_corners(
            gray,
            max_corners=0,
            min_distance=min_distance,
            mask=allowed,
            quality_level=min_corner_quality,
        )
    else:
        raise ValueError(f"Unknown query detector {detector!r}")
    if len(corners) == 0:
        return corners

    live_xy = np.asarray(live_xy, np.float32).reshape(-1, 2)
    blocked = np.zeros((height, width), np.uint8)
    r = max(int(round(min_distance)), 1)
    for x, y in live_xy[np.isfinite(live_xy).all(axis=1)]:
        cv2.circle(blocked, (int(x), int(y)), r, 1, -1)
    keep = _suppress(corners, scores, blocked, min_distance)
    cells = grid.cell_index(corners)
    keep &= (cells >= 0) & valid_cells[np.maximum(cells, 0)]
    corners, scores, cells = corners[keep], scores[keep], cells[keep]
    if len(corners) == 0:
        return corners
    reference = np.percentile(scores, 95)

    # Slot each corner would take in its cell, after the live tracks there.
    live_cells = grid.cell_index(live_xy)
    live_count = np.bincount(live_cells[live_cells >= 0], minlength=rows * cols)
    order = np.lexsort((-scores, cells))
    corners, scores, cells = corners[order], scores[order], cells[order]
    first = np.r_[True, cells[1:] != cells[:-1]]
    group_start = np.maximum.accumulate(
        np.where(first, np.arange(len(cells)), 0)
    )
    slot = np.arange(len(cells)) - group_start + live_count[cells]

    coverage = slot < per_cell
    textured = (
        ~coverage
        & (slot < max_per_cell)
        & (scores >= texture_threshold * reference)
    )
    # Coverage first (round-robin over cells by slot, then score), then the
    # textured extras by score.
    cov = np.nonzero(coverage)[0]
    cov = cov[np.lexsort((-scores[cov], slot[cov]))]
    tex = np.nonzero(textured)[0]
    tex = tex[np.argsort(-scores[tex])]
    chosen = np.concatenate([cov, tex])[:max_queries]
    return corners[chosen]
