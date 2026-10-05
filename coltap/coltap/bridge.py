# SPDX-License-Identifier: BSD-3-Clause
"""Bridge discontinuities in the track graph with wide-baseline matches.

A temporal point tracker cannot follow a cut, a dropped stretch of frames or
a very fast motion: on such a frame all tracks end and new ones start, so no
track connects the frames before and after the break and the mapper would
split the sequence into separate models. SIFT matching has no such
limitation because it matches every image pair independently.

Wherever two consecutive frames share fewer than ``min_shared_tracks``
tracks, we extract SIFT features on the ``window`` frames before and after
the break and match all pairs inside that window, plus the long-range pairs
of the sequential-matcher schedule that cross the break (ratio test + mutual
nearest neighbour). These features are appended to the frames' keypoints
and their matches go through the same COLMAP geometric verification. Away
from breaks the database contains TAPNext++ tracks only.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class BridgeOptions:
    enabled: bool = True
    min_shared_tracks: int = 30
    window: int = 3
    max_num_features: int = 4000
    max_ratio: float = 0.8


@dataclasses.dataclass
class BridgeFeatures:
    """SIFT keypoints per frame and raw matches per frame pair."""

    keypoints: dict[int, np.ndarray]  # frame -> [K, 2] (COLMAP convention)
    matches: dict[tuple[int, int], np.ndarray]  # (i, j) -> [M, 2] uint32


def weak_links(observed: np.ndarray, min_shared_tracks: int) -> list[int]:
    """Frames i such that frames i and i+1 share too few tracks."""
    shared = (observed[:, :-1] & observed[:, 1:]).sum(axis=0)
    return np.nonzero(shared < min_shared_tracks)[0].tolist()


def bridge_pairs(
    links: list[int],
    num_frames: int,
    window: int,
    schedule: list[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Pairs to match with SIFT around each weak link ``i`` (i | i+1).

    All pairs inside the ``window`` frames on either side of the break, plus
    every pair of the regular pairing ``schedule`` (e.g. the sequential
    matcher's quadratic long-range pairs) that crosses the break and ends
    within the window after it. The latter reconnects sequences whose camera
    jumps but later revisits earlier viewpoints.
    """
    pairs = set()
    for i in links:
        last = min(num_frames, i + window + 1)
        frames = range(max(0, i - window + 1), last)
        crossing = [(a, b) for a, b in schedule if a <= i < b < last]
        # Frames on each side of the break that take part in the bridge are
        # also matched among themselves, so that bridge features are seen by
        # several images per side and can be triangulated.
        before = sorted(
            {f for f in frames if f <= i} | {a for a, _ in crossing}
        )
        after = sorted({f for f in frames if f > i} | {b for _, b in crossing})
        pairs.update(crossing)
        pairs.update(itertools.combinations(before, 2))
        pairs.update(itertools.combinations(after, 2))
        pairs.update((a, b) for a in before[-window:] for b in after)
    return sorted(pairs)


def extract_sift(
    image_rgb: np.ndarray, max_num_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """SIFT keypoints (COLMAP convention) and RootSIFT descriptors."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_num_features)
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is None or not keypoints:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    xy = np.array([kp.pt for kp in keypoints], np.float32) + 0.5
    desc = desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-8)
    return xy, np.sqrt(desc).astype(np.float32)


def match_descriptors(d1, d2, max_ratio: float) -> np.ndarray:
    """Mutual nearest neighbours that pass Lowe's ratio test."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.uint32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    fwd = matcher.knnMatch(d1, d2, k=2)
    bwd = matcher.match(d2, d1)
    back = np.full(len(d2), -1, np.int64)
    for m in bwd:
        back[m.queryIdx] = m.trainIdx
    out = [
        (m.queryIdx, m.trainIdx)
        for m, n in (p for p in fwd if len(p) == 2)
        if m.distance < max_ratio * n.distance
        and back[m.trainIdx] == m.queryIdx
    ]
    return np.array(out, np.uint32).reshape(-1, 2)


def compute_bridges(
    observed: np.ndarray,
    sequence,
    options: BridgeOptions,
    schedule: list[tuple[int, int]] = (),
) -> BridgeFeatures | None:
    """SIFT features/matches around weak links of the track graph."""
    if not options.enabled or observed.shape[1] < 2:
        return None
    links = weak_links(observed, options.min_shared_tracks)
    if not links:
        return None
    pairs = bridge_pairs(links, observed.shape[1], options.window, schedule)
    frames = sorted({f for pair in pairs for f in pair})
    LOGGER.info(
        "Bridging %d weak link(s) at frames %s with SIFT on %d frames",
        len(links),
        links,
        len(frames),
    )
    keypoints, descriptors = {}, {}
    for f in frames:
        keypoints[f], descriptors[f] = extract_sift(
            sequence.image(f), options.max_num_features
        )
    matches = {
        (i, j): match_descriptors(
            descriptors[i], descriptors[j], options.max_ratio
        )
        for i, j in pairs
    }
    return BridgeFeatures(keypoints=keypoints, matches=matches)
