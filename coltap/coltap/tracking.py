# SPDX-License-Identifier: BSD-3-Clause
"""Run TAPNext++ over an ordered image sequence and produce point tracks.

TAPNext++ is an online tracker whose query points are defined on the first
frame of a tracking run. We therefore follow the same scheme as DeepMind's own
TAPNext++ VOTS tracker and start a fresh tracker *instance* for every batch of
new query points:

* frame 0 seeds queries uniformly over the image (one corner per grid cell);
* on later frames, when the share of grid cells covered by live, confident
  tracks drops below ``seed_coverage`` (new content entering the view, points
  lost), a new instance is started with queries in the uncovered cells;
* at most ``max_active_instances`` run concurrently, which bounds the cost.
  When the cap is reached the oldest instance is retired and its confidently
  tracked points are *handed off* to the new instance (re-queried at their
  current position under the same track id), so long tracks survive;
* an instance is also retired once too few of its points remain confidently
  visible.

Optionally (``bidirectional``) each instance is also run backwards in time so
that points seeded on frame k are recovered on the frames before k as well.
When several instances observe the same track on the same frame, the most
confident observation wins.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from .frames import ImageSequence
from .model import TapNextPP, image_to_model, model_to_image
from .queries import CoverageGrid, select_queries
from .tracks import Tracks

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class TrackingOptions:
    # Grid used for query placement and coverage: number of cells along the
    # long image side. Every cell gets ``queries_per_cell`` queries wherever
    # there is any texture (coverage); textured cells are filled further, up
    # to ``max_queries_per_cell``, with corners scoring at least
    # ``texture_threshold`` x the image's 95th-percentile corner score.
    grid_cells: int = 24
    queries_per_cell: int = 1
    max_queries_per_cell: int = 1
    texture_threshold: float = 0.05
    # Where queries go: "shi_tomasi" corners or "sift" keypoint locations.
    query_detector: str = "shi_tomasi"
    # Minimum distance (px) between queries and live tracks; 0 = cell / 4.
    query_min_distance: float = 0.0
    # Shi-Tomasi floor relative to the image's strongest corner.
    min_corner_quality: float = 1e-4
    max_queries_per_instance: int = 2000
    min_queries_per_instance: int = 16
    # Start a new tracker instance when the fraction of grid cells covered by
    # live, confident tracks drops below this value...
    seed_coverage: float = 0.75
    # ...but not more often than every ``min_keyframe_interval`` frames, and
    # at least every ``max_keyframe_interval`` frames (0 disables).
    min_keyframe_interval: int = 2
    max_keyframe_interval: int = 0
    # Below this coverage a new instance is started regardless of the
    # keyframe interval (e.g. after a cut or a fast motion lost all points).
    force_seed_coverage: float = 0.4
    # Observation confidence (visibility * certainty) to count as tracked.
    min_confidence: float = 0.5
    # Concurrency cap and handoff of points from the retired instance.
    max_active_instances: int = 3
    handoff_min_confidence: float = 0.8
    # Retire an instance when fewer than this fraction of its points are
    # confidently tracked for ``patience`` consecutive frames.
    min_alive_fraction: float = 0.15
    patience: int = 3
    max_instance_length: int = 0
    # Also track every instance backwards in time from its query frame.
    bidirectional: bool = False
    max_backward_length: int = 0


class _Instance:
    """One TAPNext++ run with a fixed query set."""

    def __init__(
        self,
        start: int,
        num_frames: int,
        query_xy: np.ndarray,
        track_ids: np.ndarray,
    ):
        q = len(query_xy)
        self.start = start
        self.query_xy = query_xy
        self.track_ids = track_ids
        self.xy = np.full((num_frames, q, 2), np.nan, np.float32)
        self.visibility = np.zeros((num_frames, q), np.float32)
        self.certainty = np.zeros((num_frames, q), np.float32)
        self.state = None
        self.frames_below = 0
        self.length = 0

    def record(self, t, xy, visibility, certainty, width, height, mask):
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        if mask is not None:
            xi = np.clip(xy[:, 0].astype(int), 0, width - 1)
            yi = np.clip(xy[:, 1].astype(int), 0, height - 1)
            inside &= mask[yi, xi]
        self.xy[t] = xy
        self.visibility[t] = np.where(inside, visibility, 0.0)
        self.certainty[t] = np.where(inside, certainty, 0.0)
        self.length += 1

    def confidence(self, t) -> np.ndarray:
        return self.visibility[t] * self.certainty[t]

    def update_liveness(self, t, options: TrackingOptions) -> bool:
        """Returns False once the instance should be retired."""
        alive = (self.confidence(t) >= options.min_confidence).mean()
        self.frames_below = (
            self.frames_below + 1 if alive < options.min_alive_fraction else 0
        )
        if self.frames_below >= options.patience:
            return False
        return not (
            options.max_instance_length > 0
            and self.length >= options.max_instance_length
        )


def track_sequence(
    sequence: ImageSequence,
    model: TapNextPP,
    options: TrackingOptions | None = None,
) -> Tracks:
    options = options or TrackingOptions()
    num_frames = len(sequence)
    sizes = np.zeros((num_frames, 2), np.int32)
    model_frames: list = []  # preprocessed frames, for backward runs
    masks: list[np.ndarray | None] = []
    instances: list[_Instance] = []
    active: list[_Instance] = []
    track_query_frame: list[int] = []
    last_seed = -(10**9)
    grid = None
    tic = time.time()

    for t in range(num_frames):
        image = sequence.image(t)
        height, width = image.shape[:2]
        sizes[t] = (width, height)
        mask = sequence.mask(t, (height, width))
        frame = model.preprocess(image)
        if options.bidirectional:
            model_frames.append(frame)
            masks.append(mask)
        if grid is None or (grid.width, grid.height) != (width, height):
            grid = CoverageGrid.create(width, height, options.grid_cells)

        # 1. Advance all live instances by one frame.
        still_active = []
        for inst in active:
            out, inst.state = model.step(frame, state=inst.state)
            inst.record(
                t,
                model_to_image(out.xy, width, height),
                out.visibility,
                out.certainty,
                width,
                height,
                mask,
            )
            if inst.update_liveness(t, options):
                still_active.append(inst)
            else:
                inst.state = None  # free memory
        active = still_active

        # 2. Decide whether to start a new instance.
        valid = grid.valid_cells(mask)
        live_xy = _live_points(active, t, options.min_confidence)
        occupied = grid.occupancy(live_xy)
        coverage = (occupied & valid).sum() / max(valid.sum(), 1)
        since = t - last_seed
        if not (
            t == 0
            or coverage < options.force_seed_coverage
            or (
                coverage < options.seed_coverage
                and since >= options.min_keyframe_interval
            )
            or (
                options.max_keyframe_interval > 0
                and since >= options.max_keyframe_interval
            )
        ):
            continue

        # 3. Retire the oldest instance if at the cap; hand off its points.
        handoff_xy = np.zeros((0, 2), np.float32)
        handoff_ids = np.zeros(0, np.int64)
        if len(active) >= options.max_active_instances:
            oldest = min(active, key=lambda inst: inst.start)
            active.remove(oldest)
            oldest.state = None
            keep = oldest.confidence(t) >= options.handoff_min_confidence
            handoff_xy = oldest.xy[t][keep]
            handoff_ids = oldest.track_ids[keep]
            live_xy = np.concatenate(
                [_live_points(active, t, options.min_confidence), handoff_xy]
            )

        # 4. New queries where cells are below their quota.
        new_xy = select_queries(
            image,
            grid,
            live_xy,
            max(options.max_queries_per_instance - len(handoff_xy), 0),
            valid_cells=valid,
            per_cell=options.queries_per_cell,
            max_per_cell=options.max_queries_per_cell,
            texture_threshold=options.texture_threshold,
            detector=options.query_detector,
            min_distance=options.query_min_distance,
            min_corner_quality=options.min_corner_quality,
            mask=mask,
        )
        if len(new_xy) + len(handoff_xy) < options.min_queries_per_instance:
            continue
        new_ids = np.arange(
            len(track_query_frame),
            len(track_query_frame) + len(new_xy),
            dtype=np.int64,
        )
        track_query_frame.extend([t] * len(new_xy))
        query_xy = np.concatenate([handoff_xy, new_xy]).astype(np.float32)
        inst = _Instance(
            t, num_frames, query_xy, np.concatenate([handoff_ids, new_ids])
        )
        _, inst.state = model.step(
            frame, query_xy=image_to_model(query_xy, width, height)
        )
        # On its query frame a track is, by definition, at the query.
        ones = np.ones(len(query_xy), np.float32)
        inst.record(t, query_xy, ones, ones, width, height, mask)
        instances.append(inst)
        active.append(inst)
        last_seed = t
        LOGGER.info(
            "frame %d/%d: coverage %.2f -> new instance: %d new + %d handed "
            "off queries (%d active, %.1fs)",
            t + 1,
            num_frames,
            coverage,
            len(new_xy),
            len(handoff_xy),
            len(active),
            time.time() - tic,
        )

    if options.bidirectional:
        for inst in instances:
            if inst.start > 0:
                _track_backward(
                    inst, model, model_frames, masks, sizes, options
                )

    if not instances:
        raise RuntimeError("No trackable points found in the sequence.")
    tracks = _assemble(
        instances,
        np.asarray(track_query_frame, np.int32),
        list(sequence.names),
        sizes,
    )
    LOGGER.info(
        "Tracking done: %d tracks from %d instances in %.1fs",
        tracks.num_tracks,
        len(instances),
        time.time() - tic,
    )
    return tracks


def _live_points(active, t, min_confidence) -> np.ndarray:
    """Positions of confidently tracked points of all live instances."""
    points = [
        inst.xy[t][inst.confidence(t) >= min_confidence] for inst in active
    ]
    return (
        np.concatenate(points).astype(np.float32)
        if points
        else np.zeros((0, 2), np.float32)
    )


def _assemble(instances, query_frame, names, sizes) -> Tracks:
    """Merge per-instance results into global tracks (best confidence wins)."""
    n, t = len(query_frame), len(names)
    xy = np.full((n, t, 2), np.nan, np.float32)
    visibility = np.zeros((n, t), np.float32)
    certainty = np.zeros((n, t), np.float32)
    best = np.full((n, t), -1.0, np.float32)
    for inst in instances:
        conf = (inst.visibility * inst.certainty).T  # [Q, T]
        evaluated = np.isfinite(inst.xy[..., 0]).T
        rows = inst.track_ids
        better = evaluated & (conf > best[rows])
        q_idx, f_idx = np.nonzero(better)
        g_idx = rows[q_idx]
        xy[g_idx, f_idx] = inst.xy[f_idx, q_idx]
        visibility[g_idx, f_idx] = inst.visibility[f_idx, q_idx]
        certainty[g_idx, f_idx] = inst.certainty[f_idx, q_idx]
        best[g_idx, f_idx] = conf[q_idx, f_idx]
    return Tracks(
        xy=xy,
        visibility=visibility,
        certainty=certainty,
        query_frame=query_frame,
        image_names=names,
        image_sizes=sizes,
    )


def _track_backward(inst, model, model_frames, masks, sizes, options):
    """Re-run an instance from its query frame towards frame 0."""
    width, height = sizes[inst.start]
    _, state = model.step(
        model_frames[inst.start],
        query_xy=image_to_model(inst.query_xy, width, height),
    )
    inst.frames_below = 0
    inst.length = 0
    stop = 0
    if options.max_backward_length > 0:
        stop = max(0, inst.start - options.max_backward_length)
    for t in range(inst.start - 1, stop - 1, -1):
        width, height = sizes[t]
        out, state = model.step(model_frames[t], state=state)
        inst.record(
            t,
            model_to_image(out.xy, width, height),
            out.visibility,
            out.certainty,
            width,
            height,
            masks[t],
        )
        if not inst.update_liveness(t, options):
            break
