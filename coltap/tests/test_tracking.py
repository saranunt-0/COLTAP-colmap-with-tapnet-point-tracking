# SPDX-License-Identifier: BSD-3-Clause
"""Tracking-loop logic with a fake tracker (no network weights needed)."""

import numpy as np

from coltap import model as model_lib
from coltap.model import StepOutput
from coltap.tracking import TrackingOptions, track_sequence


class _TranslatingSequence:
    """Images of a texture shifting right by ``speed`` px per frame."""

    def __init__(self, texture, num_frames, width, height, speed):
        self.texture = texture
        self.width, self.height = width, height
        self.speed = speed
        self.names = [f"frame_{i:03d}.png" for i in range(num_frames)]

    def __len__(self):
        return len(self.names)

    def image(self, index):
        x0 = int(round(200 - self.speed * index))
        return self.texture[50 : 50 + self.height, x0 : x0 + self.width].copy()

    def mask(self, index, shape):
        return None


class _FakeModel:
    """Returns the exact motion of _TranslatingSequence in model coords.

    ``preprocess`` is called once per frame in temporal order, so it tags each
    frame with its index; ``step`` then knows the true displacement between
    the query frame and the current frame in either time direction.
    """

    def __init__(self, sequence):
        self.sequence = sequence
        self.steps = 0
        self._next_index = 0

    def preprocess(self, image):
        frame = {"index": self._next_index, "image": image}
        self._next_index += 1
        return frame

    def step(self, frame, query_xy=None, state=None):
        self.steps += 1
        seq = self.sequence
        if state is None:
            state = {"xy0": query_xy.copy(), "t0": frame["index"]}
        dt = frame["index"] - state["t0"]
        shift = seq.speed * dt * model_lib.MODEL_COORD_SIZE / seq.width
        xy = state["xy0"] + np.array([shift, 0], np.float32)
        q = len(xy)
        out = StepOutput(
            xy=xy.astype(np.float32),
            visibility=np.ones(q, np.float32),
            certainty=np.ones(q, np.float32),
        )
        return out, state


def _texture():
    rng = np.random.default_rng(0)
    tex = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
    import cv2

    return cv2.resize(tex, (600, 400), interpolation=cv2.INTER_CUBIC)


def test_tracks_follow_motion_and_new_content_is_seeded():
    seq = _TranslatingSequence(_texture(), 30, 160, 120, speed=3.0)
    fake = _FakeModel(seq)
    options = TrackingOptions(grid_cells=8, max_active_instances=2)
    tracks = track_sequence(seq, fake, options)

    assert tracks.num_frames == 30
    observed = tracks.observed(0.5)
    # Every frame is covered by tracks.
    assert observed.sum(axis=0).min() >= 10
    # Tracks move right by `speed` px per frame relative to their query.
    for n in range(0, tracks.num_tracks, 7):
        frames = np.nonzero(observed[n])[0]
        q = tracks.query_frame[n]
        q_xy = tracks.xy[n, q]
        for f in frames:
            expected = q_xy + np.array([3.0 * (f - q), 0])
            np.testing.assert_allclose(tracks.xy[n, f], expected, atol=0.75)
    # New content entering on the left forces re-seeding.
    assert len(np.unique(tracks.query_frame)) > 1
    # Points leaving on the right are marked as not visible.
    assert not np.any(observed & (tracks.xy[..., 0] >= 160))


def test_instance_cap_hands_off_tracks():
    seq = _TranslatingSequence(_texture(), 40, 160, 120, speed=2.0)
    fake = _FakeModel(seq)
    options = TrackingOptions(
        grid_cells=8,
        max_active_instances=1,
        seed_coverage=0.95,
        min_keyframe_interval=5,
    )
    tracks = track_sequence(seq, fake, options)
    observed = tracks.observed(0.5)
    lengths = observed.sum(axis=1)
    # With a single instance at a time, long tracks only exist via handoff.
    assert lengths.max() > 20
    # Cost bound: at most one instance advances per frame, plus one
    # initialization step per keyframe.
    keyframes = len(np.unique(tracks.query_frame))
    assert fake.steps <= 40 + keyframes + 8


def test_bidirectional_recovers_frames_before_query():
    seq = _TranslatingSequence(_texture(), 20, 160, 120, speed=3.0)
    forward = track_sequence(
        seq, _FakeModel(seq), TrackingOptions(grid_cells=8)
    )
    both = track_sequence(
        seq,
        _FakeModel(seq),
        TrackingOptions(grid_cells=8, bidirectional=True),
    )
    late = forward.query_frame > 0
    assert late.any()
    before = np.arange(20)[None] < forward.query_frame[:, None]
    assert not forward.observed(0.5)[before].any()
    recovered = both.observed(0.5) & before
    assert recovered.any()
    # Backward positions follow the true motion as well.
    n, f = np.nonzero(recovered)
    q = both.query_frame[n]
    expected = both.xy[n, q] + np.stack([3.0 * (f - q), 0 * f], axis=1)
    np.testing.assert_allclose(both.xy[n, f], expected, atol=0.75)


def test_masked_regions_get_no_queries_or_observations():
    class _MaskedSequence(_TranslatingSequence):
        def mask(self, index, shape):
            mask = np.ones(shape, bool)
            mask[:, : shape[1] // 2] = False  # left half is "dynamic"
            return mask

    seq = _MaskedSequence(_texture(), 12, 160, 120, speed=3.0)
    tracks = track_sequence(seq, _FakeModel(seq), TrackingOptions(grid_cells=8))
    observed = tracks.observed(0.5)
    assert observed.any()
    assert (tracks.xy[observed][:, 0] >= 80).all()
