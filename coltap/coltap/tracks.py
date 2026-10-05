# SPDX-License-Identifier: BSD-3-Clause
"""Dense track container shared by the tracker, scoring and database writer."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class Tracks:
    """Point tracks over an ordered image sequence.

    All coordinates are full-resolution image pixels in COLMAP's keypoint
    convention (pixel centers at +0.5). Entries for frames a track was not
    evaluated on hold NaN coordinates and zero visibility/certainty.
    """

    xy: np.ndarray  # [N, T, 2] float32
    visibility: np.ndarray  # [N, T] float32, P(visible) from the tracker
    certainty: np.ndarray  # [N, T] float32, localization certainty
    query_frame: np.ndarray  # [N] int32, frame the query was placed on
    image_names: list[str]
    image_sizes: np.ndarray  # [T, 2] int32, (width, height) per frame

    def __post_init__(self):
        n, t = self.visibility.shape
        assert self.xy.shape == (n, t, 2)
        assert self.certainty.shape == (n, t)
        assert self.query_frame.shape == (n,)
        assert len(self.image_names) == t
        assert self.image_sizes.shape == (t, 2)

    @property
    def num_tracks(self) -> int:
        return self.xy.shape[0]

    @property
    def num_frames(self) -> int:
        return self.xy.shape[1]

    @property
    def confidence(self) -> np.ndarray:
        """Per-observation confidence: P(visible) * localization certainty."""
        return self.visibility * self.certainty

    def observed(self, min_confidence: float) -> np.ndarray:
        """[N, T] bool mask of observations that are kept."""
        return np.isfinite(self.xy[..., 0]) & (
            self.confidence >= min_confidence
        )

    def subset(self, keep: np.ndarray) -> Tracks:
        return Tracks(
            xy=self.xy[keep],
            visibility=self.visibility[keep],
            certainty=self.certainty[keep],
            query_frame=self.query_frame[keep],
            image_names=list(self.image_names),
            image_sizes=self.image_sizes.copy(),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            xy=self.xy,
            visibility=self.visibility,
            certainty=self.certainty,
            query_frame=self.query_frame,
            image_names=np.array(self.image_names),
            image_sizes=self.image_sizes,
        )

    @classmethod
    def load(cls, path: str | Path) -> Tracks:
        with np.load(path) as data:
            return cls(
                xy=data["xy"],
                visibility=data["visibility"],
                certainty=data["certainty"],
                query_frame=data["query_frame"],
                image_names=[str(n) for n in data["image_names"]],
                image_sizes=data["image_sizes"],
            )

    @classmethod
    def concatenate(cls, parts: list[Tracks]) -> Tracks:
        if not parts:
            raise ValueError("Nothing to concatenate.")
        return cls(
            xy=np.concatenate([p.xy for p in parts]),
            visibility=np.concatenate([p.visibility for p in parts]),
            certainty=np.concatenate([p.certainty for p in parts]),
            query_frame=np.concatenate([p.query_frame for p in parts]),
            image_names=list(parts[0].image_names),
            image_sizes=parts[0].image_sizes.copy(),
        )
