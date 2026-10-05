# SPDX-License-Identifier: BSD-3-Clause
"""High-level COLTAP pipeline: track -> COLMAP database -> COLMAP mapper."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import pycolmap

from .bridge import BridgeOptions
from .database import (
    PairingOptions,
    SelectionOptions,
    TrackScores,
    write_database,
)
from .frames import ImageSequence, list_images
from .model import TapNextPP
from .sift import colmap_sift_features
from .tracking import TrackingOptions, track_sequence
from .tracks import Tracks

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelOptions:
    checkpoint: str = ""  # empty: download the official checkpoint
    resolution: int = 256  # 256 or 512 (official checkpoints)
    device: str = "auto"
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    num_threads: int = 0  # torch CPU threads, 0 = torch default
    certainty_radius: float = 8.0


@dataclasses.dataclass
class TrackerOptions:
    model: ModelOptions = dataclasses.field(default_factory=ModelOptions)
    tracking: TrackingOptions = dataclasses.field(
        default_factory=TrackingOptions
    )
    selection: SelectionOptions = dataclasses.field(
        default_factory=SelectionOptions
    )
    pairing: PairingOptions = dataclasses.field(default_factory=PairingOptions)
    bridge: BridgeOptions = dataclasses.field(default_factory=BridgeOptions)
    verification: pycolmap.TwoViewGeometryOptions = dataclasses.field(
        default_factory=pycolmap.TwoViewGeometryOptions
    )
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE
    reader: pycolmap.ImageReaderOptions = dataclasses.field(
        default_factory=pycolmap.ImageReaderOptions
    )
    num_threads: int = -1


TRACKERS = ("tapnext", "sift", "hybrid")


def run_feature_tracker(
    image_path: str | Path,
    database_path: str | Path,
    options: TrackerOptions | None = None,
    image_names: list[str] | None = None,
    tracks_path: str | Path | None = None,
    tracker: str = "tapnext",
) -> tuple[Tracks | None, TrackScores | None]:
    """Drop-in replacement for ``feature_extractor`` + ``sequential_matcher``.

    Writes cameras, images, keypoints, matches and verified two-view
    geometries for the ordered images in ``image_path`` to ``database_path``.

    ``tracker`` selects the correspondence source:

    * ``"tapnext"``: TAPNext++ point tracks (COLTAP);
    * ``"sift"``: stock COLMAP (SIFT + sequential matching);
    * ``"hybrid"``: both, merged into one database. Every image pair is
      verified jointly on TAPNext++ and SIFT correspondences, so TAPNext++
      contributes long tracks and coverage of weak texture, SIFT many
      features on strong texture and wide-baseline matches.

    For the TAPNext++ modes the raw tracks and per-track scores can be saved
    to ``tracks_path`` (``.npz``, scores in ``<stem>_scores.npz``).
    """
    if tracker not in TRACKERS:
        raise ValueError(f"Unknown tracker {tracker!r}, use one of {TRACKERS}")
    options = options or TrackerOptions()
    names = image_names or list_images(image_path)
    if tracker == "sift":
        run_sift_baseline(
            image_path,
            database_path,
            camera_mode=options.camera_mode,
            reader=options.reader,
            pairing=options.pairing,
            image_names=names,
        )
        return None, None
    mask_path = str(options.reader.mask_path)
    sequence = ImageSequence(
        image_path,
        names,
        mask_path=None if mask_path in ("", ".") else mask_path,
    )
    tic = time.time()
    model = TapNextPP(
        checkpoint=options.model.checkpoint or None,
        resolution=options.model.resolution,
        device=options.model.device,
        precision=options.model.precision,
        num_threads=options.model.num_threads,
        certainty_radius=options.model.certainty_radius,
    )
    LOGGER.info(
        "Loaded TAPNext++ (%d px, %s, %s) in %.1fs",
        options.model.resolution,
        model.device,
        model.precision,
        time.time() - tic,
    )
    tracks = track_sequence(sequence, model, options.tracking)
    del model
    extra = None
    if tracker == "hybrid":
        extra = colmap_sift_features(
            image_path,
            names,
            Path(database_path).with_suffix(".sift.db"),
            camera_mode=options.camera_mode,
            reader_options=options.reader,
            overlap=options.pairing.overlap,
            quadratic_overlap=options.pairing.quadratic_overlap,
        )
    scores = write_database(
        tracks,
        database_path,
        image_path,
        camera_mode=options.camera_mode,
        reader_options=options.reader,
        selection=options.selection,
        pairing=options.pairing,
        verification=options.verification,
        num_threads=options.num_threads,
        bridge=options.bridge,
        extra_features=extra,
    )
    if tracks_path:
        tracks_path = Path(tracks_path)
        tracks.save(tracks_path)
        scores.save(tracks_path.with_name(tracks_path.stem + "_scores.npz"))
    return tracks, scores


def run_sift_baseline(
    image_path: str | Path,
    database_path: str | Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
    reader: pycolmap.ImageReaderOptions | None = None,
    pairing: PairingOptions | None = None,
    image_names: list[str] | None = None,
) -> None:
    """Stock COLMAP correspondence search (SIFT + sequential matching)."""
    pairing = pairing or PairingOptions()
    pycolmap.extract_features(
        database_path,
        image_path,
        image_names=image_names or list_images(image_path),
        camera_mode=camera_mode,
        reader_options=reader or pycolmap.ImageReaderOptions(),
    )
    pycolmap.match_sequential(
        database_path,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=pairing.overlap,
            quadratic_overlap=pairing.quadratic_overlap,
        ),
    )


def run_mapper(
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    options: pycolmap.IncrementalPipelineOptions | None = None,
    input_path: str | Path = "",
) -> dict[int, pycolmap.Reconstruction]:
    """Same as ``colmap mapper`` (COLMAP's incremental SfM, unmodified)."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return pycolmap.incremental_mapping(
        database_path,
        image_path,
        output_path,
        options=options or pycolmap.IncrementalPipelineOptions(),
        input_path=input_path,
    )


def automatic_reconstruction(
    workspace_path: str | Path,
    image_path: str | Path,
    tracker: str = "tapnext",
    tracker_options: TrackerOptions | None = None,
    mapper_options: pycolmap.IncrementalPipelineOptions | None = None,
) -> dict[int, pycolmap.Reconstruction]:
    """Sparse reconstruction with COLMAP's workspace layout.

    ``workspace/database.db``, ``workspace/sparse/<k>/`` and, for the
    TAPNext++ trackers, ``workspace/tracks.npz`` + ``tracks_scores.npz``.
    ``tracker`` is ``"tapnext"``, ``"sift"`` or ``"hybrid"``.
    """
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    timings = {}
    tic = time.time()
    run_feature_tracker(
        image_path,
        database_path,
        tracker_options,
        tracks_path=workspace / "tracks.npz",
        tracker=tracker,
    )
    timings["correspondences_s"] = time.time() - tic
    tic = time.time()
    reconstructions = run_mapper(
        database_path, image_path, workspace / "sparse", mapper_options
    )
    timings["mapping_s"] = time.time() - tic
    (workspace / "timings.json").write_text(json.dumps(timings, indent=1))
    for idx, rec in reconstructions.items():
        LOGGER.info("Model %d: %s", idx, rec.summary())
    return reconstructions
