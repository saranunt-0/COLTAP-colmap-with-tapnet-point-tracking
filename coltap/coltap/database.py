# SPDX-License-Identifier: BSD-3-Clause
"""Turn point tracks into a standard COLMAP database.

The database written here has exactly the layout produced by
``colmap feature_extractor`` + ``colmap sequential_matcher``:

* ``cameras`` / ``images`` / ``frames`` / ``rigs``: created by COLMAP's own
  image reader (``pycolmap.import_images``), so camera models, EXIF focal
  priors and ``--ImageReader.*`` options behave exactly as in COLMAP;
* ``keypoints``: one keypoint per (track, image) observation;
* ``matches``: raw correspondences between image pairs that share tracks;
* ``two_view_geometries``: COLMAP's own geometric verification
  (``pycolmap.estimate_two_view_geometry``) of those matches.

Where the track graph is broken (a cut or a jump in the input), SIFT matches
are added in a small window around the break (``bridge.py``). In hybrid mode
COLMAP's SIFT features and matches are merged for every image pair
(``sift.py``).

Descriptors are not written: the mapper and all later stages do not use them.

Track selection ("which tracks are trusted for reconstruction") happens here:

1. observation level: keep observations with confidence
   ``P(visible) * certainty >= min_confidence``;
2. track level: drop tracks with fewer than ``min_track_length`` kept
   observations;
3. static score: fraction of verified image pairs in which the track is an
   inlier of the two-view geometry. Points on independently moving objects
   violate the epipolar constraint of the (static) background and get a low
   score. Tracks below ``min_static_score`` are removed from all inlier sets;
4. weight = mean observation confidence * static score, used to rank tracks
   when ``max_tracks_per_image`` caps the number of keypoints per image.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import pycolmap

from .bridge import BridgeFeatures, BridgeOptions, compute_bridges
from .frames import ImageSequence
from .tracks import Tracks

LOGGER = logging.getLogger(__name__)

_INVALID_CONFIGS = {
    pycolmap.TwoViewGeometryConfiguration.UNDEFINED,
    pycolmap.TwoViewGeometryConfiguration.DEGENERATE,
    pycolmap.TwoViewGeometryConfiguration.WATERMARK,
}


@dataclasses.dataclass
class SelectionOptions:
    min_confidence: float = 0.5
    min_track_length: int = 3
    min_static_score: float = 0.5
    max_tracks_per_image: int = 0  # 0 = keep all


@dataclasses.dataclass
class PairingOptions:
    """Same semantics as COLMAP's sequential matcher pairing."""

    overlap: int = 10
    quadratic_overlap: bool = True


@dataclasses.dataclass
class TrackScores:
    """Per-track statistics computed during database export."""

    num_observations: np.ndarray  # [N] kept observations
    mean_confidence: np.ndarray  # [N]
    num_verified_pairs: np.ndarray  # [N] pairs where the track was matched
    num_inlier_pairs: np.ndarray  # [N] pairs where it was a geometric inlier
    static_score: np.ndarray  # [N] inlier_pairs / verified_pairs
    weight: np.ndarray  # [N]
    selected: np.ndarray  # [N] bool, written to the database
    # [T] keypoints per image that come from tracks; any keypoints after
    # them are SIFT features (hybrid mode or gap bridge).
    num_track_keypoints: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **dataclasses.asdict(self))


def sequential_pairs(num_images: int, options: PairingOptions):
    """Image index pairs (i < j) as made by COLMAP's sequential matcher.

    With ``quadratic_overlap`` image i is paired with i + 2^k for
    k < ``overlap`` (i.e. i+1, i+2, i+4, ...); otherwise with i+1 ... i+overlap
    (src/colmap/controllers/pairing.cc).
    """
    pairs = []
    for i in range(num_images):
        for k in range(options.overlap):
            j = i + (1 << k) if options.quadratic_overlap else i + k + 1
            if j >= num_images:
                break
            pairs.append((i, j))
    return pairs


def _import_images(
    database_path: Path,
    image_path: Path,
    image_names: list[str],
    camera_mode: pycolmap.CameraMode,
    reader_options: pycolmap.ImageReaderOptions,
) -> dict[str, pycolmap.Image]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    pycolmap.Database.open(database_path).close()  # create if missing
    pycolmap.import_images(
        database_path,
        image_path,
        camera_mode=camera_mode,
        image_names=image_names,
        options=reader_options,
    )
    with pycolmap.Database.open(database_path) as db:
        images = {image.name: image for image in db.read_all_images()}
    missing = [name for name in image_names if name not in images]
    if missing:
        raise RuntimeError(
            f"{len(missing)} images were not imported: {missing[:3]}"
        )
    return images


def _verify(pair_inputs, options: pycolmap.TwoViewGeometryOptions, threads):
    def run(args):
        cam1, pts1, cam2, pts2, matches = args
        return pycolmap.estimate_two_view_geometry(
            cam1, pts1, cam2, pts2, matches, options
        )

    with ThreadPoolExecutor(max_workers=threads) as executor:
        return list(executor.map(run, pair_inputs))


def write_database(
    tracks: Tracks,
    database_path: str | Path,
    image_path: str | Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
    reader_options: pycolmap.ImageReaderOptions | None = None,
    selection: SelectionOptions | None = None,
    pairing: PairingOptions | None = None,
    verification: pycolmap.TwoViewGeometryOptions | None = None,
    num_threads: int = -1,
    bridge: BridgeOptions | None = None,
    extra_features: BridgeFeatures | None = None,
) -> TrackScores:
    """Write tracks as keypoints + verified matches into a COLMAP database.

    ``extra_features`` (hybrid mode) are additional keypoints and raw matches
    per frame pair, e.g. COLMAP's SIFT features (``sift.py``). They are
    appended to each image's keypoints, and every pair they cover is verified
    jointly with the track matches. The gap bridge is not needed then.
    """
    database_path = Path(database_path)
    selection = selection or SelectionOptions()
    pairing = pairing or PairingOptions()
    verification = verification or pycolmap.TwoViewGeometryOptions()
    reader_options = reader_options or pycolmap.ImageReaderOptions()
    bridge = bridge or BridgeOptions()
    threads = num_threads if num_threads > 0 else (os.cpu_count() or 1)

    names = tracks.image_names
    images = _import_images(
        database_path, Path(image_path), names, camera_mode, reader_options
    )
    with pycolmap.Database.open(database_path) as db:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        if db.num_keypoints() or db.num_matches():
            LOGGER.warning(
                "Database already has keypoints/matches; replacing them."
            )
            db.clear_keypoints()
            db.clear_descriptors()
            db.clear_matches()
            db.clear_two_view_geometries()
    image_cams = [cameras[images[name].camera_id] for name in names]
    for f, cam in enumerate(image_cams):
        if (cam.width, cam.height) != tuple(tracks.image_sizes[f]):
            raise RuntimeError(
                f"Image {names[f]}: COLMAP reads it as {cam.width}x"
                f"{cam.height} but it was tracked at "
                f"{tracks.image_sizes[f][0]}x{tracks.image_sizes[f][1]} "
                "(EXIF orientation?)."
            )

    # 1-2. Observation and track filtering.
    observed = tracks.observed(selection.min_confidence)
    num_obs = observed.sum(axis=1)
    candidate = num_obs >= selection.min_track_length
    observed &= candidate[:, None]
    conf = np.where(observed, tracks.confidence, 0.0)
    mean_conf = conf.sum(axis=1) / np.maximum(num_obs, 1)

    # 3. Static score from COLMAP's two-view geometric verification. The
    # score is a vote over image pairs, so it uses more pairs than are
    # written to the database: COLMAP's schedule plus the dense window
    # i+1 ... i+overlap (more votes per track, better separation of moving
    # points). Only COLMAP's schedule is written, as its matcher would.
    pairs = sequential_pairs(tracks.num_frames, pairing)
    score_pairs = sorted(
        set(pairs)
        | set(
            sequential_pairs(
                tracks.num_frames,
                PairingOptions(pairing.overlap, quadratic_overlap=False),
            )
        )
    )
    pair_tracks, pair_inputs, pair_ids = [], [], []
    xy64 = tracks.xy.astype(np.float64)
    for i, j in score_pairs:
        shared = np.nonzero(observed[:, i] & observed[:, j])[0]
        if len(shared) < verification.min_num_inliers:
            continue
        local = np.arange(len(shared), dtype=np.uint32)
        pair_tracks.append(shared)
        pair_ids.append((i, j))
        pair_inputs.append(
            (
                image_cams[i],
                xy64[shared, i],
                image_cams[j],
                xy64[shared, j],
                np.stack([local, local], axis=1),
            )
        )
    LOGGER.info("Verifying %d image pairs", len(pair_inputs))
    geometries = _verify(pair_inputs, verification, threads)

    num_verified = np.zeros(tracks.num_tracks, np.int32)
    num_inlier = np.zeros(tracks.num_tracks, np.int32)
    for shared, geometry in zip(pair_tracks, geometries, strict=True):
        if geometry.config in _INVALID_CONFIGS:
            continue
        num_verified[shared] += 1
        inliers = geometry.inlier_matches[:, 0]
        num_inlier[shared[inliers]] += 1
    static = np.where(
        num_verified > 0, num_inlier / np.maximum(num_verified, 1), 0.0
    )
    weight = mean_conf * static
    selected = candidate & (static >= selection.min_static_score)

    # 4. Optional per-image cap by weight.
    if selection.max_tracks_per_image > 0:
        order = np.argsort(-weight)
        for f in range(tracks.num_frames):
            in_image = order[(observed[order, f] & selected[order])]
            observed[in_image[selection.max_tracks_per_image :], f] = False
        observed &= selected[:, None]
        selected &= observed.sum(axis=1) >= 2
    observed &= selected[:, None]

    # 5. Extra features: hybrid SIFT, or the gap bridge (see bridge.py).
    bridges = extra_features
    if bridges is None and bridge is not None and bridge.enabled:
        sequence = ImageSequence(image_path, names)
        bridges = compute_bridges(observed, sequence, bridge, pairs)

    # Keypoints: kept track observations first, then the extra features.
    keypoint_index = np.full(observed.shape, -1, np.int64)
    keypoints, sift_offset = [], np.zeros(tracks.num_frames, np.uint32)
    for f in range(tracks.num_frames):
        rows = np.nonzero(observed[:, f])[0]
        keypoint_index[rows, f] = np.arange(len(rows))
        kps = tracks.xy[rows, f]
        sift_offset[f] = len(rows)
        if bridges is not None and f in bridges.keypoints:
            kps = _stack_keypoints(kps, bridges.keypoints[f])
        keypoints.append(kps.astype(np.float32))

    # Final matches per pair: (raw matches, two-view geometry).
    final = {}
    database_pairs = set(pairs)
    for (i, j), shared, geometry in zip(
        pair_ids, pair_tracks, geometries, strict=True
    ):
        if (i, j) not in database_pairs:
            continue
        kp_i = keypoint_index[shared, i]
        kp_j = keypoint_index[shared, j]
        raw = (kp_i >= 0) & (kp_j >= 0)
        if not raw.any():
            continue
        if geometry.config not in _INVALID_CONFIGS:
            local = geometry.inlier_matches[:, 0]
            local = local[(kp_i[local] >= 0) & (kp_j[local] >= 0)]
            geometry.inlier_matches = np.stack(
                [kp_i[local], kp_j[local]], 1
            ).astype(np.uint32)
        final[(i, j)] = (
            np.stack([kp_i[raw], kp_j[raw]], 1).astype(np.uint32),
            geometry,
        )
    if bridges is not None:
        bridge_keys, bridge_inputs = [], []
        for (i, j), sift_matches in bridges.matches.items():
            shared = np.nonzero(observed[:, i] & observed[:, j])[0]
            tap = np.stack(
                [keypoint_index[shared, i], keypoint_index[shared, j]], 1
            )
            matches = np.concatenate(
                [tap, sift_matches + [sift_offset[i], sift_offset[j]]]
            ).astype(np.uint32)
            if len(matches) < verification.min_num_inliers:
                continue
            bridge_keys.append((i, j, matches))
            bridge_inputs.append(
                (
                    image_cams[i],
                    keypoints[i][:, :2].astype(np.float64),
                    image_cams[j],
                    keypoints[j][:, :2].astype(np.float64),
                    matches,
                )
            )
        for (i, j, matches), geometry in zip(
            bridge_keys,
            _verify(bridge_inputs, verification, threads),
            strict=True,
        ):
            final[(i, j)] = (matches, geometry)

    with pycolmap.Database.open(database_path) as db:
        for f, name in enumerate(names):
            db.write_keypoints(images[name].image_id, keypoints[f])
        for (i, j), (matches, geometry) in sorted(final.items()):
            id_i, id_j = images[names[i]].image_id, images[names[j]].image_id
            db.write_matches(id_i, id_j, matches)
            db.write_two_view_geometry(id_i, id_j, geometry)
    num_written = len(final)

    LOGGER.info(
        "Database: %d/%d tracks selected (%d candidates), %d track + %d "
        "SIFT keypoints, %d image pairs",
        selected.sum(),
        tracks.num_tracks,
        candidate.sum(),
        observed.sum(),
        sum(len(k) for k in keypoints) - observed.sum(),
        num_written,
    )
    return TrackScores(
        num_observations=num_obs.astype(np.int32),
        mean_confidence=mean_conf.astype(np.float32),
        num_verified_pairs=num_verified,
        num_inlier_pairs=num_inlier,
        static_score=static.astype(np.float32),
        weight=weight.astype(np.float32),
        selected=selected,
        num_track_keypoints=sift_offset.astype(np.int32),
    )


def _stack_keypoints(track_xy: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Concatenate track keypoints (x, y) with extra keypoints.

    If the extra keypoints carry an affine shape ([K, 6], as COLMAP's SIFT
    does), track keypoints get an identity shape so all rows have 6 columns.
    """
    if extra.shape[1] == 2:
        return np.concatenate([track_xy, extra])
    shape = np.tile([1.0, 0.0, 0.0, 1.0], (len(track_xy), 1))
    return np.concatenate([np.hstack([track_xy, shape]), extra[:, :6]])
