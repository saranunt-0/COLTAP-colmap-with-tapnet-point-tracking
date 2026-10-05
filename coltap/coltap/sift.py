# SPDX-License-Identifier: BSD-3-Clause
"""COLMAP's own SIFT features and sequential matches, for the hybrid mode.

The hybrid tracker merges two independent correspondence sources into one
COLMAP database: TAPNext++ tracks (long, dense in time, also on weak texture)
and COLMAP's SIFT pipeline (many features on strong texture, wide-baseline
and loop-closure matches). SIFT is run exactly as ``colmap feature_extractor``
+ ``colmap sequential_matcher`` would, into a temporary database whose
keypoints and raw matches are then merged (see ``database.write_database``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import pycolmap

from .bridge import BridgeFeatures

LOGGER = logging.getLogger(__name__)


def colmap_sift_features(
    image_path: str | Path,
    image_names: list[str],
    work_path: str | Path,
    camera_mode: pycolmap.CameraMode,
    reader_options: pycolmap.ImageReaderOptions,
    overlap: int = 10,
    quadratic_overlap: bool = True,
) -> BridgeFeatures:
    """SIFT keypoints ([K, 6], with affine shape) and raw sequential matches.

    Frames are indexed by their position in ``image_names``.
    """
    work_path = Path(work_path)
    work_path.unlink(missing_ok=True)
    pycolmap.Database.open(work_path).close()
    pycolmap.extract_features(
        work_path,
        image_path,
        image_names=image_names,
        camera_mode=camera_mode,
        reader_options=reader_options,
    )
    pycolmap.match_sequential(
        work_path,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=overlap, quadratic_overlap=quadratic_overlap
        ),
    )
    frame_of = {name: f for f, name in enumerate(image_names)}
    keypoints, matches = {}, {}
    with pycolmap.Database.open(work_path) as db:
        frame_of_id = {
            image.image_id: frame_of[image.name]
            for image in db.read_all_images()
        }
        for image_id, f in frame_of_id.items():
            keypoints[f] = db.read_keypoints(image_id)
        for pair_id, pair_matches in zip(*db.read_all_matches(), strict=True):
            id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
            f1, f2 = frame_of_id[id1], frame_of_id[id2]
            if f1 > f2:
                f1, f2 = f2, f1
                pair_matches = pair_matches[:, ::-1]
            matches[(f1, f2)] = np.ascontiguousarray(pair_matches, np.uint32)
    work_path.unlink(missing_ok=True)
    LOGGER.info(
        "SIFT: %.0f keypoints/image, %d matched pairs",
        np.mean([len(k) for k in keypoints.values()]),
        len(matches),
    )
    return BridgeFeatures(keypoints=keypoints, matches=matches)
