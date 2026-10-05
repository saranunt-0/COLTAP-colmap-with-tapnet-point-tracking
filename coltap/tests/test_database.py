# SPDX-License-Identifier: BSD-3-Clause
"""Database export, static scoring and COLMAP mapping on exact tracks."""

import cv2
import numpy as np

import pycolmap
from coltap.database import (
    PairingOptions,
    SelectionOptions,
    sequential_pairs,
    write_database,
)
from coltap.evaluate import largest_reconstruction, pose_errors
from coltap.pipeline import run_mapper
from coltap.synthetic import look_at
from coltap.tracks import Tracks

WIDTH, HEIGHT, FOCAL = 320, 240, 260.0


def _make_sequence(tmp_path, num_frames=16, num_static=400, num_moving=80):
    rng = np.random.default_rng(1)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    names, poses = [], []
    for i in range(num_frames):
        name = f"frame_{i:03d}.png"
        cv2.imwrite(
            str(image_dir / name), np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        )
        center = np.array(
            [-1.0 + 2.0 * i / (num_frames - 1), 0.1 * i / num_frames, 0]
        )
        poses.append(look_at(center, np.array([0.0, 0.0, 6.0])))
        names.append(name)

    static = np.stack(
        [
            rng.uniform(-3, 3, num_static),
            rng.uniform(-2, 2, num_static),
            rng.uniform(4, 9, num_static),
        ],
        axis=1,
    )
    moving0 = np.stack(
        [
            rng.uniform(-2, 2, num_moving),
            rng.uniform(-1.5, 1.5, num_moving),
            rng.uniform(4, 7, num_moving),
        ],
        axis=1,
    )
    # Independently moving points (think pedestrians), world units per frame.
    direction = rng.normal(0, 1, (num_moving, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    velocity = direction * rng.uniform(0.1, 0.2, (num_moving, 1))

    n = num_static + num_moving
    xy = np.full((n, num_frames, 2), np.nan, np.float32)
    for f, (rot, trans) in enumerate(poses):
        points = np.concatenate([static, moving0 + f * velocity])
        cam = points @ rot.T + trans
        proj = FOCAL * cam[:, :2] / cam[:, 2:] + [WIDTH / 2, HEIGHT / 2]
        proj += rng.normal(0, 0.3, proj.shape)
        inside = (
            (cam[:, 2] > 0)
            & (proj[:, 0] >= 0)
            & (proj[:, 0] < WIDTH)
            & (proj[:, 1] >= 0)
            & (proj[:, 1] < HEIGHT)
        )
        xy[inside, f] = proj[inside]
    visible = np.isfinite(xy[..., 0]).astype(np.float32)
    tracks = Tracks(
        xy=xy,
        visibility=visible,
        certainty=visible.copy(),
        query_frame=np.zeros(n, np.int32),
        image_names=names,
        image_sizes=np.tile([WIDTH, HEIGHT], (num_frames, 1)).astype(np.int32),
    )
    is_moving = np.r_[np.zeros(num_static, bool), np.ones(num_moving, bool)]
    gt = dict(zip(names, poses, strict=True))
    return image_dir, tracks, is_moving, gt


def test_sequential_pairs_matches_colmap_schedule():
    pairs = sequential_pairs(
        6, PairingOptions(overlap=2, quadratic_overlap=True)
    )
    assert pairs == [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
    ]
    pairs = sequential_pairs(
        20, PairingOptions(overlap=3, quadratic_overlap=True)
    )
    assert (0, 4) in pairs and (0, 5) not in pairs


def _verification():
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.random_seed = 0
    return options


def test_database_layout_and_static_selection(tmp_path):
    image_dir, tracks, is_moving, _ = _make_sequence(tmp_path)
    database_path = tmp_path / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"
    reader.camera_params = f"{FOCAL},{WIDTH / 2},{HEIGHT / 2}"
    scores = write_database(
        tracks,
        database_path,
        image_dir,
        reader_options=reader,
        selection=SelectionOptions(min_static_score=0.5),
        pairing=PairingOptions(overlap=5),
        verification=_verification(),
    )
    # Moving points violate the epipolar geometry of the static scene.
    assert scores.static_score[~is_moving].mean() > 0.95
    assert scores.static_score[is_moving].mean() < 0.5
    assert scores.selected[~is_moving].mean() > 0.95
    assert scores.selected[is_moving].mean() < 0.3

    with pycolmap.Database.open(database_path) as db:
        assert db.num_cameras() == 1
        assert db.num_images() == tracks.num_frames
        assert db.num_keypoints() > 0
        assert db.num_matched_image_pairs() > 0
        assert db.num_verified_image_pairs() > 0
        # Every written keypoint belongs to a selected track.
        observed = tracks.observed(0.5) & scores.selected[:, None]
        assert db.num_keypoints() == observed.sum()
        # Inlier matches only reference selected (static) tracks.
        image_ids = {im.name: im.image_id for im in db.read_all_images()}
        name0, name1 = tracks.image_names[0], tracks.image_names[1]
        geometry = db.read_two_view_geometry(image_ids[name0], image_ids[name1])
        kps0 = db.read_keypoints(image_ids[name0])
        rows0 = np.nonzero(observed[:, 0])[0]
        rows1 = np.nonzero(observed[:, 1])[0]
        matched0 = rows0[geometry.inlier_matches[:, 0]]
        matched1 = rows1[geometry.inlier_matches[:, 1]]
        # Matches connect observations of the same track...
        np.testing.assert_array_equal(matched0, matched1)
        # ...and only tracks that passed the static selection.
        assert scores.selected[matched0].all()
        assert is_moving[matched0].mean() < 0.1
        np.testing.assert_allclose(kps0[:, :2], tracks.xy[rows0, 0], atol=1e-4)


def test_colmap_mapper_reconstructs_from_tracks(tmp_path):
    image_dir, tracks, _, gt = _make_sequence(tmp_path, num_frames=12)
    database_path = tmp_path / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"
    reader.camera_params = f"{FOCAL},{WIDTH / 2},{HEIGHT / 2}"
    write_database(
        tracks,
        database_path,
        image_dir,
        reader_options=reader,
        verification=_verification(),
    )
    recs = run_mapper(database_path, image_dir, tmp_path / "sparse")
    rec = largest_reconstruction(recs)
    assert rec is not None
    assert rec.num_reg_images() == tracks.num_frames
    errors = pose_errors(rec, gt)
    assert errors["rre_max_deg"] < 0.5
    assert errors["ate_rmse_rel"] < 0.01
    # Output is a regular COLMAP model on disk.
    assert (tmp_path / "sparse" / "0" / "points3D.bin").exists()
