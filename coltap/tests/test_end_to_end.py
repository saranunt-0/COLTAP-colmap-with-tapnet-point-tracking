# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end CLI run with the real TAPNext++ weights.

Skipped unless the checkpoint is already cached (it is a 2.5 GB download),
or COLTAP_RUN_MODEL_TESTS=1 is set to allow downloading it.
"""

import os

import numpy as np
import pytest

import pycolmap
from coltap import synthetic
from coltap.cli import main
from coltap.evaluate import largest_reconstruction, pose_errors
from coltap.model import default_cache_dir
from coltap.synthetic import ground_truth_tracks, track_errors
from coltap.tracks import Tracks


def _have_weights() -> bool:
    cache = default_cache_dir()
    return (
        os.environ.get("COLTAP_RUN_MODEL_TESTS") == "1"
        or (cache / "tapnextpp_256_weights.pt").exists()
        or (cache / "tapnextpp_ckpt.pt").exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_weights(), reason="TAPNext++ checkpoint not cached"
)


@pytest.mark.parametrize("tracker", ["tapnext", "hybrid"])
def test_cli_automatic_reconstructor_on_synthetic_scene(tmp_path, tracker):
    textures = [synthetic.procedural_texture(512, seed=s) for s in range(6)]
    scene = synthetic.Scene(textures, texels_per_unit=120.0)
    camera = synthetic.Camera(width=320, height=240, focal=250.0)
    poses = synthetic.orbit_trajectory(12, arc_degrees=20.0)
    synthetic.write_sequence(tmp_path / "scene", scene, camera, poses)

    workspace = tmp_path / "ws"
    code = main(
        [
            "automatic_reconstructor",
            "--image_path",
            str(tmp_path / "scene" / "images"),
            "--workspace_path",
            str(workspace),
            "--TapTracker.grid_cells",
            "16",
            "--tracker",
            tracker,
        ]
    )
    assert code == 0

    # Standard COLMAP outputs.
    with pycolmap.Database.open(workspace / "database.db") as db:
        assert db.num_images() == 12
        assert db.num_keypoints() > 0
        assert db.num_verified_image_pairs() > 0
        num_keypoints = db.num_keypoints()
    scores = np.load(workspace / "tracks_scores.npz")
    num_track_keypoints = int(scores["num_track_keypoints"].sum())
    if tracker == "hybrid":
        assert num_keypoints > num_track_keypoints  # SIFT appended
        assert not (workspace / "database.sift.db").exists()
    else:
        assert num_keypoints == num_track_keypoints
    rec = largest_reconstruction(
        {
            int(p.name): pycolmap.Reconstruction(p)
            for p in (workspace / "sparse").iterdir()
        }
    )
    assert rec.num_reg_images() == 12

    _, gt = synthetic.load_ground_truth(tmp_path / "scene" / "gt.json")
    errors = pose_errors(rec, gt)
    assert errors["rre_max_deg"] < 0.5
    assert errors["ate_rmse_rel"] < 0.02

    # 2D tracking accuracy against exact ground truth.
    tracks = Tracks.load(workspace / "tracks.npz")
    gt_xy, gt_visible = ground_truth_tracks(tracks, scene, camera, gt)
    stats = track_errors(tracks, gt_xy, gt_visible)
    assert stats["visibility_precision"] > 0.95
    assert stats["error_median_px"] < 1.0
    assert np.isfinite(stats["error_mean_px"])
