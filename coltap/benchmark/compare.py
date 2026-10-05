# SPDX-License-Identifier: BSD-3-Clause
"""COLMAP (SIFT + sequential matching) vs COLTAP (TAPNext++) on one sequence.

    python benchmark/compare.py --image_path data/great_wall/images \
        --workspace_path data/great_wall [--gt_path data/x/gt.json] \
        [--gif] [-- <extra coltap options, e.g. --TapTracker.precision bf16>]

Runs both pipelines with identical camera settings and image pairs, then
writes ``stats.json``, ``results.md`` and ``compare.mp4`` (+ ``.gif``) to the
workspace. With ``--gt_path`` (gt.json as written by make_synthetic.py) it
also reports pose errors and, for synthetic scenes, 2D tracking errors.
"""

import argparse
import json
import sys
from pathlib import Path

import pycolmap
from coltap import synthetic
from coltap.cli import main as coltap_main
from coltap.evaluate import pose_errors, reconstruction_stats
from coltap.frames import list_images
from coltap.tracks import Tracks
from coltap.visualize import load_tracks, render_comparison

METHODS = {"sift": "COLMAP (SIFT)", "tapnext": "COLTAP (TAPNext++)"}


def largest_model(sparse: Path):
    models = [pycolmap.Reconstruction(p) for p in sorted(sparse.iterdir())]
    models = [m for m in models if m.num_reg_images() > 0]
    if not models:
        return None, 0, ""
    best = max(range(len(models)), key=lambda k: models[k].num_reg_images())
    return models[best], len(models), str(sorted(sparse.iterdir())[best])


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--workspace_path", required=True)
    parser.add_argument("--gt_path", default="")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.workspace_path)
    names = list_images(args.image_path)
    gt = synthetic.load_ground_truth(args.gt_path) if args.gt_path else None

    stats, model_paths = {}, {}
    for method in METHODS:
        ws = root / f"ws_{method}"
        if not (args.skip_existing and (ws / "sparse").exists()):
            code = coltap_main(
                [
                    "automatic_reconstructor",
                    "--workspace_path",
                    str(ws),
                    "--image_path",
                    args.image_path,
                    "--tracker",
                    method,
                    *extra,
                ]
            )
            if code:
                print(f"{method}: reconstruction failed")
        rec, num_models, path = largest_model(ws / "sparse")
        entry = {"num_models": num_models}
        entry.update(json.loads((ws / "timings.json").read_text()))
        if rec is not None:
            entry.update(reconstruction_stats(rec, len(names)))
            model_paths[method] = path
            if gt is not None:
                entry.update(pose_errors(rec, gt[1]))
        stats[method] = entry

    tracks_file = root / "ws_tapnext" / "tracks.npz"
    if (
        gt is not None
        and tracks_file.exists()
        and "camera" in json.loads(Path(args.gt_path).read_text())
    ):
        try:  # Only meaningful for scenes rendered by make_synthetic.py.
            tracks = Tracks.load(tracks_file)
            scene = synthetic.Scene([synthetic.procedural_texture(64)])
            gt_xy, gt_vis = synthetic.ground_truth_tracks(
                tracks, scene, gt[0], gt[1]
            )
            stats["tapnext_track_accuracy"] = synthetic.track_errors(
                tracks, gt_xy, gt_vis
            )
        except KeyError:
            pass

    (root / "stats.json").write_text(json.dumps(stats, indent=1))
    keys = [k for k in stats["sift"] if k in stats["tapnext"]]
    lines = ["| metric | " + " | ".join(METHODS.values()) + " |"]
    lines.append("|---|" + "---:|" * len(METHODS))
    for key in keys:
        row = [stats[m][key] for m in METHODS]
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if "tapnext_track_accuracy" in stats:
        print("TAPNext++ 2D track accuracy:", stats["tapnext_track_accuracy"])

    if len(model_paths) == 2:
        outputs = [root / "compare.mp4"] + (
            [root / "compare.gif"] if args.gif else []
        )
        render_comparison(
            args.image_path,
            [
                (
                    "COLMAP: SIFT + matching",
                    load_tracks(model_paths["sift"], names),
                ),
                (
                    "COLTAP: TAPNext++ tracks",
                    load_tracks(model_paths["tapnext"], names),
                ),
            ],
            outputs,
            width=360,
            fps=10,
            tail=12,
            image_names=names,
        )


if __name__ == "__main__":
    main()
