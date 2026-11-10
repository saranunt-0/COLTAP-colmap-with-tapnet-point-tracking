# SPDX-License-Identifier: BSD-3-Clause
"""``siftap`` command line: track selected points/areas in a video.

    siftap track --video clip.mp4 --select --output tracked.mp4
    siftap track --video clip.mp4 --point 320,180 --box 100,80,60,40 \
        --output tracked.mp4 --json tracks.json

Targets are given on the first (kept) frame: ``--point x,y`` or
``--box x,y,w,h`` (repeatable), or interactively with ``--select``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from .fusion import FUSIONS
from .render import render_frames, write_video
from .sift import SiftParams
from .tap import TapCalibration
from .targets import parse_target
from .tracker import SiftapTracker, TrackerOptions

LOGGER = logging.getLogger("siftap")


def read_video(path, stride=1, max_frames=0, start=0, max_size=0):
    """Frames (RGB), frame rate after striding, and the resize factor."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames, index, scale = [], -1, 1.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index < start or (index - start) % max(stride, 1):
            continue
        if max_size and max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    capture.release()
    return frames, fps / max(stride, 1), scale


def records_to_json(records, targets, names, scale=1.0, start=0, stride=1):
    """Tracks as JSON in pixels of the original video (``scale`` is the
    resize factor applied when reading it); ``video_frame`` is the frame
    index in the video, ``frame`` the index among the processed frames."""

    def xy(p):
        return [float(v) / scale for v in p]

    out = []
    for k, recs in enumerate(records):
        rows = []
        for r in recs:
            lost = r.fused.lost
            row = {
                "frame": r.frame,
                "video_frame": start + r.frame * stride,
                "fused": None if lost else xy(r.fused.position),
                "fused_r95": None if lost else r.fused.radius() / scale,
            }
            for m in (r.sift, r.tap):
                if m is None:
                    continue
                row[m.source] = xy(m.position)
                row[f"{m.source}_r95"] = m.radius() / scale
                if m.p_outlier:
                    row[f"{m.source}_p_outlier"] = m.p_outlier
                row[f"{m.source}_accepted"] = bool(
                    r.fused.accepted.get(m.source, False)
                )
            rows.append(row)
        t = targets[k]
        out.append(
            {
                "name": names[k],
                "type": "area" if t.is_area else "point",
                "center": xy(t.center),
                "box": None if t.box is None else xy(t.box),
                "frames": rows,
            }
        )
    return out


def cmd_track(argv):
    p = argparse.ArgumentParser(
        prog="siftap track",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument(
        "--point", action="append", default=[], help="x,y (video pixels)"
    )
    p.add_argument(
        "--box", action="append", default=[], help="x,y,w,h (video pixels)"
    )
    p.add_argument(
        "--name", action="append", default=[], help="label per target"
    )
    p.add_argument(
        "--select", action="store_true", help="click/drag on frame 0"
    )
    p.add_argument(
        "--radius",
        type=float,
        default=24.0,
        help="neighbourhood radius of point targets (video pixels)",
    )
    p.add_argument("--fusion", default="kalman", choices=FUSIONS)
    p.add_argument(
        "--accel_sigma", type=float, default=TrackerOptions.accel_sigma
    )
    p.add_argument("--no_sift", action="store_true")
    p.add_argument("--no_tap", action="store_true")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--max_size", type=int, default=960)
    p.add_argument("--precision", default="auto", help="auto|fp32|bf16|fp16")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="", help=".mp4 or .gif")
    p.add_argument("--output_width", type=int, default=0)
    p.add_argument("--json", default="")
    p.add_argument(
        "--tap_cache",
        default="",
        help="pickle of the raw TAPNext++ output: reused when the video, "
        "frames and targets match (fast re-runs of fusion/rendering)",
    )
    args = p.parse_args(argv)

    frames, fps, scale = read_video(
        args.video, args.stride, args.max_frames, args.start, args.max_size
    )
    if not frames:
        raise SystemExit("No frames read.")
    # --point/--box/--radius are in pixels of the original video.
    targets = [
        parse_target(s, args.radius).scaled(scale)
        for s in args.point + args.box
    ]
    if args.select:
        from .select import select_targets

        targets += select_targets(frames[0], args.radius)
    if not targets:
        raise SystemExit("No targets: use --point, --box or --select.")
    names = [
        args.name[k] if k < len(args.name) else (t.name or f"T{k + 1}")
        for k, t in enumerate(targets)
    ]
    model, tap_raw = None, None
    cache_key = {
        "video": str(Path(args.video).resolve()),
        "frames": [args.start, args.stride, len(frames), *frames[0].shape],
        "targets": [
            [*map(float, t.center), t.radius, t.is_area] for t in targets
        ],
    }
    if args.tap_cache and Path(args.tap_cache).exists():
        with open(args.tap_cache, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            tap_raw = cached["raw"]
            LOGGER.info("using cached TAPNext++ output %s", args.tap_cache)
        else:
            LOGGER.info("TAP cache does not match video/targets; recomputing")
    if not args.no_tap and tap_raw is None:
        from coltap.model import TapNextPP

        model = TapNextPP(device=args.device, precision=args.precision)
    options = TrackerOptions(
        fusion=args.fusion,
        accel_sigma=args.accel_sigma,
        use_tap=not args.no_tap,
        use_sift=not args.no_sift,
    )
    tracker = SiftapTracker(
        targets, model, options, TapCalibration(), SiftParams()
    )
    records = tracker.run(frames, tap_raw=tap_raw)
    if args.tap_cache and model is not None:
        with open(args.tap_cache, "wb") as f:
            pickle.dump({"key": cache_key, "raw": tracker.tap_raw}, f)
    for k, recs in enumerate(records):
        radii = [r.fused.radius() for r in recs[1:] if not r.fused.lost]
        LOGGER.info(
            "%s: median 95%% radius %.2f px, lost in %d/%d frames",
            names[k],
            float(np.median(radii)) if radii else float("nan"),
            sum(r.fused.lost for r in recs),
            len(recs),
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                records_to_json(
                    records, targets, names, scale, args.start, args.stride
                ),
                indent=1,
            )
        )
    if args.output:
        write_video(
            render_frames(
                frames, records, targets, names, width=args.output_width
            ),
            args.output,
            fps=min(fps, 15.0),
        )
        LOGGER.info("wrote %s", args.output)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
    )
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "track":
        return cmd_track(argv[1:])
    raise SystemExit(f"Unknown command {argv[0]!r} (use: track)")


if __name__ == "__main__":
    sys.exit(main())
