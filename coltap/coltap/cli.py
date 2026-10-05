# SPDX-License-Identifier: BSD-3-Clause
"""``coltap`` command line, mirroring the ``colmap`` command line.

COLTAP-specific commands:

  feature_tracker         Correspondences -> COLMAP database. Replaces
                          ``colmap feature_extractor`` + ``colmap *_matcher``
                          for ordered image sequences / videos.
                          ``--tracker tapnext | sift | hybrid``.
  automatic_reconstructor feature_tracker + COLMAP mapper.
  mapper                  COLMAP incremental mapper (via pycolmap).
  extract_frames          Decode a video into an image folder.
  visualize_tracks        Render tracks of a database/model/tracks file.
  compare_tracking        Side-by-side COLMAP vs COLTAP track video/GIF.

Any other command (``image_undistorter``, ``patch_match_stereo``,
``model_converter``, ...) is forwarded verbatim to the ``colmap`` binary, so
``coltap`` can be used wherever ``colmap`` is used.

Options use COLMAP's ``--Section.option value`` syntax; booleans accept
``1/0/true/false``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pycolmap

from .bridge import BridgeOptions
from .database import PairingOptions, SelectionOptions
from .frames import extract_video_frames
from .pipeline import (
    TRACKERS,
    ModelOptions,
    TrackerOptions,
    automatic_reconstruction,
    run_feature_tracker,
    run_mapper,
)
from .tracking import TrackingOptions

LOGGER = logging.getLogger("coltap")

_TRACKER_HELP = (
    "tapnext: TAPNext++ tracks (COLTAP); sift: stock COLMAP SIFT + "
    "sequential matching; hybrid: both merged into one database."
)


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _arg_type(default):
    if isinstance(default, bool):
        return _parse_bool
    if isinstance(default, (int, float, str)):
        return type(default)
    return str


def _add_dataclass(parser, prefix: str, instance) -> None:
    group = parser.add_argument_group(prefix)
    for field in dataclasses.fields(instance):
        default = getattr(instance, field.name)
        group.add_argument(
            f"--{prefix}.{field.name}",
            type=_arg_type(default),
            default=None,
            metavar=type(default).__name__.upper(),
            help=f"(default: {default})",
        )


def _apply_dataclass(args, prefix: str, instance) -> None:
    for field in dataclasses.fields(instance):
        value = getattr(args, f"{prefix}.{field.name}")
        if value is not None:
            setattr(instance, field.name, value)


_READER_ARGS = {
    "camera_model": str,
    "single_camera": _parse_bool,
    "single_camera_per_folder": _parse_bool,
    "single_camera_per_image": _parse_bool,
    "camera_params": str,
    "mask_path": str,
    "default_focal_length_factor": float,
}
_VERIFICATION_ARGS = {
    "min_num_inliers": int,
    "min_inlier_ratio": float,
    "max_error": float,
    "confidence": float,
    "max_num_trials": int,
}


def _add_tracker_args(parser: argparse.ArgumentParser) -> None:
    reader = parser.add_argument_group("ImageReader")
    for name, kind in _READER_ARGS.items():
        reader.add_argument(f"--ImageReader.{name}", type=kind, default=None)
    _add_dataclass(parser, "TapTracker", ModelOptions())
    _add_dataclass(parser, "TapTracker", TrackingOptions())
    _add_dataclass(parser, "TrackSelection", SelectionOptions())
    _add_dataclass(parser, "SequentialMatching", PairingOptions())
    _add_dataclass(parser, "GapBridge", BridgeOptions())
    verification = parser.add_argument_group("TwoViewGeometry")
    for name, kind in _VERIFICATION_ARGS.items():
        verification.add_argument(
            f"--TwoViewGeometry.{name}", type=kind, default=None
        )
    parser.add_argument("--num_threads", type=int, default=-1)
    video = parser.add_argument_group("VideoReader")
    parser.add_argument(
        "--video_path",
        default="",
        help="Decode this video into --image_path before tracking.",
    )
    video.add_argument("--VideoReader.stride", type=int, default=1)
    video.add_argument("--VideoReader.max_frames", type=int, default=0)
    video.add_argument("--VideoReader.max_size", type=int, default=0)


def _tracker_options(args) -> TrackerOptions:
    options = TrackerOptions()
    _apply_dataclass(args, "TapTracker", options.model)
    _apply_dataclass(args, "TapTracker", options.tracking)
    _apply_dataclass(args, "TrackSelection", options.selection)
    _apply_dataclass(args, "SequentialMatching", options.pairing)
    _apply_dataclass(args, "GapBridge", options.bridge)
    reader = {
        name: getattr(args, f"ImageReader.{name}") for name in _READER_ARGS
    }
    # Videos come from one camera: default to a single shared camera.
    options.camera_mode = pycolmap.CameraMode.SINGLE
    if reader["single_camera"] is False:
        options.camera_mode = pycolmap.CameraMode.AUTO
    if reader["single_camera_per_folder"]:
        options.camera_mode = pycolmap.CameraMode.PER_FOLDER
    if reader["single_camera_per_image"]:
        options.camera_mode = pycolmap.CameraMode.PER_IMAGE
    for name in (
        "camera_model",
        "camera_params",
        "mask_path",
        "default_focal_length_factor",
    ):
        if reader[name] is not None:
            setattr(options.reader, name, reader[name])
    for name in _VERIFICATION_ARGS:
        value = getattr(args, f"TwoViewGeometry.{name}")
        if value is None:
            continue
        if name in ("max_error", "confidence", "max_num_trials"):
            setattr(options.verification.ransac, name, value)
        else:
            setattr(options.verification, name, value)
    options.num_threads = args.num_threads
    return options


def _maybe_extract_video(args) -> None:
    if not args.video_path:
        return
    extract_video_frames(
        args.video_path,
        args.image_path,
        stride=getattr(args, "VideoReader.stride"),
        max_frames=getattr(args, "VideoReader.max_frames"),
        max_size=getattr(args, "VideoReader.max_size"),
    )


def _mapper_options(extra: list[str]) -> pycolmap.IncrementalPipelineOptions:
    """Parse ``--Mapper.<name> <value>`` pairs into pipeline options.

    Names are looked up on IncrementalPipelineOptions, then on its ``mapper``
    and ``triangulation`` sub-options, like COLMAP's option manager does.
    """
    options = pycolmap.IncrementalPipelineOptions()
    if len(extra) % 2:
        raise SystemExit(f"Unpaired mapper arguments: {extra}")
    for key, value in zip(extra[::2], extra[1::2], strict=True):
        if not key.startswith("--Mapper."):
            raise SystemExit(f"Unknown argument: {key}")
        name = key[len("--Mapper.") :]
        for target in (options, options.mapper, options.triangulation):
            if hasattr(target, name):
                current = getattr(target, name)
                if isinstance(current, bool):
                    value = _parse_bool(value)
                elif isinstance(current, (int, float)):
                    value = type(current)(value)
                setattr(target, name, value)
                break
        else:
            raise SystemExit(f"Unknown mapper option: {name}")
    return options


def cmd_feature_tracker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="coltap feature_tracker",
        description="Track points (TAPNext++, SIFT or both) and write a "
        "COLMAP database (replaces feature_extractor + sequential_matcher).",
    )
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--tracks_path",
        default="",
        help="Optional .npz to store raw tracks (+ _scores.npz).",
    )
    parser.add_argument(
        "--tracker", default="tapnext", choices=TRACKERS, help=_TRACKER_HELP
    )
    _add_tracker_args(parser)
    args = parser.parse_args(argv)
    _maybe_extract_video(args)
    run_feature_tracker(
        args.image_path,
        args.database_path,
        _tracker_options(args),
        tracks_path=args.tracks_path or None,
        tracker=args.tracker,
    )
    return 0


def cmd_mapper(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="coltap mapper", description="COLMAP incremental mapper."
    )
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--input_path", default="")
    args, extra = parser.parse_known_args(argv)
    reconstructions = run_mapper(
        args.database_path,
        args.image_path,
        args.output_path,
        _mapper_options(extra),
        input_path=args.input_path,
    )
    for idx, rec in reconstructions.items():
        print(f"Model {idx}:\n{rec.summary()}")
    return 0 if reconstructions else 1


def cmd_automatic_reconstructor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="coltap automatic_reconstructor",
        description="Tracking + COLMAP mapper (+ COLMAP dense if requested).",
    )
    parser.add_argument("--workspace_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--tracker",
        default="tapnext",
        choices=TRACKERS,
        help=_TRACKER_HELP,
    )
    parser.add_argument(
        "--dense",
        type=_parse_bool,
        default=False,
        help="Run COLMAP's dense pipeline via the colmap binary (needs CUDA).",
    )
    _add_tracker_args(parser)
    args, extra = parser.parse_known_args(argv)
    _maybe_extract_video(args)
    reconstructions = automatic_reconstruction(
        args.workspace_path,
        args.image_path,
        tracker=args.tracker,
        tracker_options=_tracker_options(args),
        mapper_options=_mapper_options(extra),
    )
    if not reconstructions:
        LOGGER.error("Mapping failed: no model was reconstructed.")
        return 1
    if args.dense:
        return _run_dense(Path(args.workspace_path), args.image_path)
    return 0


def _run_dense(workspace: Path, image_path: str) -> int:
    """COLMAP's dense pipeline on the largest model (unchanged COLMAP)."""
    best = max(
        (p for p in (workspace / "sparse").iterdir() if p.is_dir()),
        key=lambda p: pycolmap.Reconstruction(p).num_reg_images(),
    )
    dense = workspace / "dense"
    steps = [
        [
            "image_undistorter",
            "--image_path",
            image_path,
            "--input_path",
            str(best),
            "--output_path",
            str(dense),
        ],
        ["patch_match_stereo", "--workspace_path", str(dense)],
        [
            "stereo_fusion",
            "--workspace_path",
            str(dense),
            "--output_path",
            str(dense / "fused.ply"),
        ],
    ]
    for step in steps:
        code = _run_colmap(step)
        if code:
            return code
    return 0


def cmd_extract_frames(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="coltap extract_frames")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_size", type=int, default=0)
    args = parser.parse_args(argv)
    names = extract_video_frames(
        args.video_path,
        args.image_path,
        stride=args.stride,
        max_frames=args.max_frames,
        start=args.start,
        max_size=args.max_size,
    )
    print(f"Wrote {len(names)} frames to {args.image_path}")
    return 0


def cmd_visualize_tracks(argv: list[str]) -> int:
    from . import visualize

    return visualize.main_visualize(argv)


def cmd_compare_tracking(argv: list[str]) -> int:
    from . import visualize

    return visualize.main_compare(argv)


COMMANDS = {
    "feature_tracker": cmd_feature_tracker,
    "mapper": cmd_mapper,
    "automatic_reconstructor": cmd_automatic_reconstructor,
    "extract_frames": cmd_extract_frames,
    "visualize_tracks": cmd_visualize_tracks,
    "compare_tracking": cmd_compare_tracking,
}


def _colmap_binary() -> str | None:
    return os.environ.get("COLTAP_COLMAP_BINARY") or shutil.which("colmap")


def _run_colmap(argv: list[str]) -> int:
    binary = _colmap_binary()
    if binary is None:
        LOGGER.error(
            "'%s' is a COLMAP command; install COLMAP (or set "
            "COLTAP_COLMAP_BINARY) to forward it. The pycolmap wheel does "
            "not ship the colmap executable.",
            argv[0],
        )
        return 2
    return subprocess.call([binary, *argv])


def _usage() -> str:
    lines = [__doc__ or "", "Commands:"]
    lines += [f"  {name}" for name in COMMANDS]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
    )
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command in COMMANDS:
        return COMMANDS[command](rest)
    return _run_colmap([command, *rest])


if __name__ == "__main__":
    sys.exit(main())
