# SPDX-License-Identifier: BSD-3-Clause
"""Track visualizations: single sequence and COLMAP-vs-COLTAP side by side.

Tracks can come from

* a COLMAP sparse model directory (the 2D observations of every 3D point,
  i.e. the tracks that were actually used for reconstruction), or
* a COLTAP ``tracks.npz`` file (raw TAPNext++ tracks, confidence-filtered).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .frames import list_images, read_image


@dataclasses.dataclass
class FrameTracks:
    """Per-frame 2D track observations: frame -> (track_ids, xy)."""

    ids: list[np.ndarray]
    xy: list[np.ndarray]
    label: str = ""

    @property
    def num_frames(self) -> int:
        return len(self.ids)

    def track_lengths(self) -> dict[int, int]:
        ids, counts = np.unique(np.concatenate(self.ids), return_counts=True)
        return dict(zip(ids.tolist(), counts.tolist(), strict=True))


def tracks_from_model(model_path: str | Path, image_names: list[str]):
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_path))
    frame_of = {name: i for i, name in enumerate(image_names)}
    per_frame: list[list[tuple[int, float, float]]] = [[] for _ in image_names]
    for image in rec.images.values():
        if not image.has_pose or image.name not in frame_of:
            continue
        f = frame_of[image.name]
        for p2d in image.points2D:
            if p2d.has_point3D():
                per_frame[f].append((p2d.point3D_id, *p2d.xy))
    return FrameTracks(
        ids=[np.array([o[0] for o in obs], np.int64) for obs in per_frame],
        xy=[
            np.array([o[1:] for o in obs], np.float32).reshape(-1, 2)
            for obs in per_frame
        ],
    )


def tracks_from_npz(
    path: str | Path, image_names: list[str], min_confidence: float = 0.5
):
    from .tracks import Tracks

    tracks = Tracks.load(path)
    frame_of = {name: i for i, name in enumerate(tracks.image_names)}
    observed = tracks.observed(min_confidence)
    scores_path = Path(path).with_name(Path(path).stem + "_scores.npz")
    if scores_path.exists():
        observed &= np.load(scores_path)["selected"][:, None]
    ids, xy = [], []
    for name in image_names:
        f = frame_of.get(name)
        if f is None:
            ids.append(np.zeros(0, np.int64))
            xy.append(np.zeros((0, 2), np.float32))
            continue
        rows = np.nonzero(observed[:, f])[0]
        ids.append(rows.astype(np.int64))
        xy.append(tracks.xy[rows, f])
    return FrameTracks(ids=ids, xy=xy)


def load_tracks(source: str | Path, image_names: list[str]) -> FrameTracks:
    source = Path(source)
    if source.suffix == ".npz":
        return tracks_from_npz(source, image_names)
    return tracks_from_model(source, image_names)


def _colors(ids: np.ndarray) -> np.ndarray:
    """Stable, saturated color per track id (RGB uint8)."""
    hue = ((ids * 2654435761) % 180).astype(np.uint8)
    hsv = np.stack(
        [hue, np.full_like(hue, 230), np.full_like(hue, 255)], axis=-1
    )[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def draw_tracks(
    image: np.ndarray,
    tracks: FrameTracks,
    frame: int,
    tail: int = 10,
    scale: float = 1.0,
) -> np.ndarray:
    """Draw tracks visible in ``frame`` with a tail over the previous frames."""
    canvas = image.copy()
    ids = tracks.ids[frame]
    if len(ids) == 0:
        return canvas
    colors = _colors(ids)
    history = np.full((len(ids), tail + 1, 2), np.nan, np.float32)
    history[:, -1] = tracks.xy[frame]
    index = {tid: k for k, tid in enumerate(ids.tolist())}
    for back in range(1, tail + 1):
        f = frame - back
        if f < 0:
            break
        for tid, xy in zip(tracks.ids[f].tolist(), tracks.xy[f], strict=True):
            k = index.get(tid)
            if k is not None:
                history[k, tail - back] = xy
    shift = 4
    fixed = (history * scale * (1 << shift)).round()
    for k in range(len(ids)):
        color = tuple(int(c) for c in colors[k])
        valid = np.isfinite(fixed[k, :, 0])
        pts = fixed[k][valid].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(
                canvas, [pts], False, color, 1, cv2.LINE_AA, shift=shift
            )
        cv2.circle(
            canvas,
            tuple(pts[-1]),
            int(2 * (1 << shift)),
            color,
            -1,
            cv2.LINE_AA,
            shift=shift,
        )
    return canvas


def _put_text(img, text, org, size=0.5, color=(255, 255, 255)):
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA
    )


def _panel(image, tracks: FrameTracks, frame, tail, scale, lengths, label):
    panel = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    panel = (panel * 0.75).astype(np.uint8)  # dim so tracks stand out
    panel = draw_tracks(panel, tracks, frame, tail=tail, scale=scale)
    ids = tracks.ids[frame]
    mean_len = np.mean([lengths[i] for i in ids.tolist()]) if len(ids) else 0
    size = 0.45 * max(1.0, panel.shape[1] / 480)
    line = int(20 * size / 0.45)
    header = np.full((2 * line + 8, panel.shape[1], 3), 24, np.uint8)
    _put_text(header, label, (8, line), size, (255, 220, 120))
    _put_text(
        header,
        f"points in frame: {len(ids)}   mean track length: {mean_len:.1f}",
        (8, 2 * line),
        size * 0.9,
    )
    return np.concatenate([header, panel], axis=0)


def render_comparison(
    image_path: str | Path,
    sources: list[tuple[str, FrameTracks]],
    output_paths: list[str | Path],
    tail: int = 10,
    width: int = 480,
    fps: float = 8.0,
    max_frames: int = 0,
    image_names: list[str] | None = None,
) -> None:
    """Write an MP4 and/or GIF with one panel per track source."""
    names = image_names or list_images(image_path)
    num = len(names) if max_frames <= 0 else min(max_frames, len(names))
    lengths = [
        t.track_lengths() if any(len(i) for i in t.ids) else {}
        for _, t in sources
    ]
    frames = []
    for f in range(num):
        image = read_image(Path(image_path) / names[f])
        scale = width / image.shape[1]
        panels = [
            _panel(image, tracks, f, tail, scale, lengths[k], label)
            for k, (label, tracks) in enumerate(sources)
        ]
        sep = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.concatenate([row, sep, p], axis=1)
        frames.append(row)
    for out in output_paths:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".gif":
            _write_gif(out, frames, fps)
        else:
            _write_mp4(out, frames, fps)


def _write_mp4(path: Path, frames, fps: float) -> None:
    even = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
    imageio.mimsave(
        path,
        even,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "26", "-preset", "slow"],
    )


def _write_gif(path: Path, frames, fps: float) -> None:
    """Palette-optimized GIF via ffmpeg when available (much smaller)."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
        return
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "frames.mp4"
        _write_mp4(video, frames, fps)
        palette = (
            "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                palette,
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )


def main_visualize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="coltap visualize_tracks")
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--input_path", required=True, help="Sparse model dir or tracks.npz"
    )
    parser.add_argument("--output_path", required=True, help=".mp4 or .gif")
    parser.add_argument("--label", default="")
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args(argv)
    names = list_images(args.image_path)
    tracks = load_tracks(args.input_path, names)
    render_comparison(
        args.image_path,
        [(args.label or Path(args.input_path).name, tracks)],
        [args.output_path],
        tail=args.tail,
        width=args.width,
        fps=args.fps,
        max_frames=args.max_frames,
        image_names=names,
    )
    return 0


def main_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="coltap compare_tracking")
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--colmap_path",
        required=True,
        help="COLMAP sparse model (or tracks.npz)",
    )
    parser.add_argument(
        "--coltap_path",
        required=True,
        help="COLTAP sparse model (or tracks.npz)",
    )
    parser.add_argument("--colmap_label", default="COLMAP: SIFT + matching")
    parser.add_argument("--coltap_label", default="COLTAP: TAPNext++ tracks")
    parser.add_argument(
        "--output_path",
        nargs="+",
        required=True,
        help="One or more .mp4 / .gif outputs",
    )
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args(argv)
    names = list_images(args.image_path)
    sources = [
        (args.colmap_label, load_tracks(args.colmap_path, names)),
        (args.coltap_label, load_tracks(args.coltap_path, names)),
    ]
    render_comparison(
        args.image_path,
        sources,
        args.output_path,
        tail=args.tail,
        width=args.width,
        fps=args.fps,
        max_frames=args.max_frames,
        image_names=names,
    )
    return 0
