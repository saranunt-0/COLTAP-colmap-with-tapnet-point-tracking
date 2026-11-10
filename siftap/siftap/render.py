# SPDX-License-Identifier: BSD-3-Clause
"""Drawing SIFTAP results: GPS-style uncertainty circles and timelines.

Every circle is a 95 % confidence radius: the true position should lie inside
it 95 % of the time (the covariances are calibrated for this). Small circle =
the tracker is sure; large circle = it is confused.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .geometry import CHI2_2DOF_95
from .tracker import FrameRecord

SIFT_COLOR = (70, 170, 255)
TAP_COLOR = (255, 150, 40)
FUSED_COLOR = (70, 230, 100)
REJECT_COLOR = (255, 60, 60)
TRUTH_COLOR = (255, 255, 255)
DROPPED_COLOR = (150, 150, 150)
STATUS_COLORS = {
    "tracked": FUSED_COLOR,
    "predicted": TAP_COLOR,
    "dropped": DROPPED_COLOR,
}
SQRT_CHI2 = float(np.sqrt(CHI2_2DOF_95))
_SHIFT = 4
_ONE = 1 << _SHIFT


def _pt(xy, scale=1.0):
    # COLMAP convention (+0.5) -> OpenCV pixel centers; fixed point.
    return tuple(int(round((v - 0.5) * scale * _ONE)) for v in xy)


def radius95(cov: np.ndarray) -> float:
    """95 % radius of a Gaussian (estimates also have a mixture-aware
    ``.radius()``, which the drawing uses)."""
    return SQRT_CHI2 * float(np.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0)))


def dashed_circle(img, center, radius, color, thickness=1, dash_deg=14):
    r = int(round(radius * _ONE))
    if r <= 0:
        return
    for start in range(0, 360, 2 * dash_deg):
        cv2.ellipse(
            img,
            center,
            (r, r),
            0,
            start,
            start + dash_deg,
            color,
            thickness,
            cv2.LINE_AA,
            _SHIFT,
        )


def _text(img, text, org, scale=0.45, color=(255, 255, 255), thick=1):
    # Dark outline with the *same* thickness (in Hershey fonts the glyph
    # advance grows with thickness, so a thicker shadow drifts off).
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
        cv2.putText(
            img,
            text,
            (org[0] + dx, org[1] + dy),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def draw_target(
    canvas, overlay, rec: FrameRecord, history, scale, name, corners=None
):
    """Draw one target's sources, fusion and trail onto ``canvas``."""
    max_r = 2.0 * max(canvas.shape[:2])
    if rec.status == "dropped":
        # Nothing is reported for this frame: a faint note where the
        # target was last reported.
        where = history[-1] if history else rec.fused.position
        org = (
            int(np.clip(where[0] * scale, 6, canvas.shape[1] - 120)),
            int(np.clip(where[1] * scale, 18, canvas.shape[0] - 24)),
        )
        _text(canvas, f"{name}: hidden", org, 0.4, DROPPED_COLOR)
        return
    for meas, color in ((rec.sift, SIFT_COLOR), (rec.tap, TAP_COLOR)):
        if meas is None:
            continue
        ok = rec.fused.accepted.get(meas.source, True)
        c = color if ok else REJECT_COLOR
        p = _pt(meas.position, scale)
        dashed_circle(canvas, p, min(meas.radius() * scale, max_r), c, 1)
        cv2.circle(canvas, p, 2 * _ONE, c, -1, cv2.LINE_AA, _SHIFT)
        if not ok:
            d = 4 * _ONE
            cv2.line(
                canvas,
                (p[0] - d, p[1] - d),
                (p[0] + d, p[1] + d),
                c,
                1,
                cv2.LINE_AA,
                _SHIFT,
            )
            cv2.line(
                canvas,
                (p[0] - d, p[1] + d),
                (p[0] + d, p[1] - d),
                c,
                1,
                cv2.LINE_AA,
                _SHIFT,
            )
    fused = rec.fused
    if len(history) > 1:
        pts = np.array([_pt(h, scale) for h in history], np.int32)
        cv2.polylines(canvas, [pts], False, FUSED_COLOR, 1, cv2.LINE_AA, _SHIFT)
    if fused.lost:
        _text(
            canvas,
            f"{name} lost",
            tuple(int(v * scale) for v in fused.position),
            0.45,
            REJECT_COLOR,
        )
        return
    height, width = canvas.shape[:2]
    x, y = fused.position * scale
    r_px = fused.radius() * scale
    if not (0 <= x < width and 0 <= y < height) or r_px > 0.5 * max(
        width, height
    ):
        # Out of view or very uncertain: a thin dashed circle and a label
        # clamped to the frame instead of a huge filled disc.
        p = _pt(fused.position, scale)
        dashed_circle(canvas, p, min(r_px, max_r), FUSED_COLOR, 1, dash_deg=6)
        where = (
            "out of view"
            if not (0 <= x < width and 0 <= y < height)
            else "uncertain"
        )
        org = (
            int(np.clip(x, 6, width - 190)),
            int(np.clip(y, 18, height - 24)),
        )
        _text(
            canvas,
            f"{name}: {where} +/-{fused.radius():.0f}px",
            org,
            0.42,
            FUSED_COLOR,
        )
        return
    p = _pt(fused.position, scale)
    r = min(r_px, max_r)
    if rec.status == "predicted":
        # Hidden target, position from the trajectory: dashed, no fill.
        dashed_circle(canvas, p, r, FUSED_COLOR, 2, dash_deg=10)
    else:
        cv2.circle(
            overlay,
            p,
            int(round(r * _ONE)),
            FUSED_COLOR,
            -1,
            cv2.LINE_AA,
            _SHIFT,
        )
        cv2.circle(
            canvas,
            p,
            int(round(r * _ONE)),
            FUSED_COLOR,
            2,
            cv2.LINE_AA,
            _SHIFT,
        )
    d = 5 * _ONE
    white = (255, 255, 255)
    cv2.line(
        canvas,
        (p[0] - d, p[1]),
        (p[0] + d, p[1]),
        white,
        1,
        cv2.LINE_AA,
        _SHIFT,
    )
    cv2.line(
        canvas,
        (p[0], p[1] - d),
        (p[0], p[1] + d),
        white,
        1,
        cv2.LINE_AA,
        _SHIFT,
    )
    if corners is not None:
        poly = np.array([_pt(c, scale) for c in corners], np.int32)
        cv2.polylines(canvas, [poly], True, FUSED_COLOR, 2, cv2.LINE_AA, _SHIFT)
    state = " occluded" if rec.status == "predicted" else ""
    label = (
        f"{name}{state} +/-{fused.radius():.1f}px {100 * rec.confidence:.0f}%"
    )
    lx = int(x + r + 6)
    ly = int(y - 6)
    _text(canvas, label, (min(lx, width - 120), max(ly, 14)), 0.42)


def _timeline(
    width, records, t, names, row_h=34, lo=0.3, hi=150.0, errors=None
):
    """Per target: log-scale 95 % radius over time for the three estimates
    (and, with ``errors`` [K, T], the fused estimate's true error, dotted:
    it should stay below the green radius line)."""
    rows = []
    n = len(records[0])
    label_w = 70
    for k, recs in enumerate(records):
        row = np.full((row_h, width, 3), 22, np.uint8)
        _text(row, names[k], (6, row_h // 2 + 5), 0.42, (230, 230, 230))
        plot_w = width - label_w - 8

        def ypos(r):
            r = np.clip(r, lo, hi)
            return (row_h - 4) - (np.log(r / lo) / np.log(hi / lo)) * (
                row_h - 8
            )

        for ref in (1.0, 10.0, 100.0):
            yy = int(ypos(ref))
            cv2.line(row, (label_w, yy), (width - 8, yy), (55, 55, 55), 1)
        for getter, color in (
            (lambda r: r.sift, SIFT_COLOR),
            (lambda r: r.tap, TAP_COLOR),
            (lambda r: r.fused if r.reported else None, FUSED_COLOR),
        ):
            pts = []
            for i, rec in enumerate(recs[: t + 1]):
                m = getter(rec)
                if m is None or (hasattr(m, "lost") and m.lost):
                    if len(pts) > 1:
                        cv2.polylines(
                            row,
                            [np.array(pts, np.int32)],
                            False,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    pts = []
                    continue
                x = label_w + int(i / max(n - 1, 1) * plot_w)
                pts.append((x, int(ypos(m.radius()))))
            if len(pts) > 1:
                cv2.polylines(
                    row, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA
                )
        if errors is not None:
            for i in range(1, t + 1):
                e = errors[k][i]
                if np.isfinite(e):
                    x = label_w + int(i / max(n - 1, 1) * plot_w)
                    cv2.circle(row, (x, int(ypos(e))), 1, TRUTH_COLOR, -1)
        # Status strip: green tracked, orange predicted (hidden), grey dropped.
        for i, rec in enumerate(recs[: t + 1]):
            x0 = label_w + int(i / max(n - 1, 1) * plot_w)
            x1 = label_w + int((i + 1) / max(n - 1, 1) * plot_w)
            color = STATUS_COLORS.get(rec.status, DROPPED_COLOR)
            cv2.rectangle(
                row, (x0, row_h - 3), (max(x1 - 1, x0), row_h - 1), color, -1
            )
        x = label_w + int(t / max(n - 1, 1) * plot_w)
        cv2.line(row, (x, 2), (x, row_h - 2), (200, 200, 200), 1)
        rows.append(row)
    return np.concatenate(rows, 0)


def _legend(width, truth=False, statuses=()):
    bar = np.full((26, width, 3), 22, np.uint8)
    x = 8
    items = [
        ("SIFT", SIFT_COLOR),
        ("TAPNext++", TAP_COLOR),
        ("fused", FUSED_COLOR),
        ("rejected", REJECT_COLOR),
    ]
    if "predicted" in statuses:
        items.append(("predicted (hidden)", FUSED_COLOR))
    if "dropped" in statuses:
        items.append(("dropped", DROPPED_COLOR))
    if truth:
        items.append(("truth", TRUTH_COLOR))
    for text, color in items:
        cv2.circle(bar, (x + 5, 13), 5, color, -1, cv2.LINE_AA)
        _text(bar, text, (x + 14, 18), 0.42, (235, 235, 235))
        x += 30 + 8 * len(text)
    if x < width - 150:
        _text(
            bar, "circles: 95% radius", (width - 150, 18), 0.4, (180, 180, 180)
        )
    return bar


def _history_and_corners(target, recs, t, trail, max_radius=np.inf):
    rec = recs[t]
    # Only confident positions form the trail (not coasting predictions).
    hist = [
        r.fused.position
        for r in recs[max(0, t - trail) : t + 1]
        if r.reported and not r.fused.lost and r.fused.radius() <= max_radius
    ]
    corners = None
    if target.is_area and rec.transform is not None:
        a, b, tx, ty = rec.transform
        rot = np.array([[a, -b], [b, a]])
        corners = target.corner_offsets() @ rot.T + [tx, ty]
    return rec, hist, corners


def render_frames(
    images,
    records: list[list[FrameRecord]],
    targets,
    names=None,
    width: int = 0,
    trail: int = 40,
    timeline: bool = True,
    truth: np.ndarray | None = None,
    truth_visible: np.ndarray | None = None,
):
    """Yield annotated RGB frames (legend on top, timelines below).

    ``truth`` [T, K, 2] (and ``truth_visible`` [T, K]) draws the true
    position (white ring) and the fused error in the timelines.
    """
    names = names or [t.name or f"T{k + 1}" for k, t in enumerate(targets)]
    statuses = {r.status for recs in records for r in recs}
    errors = None
    if truth is not None:
        vis = (
            np.ones(truth.shape[:2], bool)
            if truth_visible is None
            else truth_visible
        )
        errors = np.full((len(targets), len(images)), np.nan)
        for k, recs in enumerate(records):
            for i, r in enumerate(recs):
                if vis[i, k] and r.reported and not r.fused.lost:
                    errors[k, i] = np.linalg.norm(
                        r.fused.position - truth[i, k]
                    )
    for t, image in enumerate(images):
        scale = 1.0 if width <= 0 else width / image.shape[1]
        trail_radius = 0.05 * max(image.shape[:2])
        canvas = (
            image
            if scale == 1.0
            else cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        )
        canvas = (canvas * 0.85).astype(np.uint8)
        overlay = canvas.copy()
        for k, target in enumerate(targets):
            rec, hist, corners = _history_and_corners(
                target, records[k], t, trail, trail_radius
            )
            draw_target(canvas, overlay, rec, hist, scale, names[k], corners)
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        # Re-draw outlines on top of the blend for crisp edges.
        for k, target in enumerate(targets):
            rec, hist, corners = _history_and_corners(
                target, records[k], t, trail, trail_radius
            )
            draw_target(
                canvas, canvas.copy(), rec, hist, scale, names[k], corners
            )
        if truth is not None:
            for k in range(len(targets)):
                if truth_visible is None or truth_visible[t, k]:
                    p = _pt(truth[t, k], scale)
                    cv2.circle(
                        canvas, p, 3 * _ONE, (0, 0, 0), 3, cv2.LINE_AA, _SHIFT
                    )
                    cv2.circle(
                        canvas, p, 3 * _ONE, TRUTH_COLOR, 1, cv2.LINE_AA, _SHIFT
                    )
        _text(canvas, f"frame {t}", (8, canvas.shape[0] - 8), 0.42)
        parts = [_legend(canvas.shape[1], truth is not None, statuses), canvas]
        if timeline:
            parts.append(
                _timeline(canvas.shape[1], records, t, names, errors=errors)
            )
        yield np.concatenate(parts, 0)


def write_video(frames, path: str | Path, fps: float = 10.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    if path.suffix.lower() == ".gif":
        _write_gif(frames, path, fps)
        return
    even = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
    imageio.mimsave(
        path,
        even,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "24", "-preset", "slow"],
    )


def _write_gif(frames, path: Path, fps: float):
    if shutil.which("ffmpeg") is None:
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
        return
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "f.mp4"
        write_video(frames, video, fps)
        palette = (
            "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
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
