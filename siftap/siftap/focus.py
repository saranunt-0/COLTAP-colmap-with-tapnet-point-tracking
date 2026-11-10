# SPDX-License-Identifier: BSD-3-Clause
"""Focus mode: a stabilized view of each focused target for quality checks.

Every frame, a window around the fused position is resampled (sub-pixel,
cubic) so that the tracked point sits exactly at the centre of the panel.
For areas the tracked rotation and scale are removed as well. If the
tracking is right, the feature stands still in the panel; popping or
jitter shows up as the content moving.

The motion is also measured: aligning consecutive stabilized crops
(template matching, then ECC sub-pixel refinement) gives the residual
frame-to-frame shift of the content ("jitter", image pixels). On a static,
textured surface it is the frame-to-frame tracking error; on textureless
content the alignment is unreliable and no value is reported.
"""

from __future__ import annotations

import cv2
import numpy as np

from .render import (
    DROPPED_COLOR,
    FUSED_COLOR,
    SIFT_COLOR,
    STATUS_COLORS,
    TAP_COLOR,
    TRUTH_COLOR,
    _text,
    dashed_circle,
)

_SHIFT = 4
_ONE = 1 << _SHIFT
MIN_MATCH = 0.6  # template-match score below this: no jitter value


def focus_matrix(rec, target, half: float, size: int, mode: str = "auto"):
    """2x2 linear part A and centre c of the map from the stabilized panel
    to the image: image = A @ u + c, u = panel offset in reference px.

    ``mode``: "translate" (A = I), "similarity" (tracked rotation/scale) or
    "auto" (similarity for areas, translation for points).
    """
    use_sim = mode == "similarity" or (mode == "auto" and target.is_area)
    a, b = (1.0, 0.0)
    if use_sim and rec.transform is not None:
        a, b = rec.transform[:2]
    return np.array([[a, -b], [b, a]]), np.asarray(rec.fused.position, float)


def stabilized_crop(image, A, center, half: float, size: int):
    """Resample a (2 half)^2 reference-px window around ``center`` into a
    size x size panel. Coordinates follow the COLMAP convention (pixel
    centres at +0.5); OpenCV's are at integers, hence the -0.5 shifts."""
    zoom = size / (2.0 * half)
    lin = A / zoom
    # Panel pixel i (OpenCV centre) -> panel offset u = (i + 0.5 - size/2)
    # / zoom -> image (COLMAP) A u + c -> OpenCV image coordinate - 0.5.
    offset = lin @ np.full(2, 0.5 - size / 2.0) + center - 0.5
    M = np.c_[lin, offset].astype(np.float64)
    return cv2.warpAffine(
        image,
        M,
        (size, size),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def to_panel(points, A, center, half: float, size: int):
    """Image (COLMAP) positions -> panel pixel coordinates in OpenCV's
    convention (pixel centres at integers), for drawing."""
    zoom = size / (2.0 * half)
    u = (np.atleast_2d(points) - center) @ np.linalg.inv(A).T
    return u * zoom + size / 2.0 - 0.5


def residual_shift(prev: np.ndarray, cur: np.ndarray) -> float | None:
    """Sub-pixel shift (panel px) of ``cur``'s content relative to
    ``prev``, or None when the content cannot be aligned reliably.

    Coarse: normalized template matching of cur's centre in prev (robust to
    pops of up to a quarter of the panel). Fine: ECC translation refinement.
    (Phase correlation is not used: it weights every frequency equally and
    is dominated by noise on smooth, nearly aligned crops.)
    """
    size = prev.shape[0]
    h = size // 4
    res = cv2.matchTemplate(prev, cur[h:-h, h:-h], cv2.TM_CCOEFF_NORMED)
    _, score, _, (ix, iy) = cv2.minMaxLoc(res)
    if score < MIN_MATCH:
        return None
    # cur(x) ~ prev(x + d) with d = location of cur's centre in prev - h.
    d = np.array([ix - h, iy - h], np.float32)
    warp = np.array([[1, 0, d[0]], [0, 1, d[1]]], np.float32)
    try:
        _, warp = cv2.findTransformECC(
            cur,
            prev,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
            None,
            3,
        )
        d = warp[:, 2]
    except cv2.error:
        pass  # keep the integer estimate
    return float(np.hypot(*d))


def jitter_series(images, recs, target, half=40.0, size=160, mode="auto"):
    """Residual frame-to-frame shift (image px) of the stabilized content,
    NaN unless both frames are tracked (visible: while hidden the panel
    shows the occluder moving past) or when the content cannot be aligned
    (e.g. textureless)."""
    out = np.full(len(recs), np.nan)
    prev = None
    for t, rec in enumerate(recs):
        if rec.status != "tracked" or rec.fused.lost:
            prev = None
            continue
        A, c = focus_matrix(rec, target, half, size, mode)
        gray = cv2.cvtColor(images[t], cv2.COLOR_RGB2GRAY)
        crop = stabilized_crop(gray, A, c, half, size).astype(np.float32)
        if prev is not None:
            shift = residual_shift(prev, crop)
            if shift is not None:
                zoom = size / (2.0 * half)
                scale = float(np.sqrt(abs(np.linalg.det(A))))
                out[t] = shift / zoom * scale
        prev = crop
    return out


def _sparkline(width, values, t, height=26, hi=2.0):
    bar = np.full((height, width, 3), 30, np.uint8)
    n = len(values)
    for ref in (0.5, 1.0):
        y = int(height - 3 - ref / hi * (height - 6))
        cv2.line(bar, (0, y), (width - 1, y), (60, 60, 60), 1)
    pts = []
    for i in range(min(t + 1, n)):
        v = values[i]
        x = int(i / max(n - 1, 1) * (width - 1))
        if not np.isfinite(v):
            if len(pts) > 1:
                cv2.polylines(
                    bar, [np.array(pts, np.int32)], False, (220, 220, 220), 1
                )
            pts = []
            continue
        y = int(height - 3 - min(v, hi) / hi * (height - 6))
        pts.append((x, y))
    if len(pts) > 1:
        cv2.polylines(bar, [np.array(pts, np.int32)], False, (220, 220, 220), 1)
    x = int(min(t, n - 1) / max(n - 1, 1) * (width - 1))
    cv2.line(bar, (x, 0), (x, height - 1), (120, 120, 120), 1)
    return bar


def focus_panel(
    image,
    rec,
    target,
    name,
    jitter,
    t,
    half=40.0,
    size=240,
    mode="auto",
    truth=None,
    last=None,
):
    """One target's stabilized panel: header, crop with overlays, jitter.

    ``last``: the last reported record, used to show the frozen view while
    the target is dropped.
    """
    header = np.full((22, size, 3), 22, np.uint8)
    shown = rec if rec.reported and not rec.fused.lost else last
    status = rec.status if not rec.fused.lost else "dropped"
    color = STATUS_COLORS.get(status, DROPPED_COLOR)
    _text(header, f"{name}: {status}", (6, 16), 0.42, color)
    if shown is None:
        crop = np.full((size, size, 3), 40, np.uint8)
    else:
        A, c = focus_matrix(shown, target, half, size, mode)
        crop = stabilized_crop(image, A, c, half, size)
        if shown is not rec:  # dropped: freeze and grey out
            crop = (0.35 * crop + 0.65 * 60).astype(np.uint8)
        zoom = size / (2.0 * half)
        # Panel centre in OpenCV pixel coordinates (centres at integers).
        mid = int(round((size / 2.0 - 0.5) * _ONE))
        # Fixed reticle: where the tracked feature must stay.
        d = int(0.25 * size * _ONE)
        for p0, p1 in (
            ((mid - d, mid), (mid - d // 3, mid)),
            ((mid + d // 3, mid), (mid + d, mid)),
            ((mid, mid - d), (mid, mid - d // 3)),
            ((mid, mid + d // 3), (mid, mid + d)),
        ):
            cv2.line(crop, p0, p1, (255, 255, 255), 1, cv2.LINE_AA, _SHIFT)
        if shown is rec:
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            r = rec.fused.radius() / scale * zoom
            if r < 2 * size:
                if rec.status == "predicted":
                    dashed_circle(
                        crop, (mid, mid), r, FUSED_COLOR, 1, dash_deg=10
                    )
                else:
                    cv2.circle(
                        crop,
                        (mid, mid),
                        int(r * _ONE),
                        FUSED_COLOR,
                        1,
                        cv2.LINE_AA,
                        _SHIFT,
                    )
            for meas, mcolor in ((rec.sift, SIFT_COLOR), (rec.tap, TAP_COLOR)):
                if meas is None:
                    continue
                q = to_panel(meas.position, A, c, half, size)[0]
                if np.all((q > -size) & (q < 2 * size)):
                    cv2.circle(
                        crop,
                        tuple(int(v * _ONE) for v in q),
                        3 * _ONE,
                        mcolor,
                        -1,
                        cv2.LINE_AA,
                        _SHIFT,
                    )
            if truth is not None:
                q = to_panel(truth, A, c, half, size)[0]
                if np.all((q > 0) & (q < size)):
                    p = tuple(int(v * _ONE) for v in q)
                    cv2.circle(
                        crop, p, 5 * _ONE, (0, 0, 0), 3, cv2.LINE_AA, _SHIFT
                    )
                    cv2.circle(
                        crop, p, 5 * _ONE, TRUTH_COLOR, 1, cv2.LINE_AA, _SHIFT
                    )
            _text(
                crop,
                f"+/-{rec.fused.radius():.1f}px {100 * rec.confidence:.0f}%",
                (6, size - 8),
                0.4,
            )
    j = jitter[t] if t < len(jitter) else np.nan
    spark = _sparkline(size, jitter, t)
    label = "jitter n/a" if not np.isfinite(j) else f"jitter {j:.2f}px"
    _text(spark, label, (4, 17), 0.38, (230, 230, 230))
    return np.concatenate([header, crop, spark], 0)


def render_focus(
    frames,
    images,
    records,
    targets,
    names,
    focus,
    half=40.0,
    mode="auto",
    truth=None,
    truth_visible=None,
):
    """Yield ``frames`` (the annotated full frames) with a column of
    stabilized panels for the targets indexed by ``focus`` on the right."""
    jitters = {
        k: jitter_series(images, records[k], targets[k], half, 160, mode)
        for k in focus
    }
    last = {k: None for k in focus}
    for t, frame in enumerate(frames):
        height = frame.shape[0]
        n = len(focus)
        size = int(min(320, max(120, (height - n * 48) // max(n, 1))))
        panels = []
        for k in focus:
            rec = records[k][t]
            if rec.reported and not rec.fused.lost:
                last[k] = rec
            gt = None
            if truth is not None and (
                truth_visible is None or truth_visible[t, k]
            ):
                gt = truth[t, k]
            panels.append(
                focus_panel(
                    images[t],
                    rec,
                    targets[k],
                    names[k],
                    jitters[k],
                    t,
                    half,
                    size,
                    mode,
                    gt,
                    last[k],
                )
            )
        column = np.concatenate(panels, 0)
        if column.shape[0] < height:
            pad = np.full((height - column.shape[0], size, 3), 22, np.uint8)
            column = np.concatenate([column, pad], 0)
        elif column.shape[0] > height:
            pad = np.full(
                (column.shape[0] - height, frame.shape[1], 3), 22, np.uint8
            )
            frame = np.concatenate([frame, pad], 0)
        sep = np.full((frame.shape[0], 4, 3), 60, np.uint8)
        yield np.concatenate([frame, sep, column], 1)


def jitter_summary(
    images, records, targets, names, focus, half=40.0, mode="auto"
):
    """Median / 95th percentile residual jitter (image px) per target."""
    out = {}
    for k in focus:
        j = jitter_series(images, records[k], targets[k], half, 160, mode)
        ok = j[np.isfinite(j)]
        out[names[k]] = {
            "median_px": float(np.median(ok)) if ok.size else None,
            "p95_px": float(np.quantile(ok, 0.95)) if ok.size else None,
            "measured_frames": int(ok.size),
        }
    return out
