# SPDX-License-Identifier: BSD-3-Clause
"""Ground-truth benchmark for SIFTAP on the synthetic street scene.

    python benchmark/evaluate.py --work_path WORK --poster_path DIR \
        [--stage render|tap|calibrate|compare|occlusion|demo|tables|all] \
        [--output_path DIR]

* render: four sequences with exact GT. ``calib`` (seed 0) and ``calib2``
  (seed 3, 1.5x speed) are used only to calibrate the uncertainties;
  ``test`` (seed 1) and ``fast`` (seed 2, twice the speed, motion blur) are
  held out.
* tap: one TAPNext++ pass per sequence for all targets (cached).
* calibrate: is each source's covariance honest? NEES / coverage of the
  95 % radius, before and after fitting a scale + floor on ``calib``.
* compare: every fusion strategy on every sequence, per surface category.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

from coltap.frames import list_images, read_image
from siftap import city
from siftap.fusion import FUSIONS
from siftap.geometry import mixture_radius
from siftap.sift import SiftFrontEnd, SiftParams, gross_probability
from siftap.tap import TapCalibration, TapFrontEnd
from siftap.targets import Target
from siftap.tracker import SiftapTracker, TrackerOptions

LOGGER = logging.getLogger("benchmark")
CATEGORIES = city.CATEGORIES
GROSS_PX = 3.0  # a SIFT measurement further than this from GT is a mismatch
SEQUENCES = {
    "calib": dict(seed=0, speed=0.12, frames=60, subframes=1),
    "calib2": dict(seed=3, speed=0.18, frames=60, subframes=2),
    "test": dict(seed=1, speed=0.12, frames=60, subframes=1),
    "fast": dict(seed=2, speed=0.24, frames=48, subframes=4),
    # Near pillars sweep across the view: targets disappear and reappear.
    # Used only for the occlusion-policy evaluation.
    "occl": dict(seed=4, speed=0.12, frames=60, subframes=1, pillars=6),
}
CALIBRATION = ("calib", "calib2")  # everything else is held out


def load_posters(path, n=6):
    posters = []
    for name in list_images(path)[:n]:
        im = read_image(Path(path) / name)
        s = 512 / max(im.shape[:2])
        posters.append(
            cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        )
    return posters


def render(work: Path, posters):
    for name, cfg in SEQUENCES.items():
        out = work / f"{name}.pkl"
        if out.exists():
            continue
        c = city.make_city(
            posters, seed=cfg["seed"], pillars=cfg.get("pillars", 0)
        )
        poses = city.street_trajectory(
            cfg["frames"],
            speed=cfg["speed"],
            seed=cfg["seed"],
            subframes=cfg["subframes"],
        )
        frames, frame_poses = city.render_sequence(
            c, poses, subframes=cfg["subframes"], seed=cfg["seed"]
        )
        targets = city.sample_targets(
            c, frames[0], frame_poses[0], seed=cfg["seed"]
        )
        cats, pts = [], []
        for cat, p in targets.items():
            cats += [cat] * len(p)
            pts.append(p)
        pts = np.concatenate(pts)
        gt_xy, gt_vis = city.ground_truth(c, frame_poses, pts)
        with open(out, "wb") as f:
            pickle.dump(
                dict(
                    frames=frames,
                    points=pts,
                    categories=cats,
                    gt_xy=gt_xy,
                    gt_vis=gt_vis,
                ),
                f,
            )
        LOGGER.info(
            "rendered %s: %d frames, %d targets", name, len(frames), len(pts)
        )


def load(work: Path, name: str):
    with open(work / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def make_targets(seq, radius=24.0):
    return [Target.point(*p, radius=radius) for p in seq["points"]]


def run_tap(work: Path, precision: str):
    from coltap.model import TapNextPP

    model = None
    for name in SEQUENCES:
        out = work / f"{name}_tap.pkl"
        if out.exists():
            continue
        model = model or TapNextPP(precision=precision)
        seq = load(work, name)
        front = TapFrontEnd(model, make_targets(seq))
        raws = [front.step(img) for img in seq["frames"]]
        with open(out, "wb") as f:
            pickle.dump(raws, f)
        LOGGER.info("tap %s done", name)


# ---------------------------------------------------------------- metrics
def fit_logistic(features, labels, l2=1e-2, iterations=50):
    """Newton / IRLS logistic regression; returns [bias, *weights]."""
    x = np.c_[np.ones(len(features)), features]
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - labels) + l2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def sift_samples(seq, params: SiftParams):
    """SIFT measurement errors with the search window centred on GT."""
    targets = make_targets(seq)
    fronts = [SiftFrontEnd(t, params) for t in targets]
    gray0 = cv2.cvtColor(seq["frames"][0], cv2.COLOR_RGB2GRAY)
    for f, t in zip(fronts, targets, strict=True):
        f.add_reference(gray0, t.center, 0)
    errs, covs, cats, tried = [], [], [], {}
    for t in range(1, len(seq["frames"])):
        gray = cv2.cvtColor(seq["frames"][t], cv2.COLOR_RGB2GRAY)
        for k, f in enumerate(fronts):
            if not seq["gt_vis"][t, k]:
                continue
            cat = seq["categories"][k]
            tried[cat] = tried.get(cat, 0) + 1
            m = f.measure(gray, seq["gt_xy"][t, k], 2.0)
            if m is None:
                continue
            errs.append(m.position - seq["gt_xy"][t, k])
            covs.append(m.cov)
            cats.append(
                (
                    cat,
                    m.info["inliers"],
                    m.info["inlier_ratio"],
                    m.radius(),
                    m.info["texture_sigma"],
                )
            )
    return errs, covs, cats, tried


K95 = float(np.sqrt(-2 * np.log(0.05)))  # 95 % radius / sigma (2D)


def fit_radius_model(
    dist, sigma, cum=None, time=None, drift=False, texture=None
):
    """Fit radius = K95 * sqrt((a sigma)^2 + f^2 + kappa^2 cum).

    The drawn 95 % circle must cover 95 % of the errors overall *and* in
    every cell of sigma quintiles x time thirds (a reliability-diagram
    criterion: the radius has to be honest whether it is small or large,
    early or late); among equally calibrated models the sharpest wins.
    ``cum`` is the running sum of texture_sigma^2 since the query (drift
    variance); with ``texture``, cells of texture quintiles x time thirds
    are checked too.
    """
    dist, sigma = np.asarray(dist), np.asarray(sigma)
    quint = [0.2, 0.4, 0.6, 0.8]
    masks = []
    for var in (sigma, texture):
        if var is None:
            continue
        cell = np.digitize(var, np.quantile(var, quint)) * 3
        if time is not None:
            cell = cell + np.digitize(time, [1 / 3, 2 / 3])
        masks += [cell == c for c in np.unique(cell) if (cell == c).sum() >= 30]
    kappas = np.r_[0.0, np.geomspace(0.05, 50.0, 60)] if drift else [0.0]
    best = None
    for kappa in kappas:
        for f in np.r_[0.0, np.linspace(0.1, 1.0, 10)]:
            for a in np.geomspace(0.1, 5.0, 80):
                var = (a * sigma) ** 2 + f * f
                if drift:
                    var = var + kappa * kappa * cum
                r = K95 * np.sqrt(var)
                cov = dist <= r
                worst = max(abs(cov[m].mean() - 0.95) for m in masks)
                cost = worst + abs(cov.mean() - 0.95) + 0.01 * np.median(r)
                if best is None or cost < best[0]:
                    best = (cost, float(a), float(f), float(kappa))
    return best[1:]


def coverage_report(dist, radius, cats):
    dist, radius, cats = map(np.asarray, (dist, radius, cats))
    out = {
        "all": {
            "n": len(dist),
            "circle95": float(np.mean(dist <= radius)),
            "median_error": float(np.median(dist)),
            "median_r95": float(np.median(radius)),
        }
    }
    for c in CATEGORIES:
        m = cats == c
        if m.any():
            out[c] = {
                "n": int(m.sum()),
                "circle95": float(np.mean(dist[m] <= radius[m])),
                "median_error": float(np.median(dist[m])),
                "median_r95": float(np.median(radius[m])),
            }
    return out


def calibrate_sift(seqs):
    """Fit the SIFT covariance (scale, floor, texture term) and the mismatch
    model."""
    errs, covs, meta = [], [], []
    tried = {}
    raw_params = SiftParams(
        sigma_scale=1.0, sigma_floor=0.0, texture_scale=0.0, gross_bias=-50.0
    )
    for seq in seqs:
        e, c, m, t = sift_samples(seq, raw_params)
        errs += e
        covs += c
        meta += m
        for k, v in t.items():
            tried[k] = tried.get(k, 0) + v
    cats = [m[0] for m in meta]
    dist = np.linalg.norm(np.array(errs), axis=1)
    sig = np.array([np.sqrt(np.linalg.eigvalsh(c)[-1]) for c in covs])
    tex = np.array([m[4] for m in meta])
    report = {
        "sift_found": {
            c: sum(1 for x in cats if x == c) / max(tried.get(c, 1), 1)
            for c in tried
        },
        "sift_raw": coverage_report(dist, K95 * sig, cats),
    }
    gross = dist > GROSS_PX
    feats = np.array([[np.log(m[1]), m[2]] for m in meta])
    bias, w_inl, w_ratio = fit_logistic(feats, gross.astype(float))
    gross_sigma = (
        float(np.sqrt(np.mean(dist[gross] ** 2) / 2)) if gross.any() else 10.0
    )
    # Scale/floor/texture on correct matches only; mismatches are the
    # mixture's job.
    ok = ~gross
    s, f, kappa = fit_radius_model(
        dist[ok], sig[ok], tex[ok] ** 2, drift=True, texture=tex[ok]
    )
    report["sift_fit"] = {
        "scale": s,
        "floor": f,
        "texture_scale": kappa,
        "gross_rate": float(gross.mean()),
        "gross_bias": float(bias),
        "gross_w_inliers": float(w_inl),
        "gross_w_ratio": float(w_ratio),
        "gross_sigma": gross_sigma,
    }
    cal = sift_params_from(report["sift_fit"])
    p_out = np.array([gross_probability(m[1], m[2], cal) for m in meta])
    radius = np.array(
        [
            mixture_radius(
                np.sqrt(s * s * x * x + f * f + (kappa * tx) ** 2),
                p,
                gross_sigma,
            )
            for x, tx, p in zip(sig, tex, p_out, strict=True)
        ]
    )
    report["sift_calibrated"] = coverage_report(dist, radius, cats)
    report["sift_mismatch_model"] = {
        f"p<{hi}": {
            "n": int(((p_out >= lo) & (p_out < hi)).sum()),
            "gross_rate": float(np.mean(gross[(p_out >= lo) & (p_out < hi)]))
            if ((p_out >= lo) & (p_out < hi)).any()
            else None,
        }
        for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01))
    }
    return report


def tap_series(seq, raws, calib: TapCalibration):
    """Per (frame, target) with a TAP measurement and visible GT: error,
    point sigma, texture sigma, running sum of texture sigma^2 (the drift
    variance up to kappa^2), normalized time, category."""
    front = TapFrontEnd(None, make_targets(seq), calib)
    n_t, n_k = len(raws), len(seq["categories"])
    dist = np.full((n_t, n_k), np.nan)
    sigma = np.zeros((n_t, n_k))
    tex = np.zeros((n_t, n_k))
    for t in range(1, n_t):
        gray = cv2.cvtColor(seq["frames"][t], cv2.COLOR_RGB2GRAY)
        for k, m in enumerate(front.measurements(raws[t], gray)):
            if m is None or m.occluded:  # visible frames only
                continue
            sigma[t, k] = m.sigma
            tex[t, k] = m.info["texture_sigma"]
            if seq["gt_vis"][t, k]:
                dist[t, k] = np.linalg.norm(m.position - seq["gt_xy"][t, k])
    cum = np.cumsum(tex**2, axis=0)
    time = np.repeat((np.arange(n_t) / n_t)[:, None], n_k, 1)
    cats = np.repeat(np.array(seq["categories"])[None], n_t, 0)
    ok = ~np.isnan(dist)
    return dist[ok], sigma[ok], tex[ok], cum[ok], time[ok], cats[ok]


def occluded_series(seq, raws, calib: TapCalibration):
    """Frames where TAPNext++ reports the target hidden and the truth is in
    the image: error of its estimate, global heatmap sigma, normalized
    time since the occlusion started (capped at 30 frames)."""
    front = TapFrontEnd(None, make_targets(seq), calib)
    gt = seq["gt_xy"]
    height, width = seq["frames"][0].shape[:2]
    run = np.zeros(len(seq["categories"]), int)
    dist, sigma, since = [], [], []
    for t in range(1, len(raws)):
        for k, m in enumerate(front.measurements(raws[t])):
            if not m.occluded:
                run[k] = 0
                continue
            run[k] += 1
            x, y = gt[t, k]
            if not (0 <= x < width and 0 <= y < height):
                continue
            dist.append(np.linalg.norm(m.position - gt[t, k]))
            sigma.append(
                float(np.max(raws[t].sigma_global[front.slices[k]][0]))
            )
            since.append(min(run[k] / 30, 0.99))
    return np.array(dist), np.array(sigma), np.array(since)


def calibrate_occluded(seqs, raws_list, held_out=()):
    """std = scale * global heatmap sigma (+ floor) for hidden targets."""
    raw = TapCalibration(occluded_scale=1.0, occluded_floor=0.0)
    d, s, tm = _concat(
        [
            occluded_series(q, r, raw)
            for q, r in zip(seqs, raws_list, strict=True)
        ]
    )
    a, f, _ = fit_radius_model(d, s, time=tm)
    out = {"scale": a, "floor": f}
    for name, data in [
        ("calib", (d, s)),
        *[(n, occluded_series(q, r, raw)[:2]) for n, (q, r) in held_out],
    ]:
        dd, ss = data
        r95 = K95 * np.sqrt((a * ss) ** 2 + f * f)
        out[name] = {
            "n": len(dd),
            "median_error": float(np.median(dd)),
            "p90_error": float(np.quantile(dd, 0.9)),
            "circle95": float(np.mean(dd <= r95)),
            "median_r95": float(np.median(r95)),
        }
    return out


def _concat(parts):
    return [np.concatenate(x) for x in zip(*parts, strict=True)]


def calibrate_tap(seqs, raws_list, held_out=()):
    """Heatmap sigma calibration (white part) + drift, per sigma variant.

    ``drift_scale`` (kappa) makes the drift std after n frames
    kappa * sqrt(sum texture_sigma^2); without it, TAPNext++'s per-frame
    sigma cannot explain errors that slowly accumulate on weak texture.
    """
    report = {"tap_fits": {}}
    for mode in ("local", "global"):
        raw = TapCalibration(
            sigma_mode=mode, sigma_scale=1.0, sigma_floor=0.0, drift_scale=0.0
        )
        d, sg, tx, cum, tm, cats = _concat(
            [
                tap_series(q, r, raw)
                for q, r in zip(seqs, raws_list, strict=True)
            ]
        )
        entry = {"raw": coverage_report(d, K95 * sg, cats)}
        for drift in (False, True):
            a, f, kappa = fit_radius_model(
                d, sg, cum, tm, drift=drift, texture=tx
            )
            r = K95 * np.sqrt((a * sg) ** 2 + f * f + kappa**2 * cum)
            key = "drift" if drift else "white"
            entry[key] = {
                "scale": a,
                "floor": f,
                "drift_scale": kappa,
                "calib": coverage_report(d, r, cats),
            }
            for name, (q, rr) in held_out:
                d2, s2, _, c2, _, k2 = tap_series(q, rr, raw)
                r2 = K95 * np.sqrt((a * s2) ** 2 + f * f + kappa**2 * c2)
                entry[key][name] = coverage_report(d2, r2, k2)
            LOGGER.info(
                "tap %s %s: %s",
                mode,
                key,
                {
                    k: v["all"] if isinstance(v, dict) and "all" in v else v
                    for k, v in entry[key].items()
                },
            )
        report["tap_fits"][mode] = entry
    report["tap_best"] = "local"  # sharper and better calibrated (see report)
    return report


def calibrate(work: Path):
    seqs = [load(work, n) for n in CALIBRATION]
    report = calibrate_sift(seqs)
    LOGGER.info("sift fit %s", report["sift_fit"])
    LOGGER.info("sift calibrated %s", report["sift_calibrated"])
    raws = []
    for n in CALIBRATION:
        with open(work / f"{n}_tap.pkl", "rb") as f:
            raws.append(pickle.load(f))
    held_out = []
    for n in SEQUENCES:
        if n in CALIBRATION:
            continue
        with open(work / f"{n}_tap.pkl", "rb") as f:
            held_out.append((n, (load(work, n), pickle.load(f))))
    report.update(calibrate_tap(seqs, raws, held_out))
    report["tap_occluded"] = calibrate_occluded(seqs, raws, held_out)
    LOGGER.info("tap occluded %s", report["tap_occluded"])
    (work / "calibration.json").write_text(json.dumps(report, indent=1))
    return report


def sift_params_from(fit: dict) -> SiftParams:
    return SiftParams(
        sigma_scale=fit["scale"],
        sigma_floor=fit["floor"],
        texture_scale=fit.get("texture_scale", 0.0),
        gross_bias=fit["gross_bias"],
        gross_w_inliers=fit["gross_w_inliers"],
        gross_w_ratio=fit["gross_w_ratio"],
        gross_sigma=fit["gross_sigma"],
    )


# ---------------------------------------------------------------- compare
def calibrated_params(work: Path):
    rep = json.loads((work / "calibration.json").read_text())
    tf = rep["tap_fits"][rep["tap_best"]]["drift"]
    occ = rep.get("tap_occluded", {})
    tap = TapCalibration(
        sigma_mode=rep["tap_best"],
        sigma_scale=tf["scale"],
        sigma_floor=tf["floor"],
        drift_scale=tf["drift_scale"],
        occluded_scale=occ.get("scale", TapCalibration.occluded_scale),
        occluded_floor=occ.get("floor", TapCalibration.occluded_floor),
    )
    return tap, sift_params_from(rep["sift_fit"])


def evaluate_run(seq, records, fail_px=8.0):
    cats = np.array(seq["categories"])
    rows = {}
    for cat in list(city.CATEGORIES) + ["all"]:
        sel = np.ones(len(cats), bool) if cat == "all" else cats == cat
        errs, covered, lost, radii = [], [], [], []
        for k in np.nonzero(sel)[0]:
            for rec in records[k][1:]:
                t = rec.frame
                if not seq["gt_vis"][t, k]:
                    continue
                e = rec.fused.position - seq["gt_xy"][t, k]
                errs.append(np.linalg.norm(e))
                if rec.fused.lost:
                    lost.append(1)
                    covered.append(0)
                    continue
                lost.append(0)
                # Is the truth inside the drawn 95 % circle?
                r95 = rec.fused.radius()
                covered.append(np.linalg.norm(e) <= r95)
                radii.append(r95)
        errs = np.array(errs)
        rows[cat] = {
            "median": float(np.median(errs)),
            "mean": float(np.mean(np.minimum(errs, 100))),
            "fail%": float(100 * np.mean(errs > fail_px)),
            "lost%": float(100 * np.mean(lost)),
            "cover95%": float(100 * np.mean(covered)),
            "r95": float(np.median(radii)) if radii else float("inf"),
        }
    return rows


def compare(work: Path, fusions=FUSIONS, accel_sigma=1.5):
    tap_cal, sift_par = calibrated_params(work)
    results = {}
    for name in [n for n, c in SEQUENCES.items() if "pillars" not in c]:
        seq = load(work, name)
        with open(work / f"{name}_tap.pkl", "rb") as f:
            raws = pickle.load(f)
        for fusion in (*fusions, "kalman_taponly"):
            opts = TrackerOptions(
                fusion=fusion.replace("_taponly", ""),
                accel_sigma=accel_sigma,
                use_tap=fusion != "sift",
                use_sift=fusion not in ("tap", "kalman_taponly"),
            )
            tracker = SiftapTracker(
                make_targets(seq), None, opts, tap_cal, sift_par
            )
            records = tracker.run(seq["frames"], tap_raw=raws)
            results[f"{name}/{fusion}"] = evaluate_run(seq, records)
            LOGGER.info(
                "%s/%s all=%s", name, fusion, results[f"{name}/{fusion}"]["all"]
            )
    (work / "compare.json").write_text(json.dumps(results, indent=1))
    return results


def _render_demo(seq, records, targets, pick, names, path, width, focus):
    from siftap.focus import render_focus
    from siftap.render import render_frames, write_video

    recs = [records[k] for k in pick]
    tgts = [targets[k] for k in pick]
    truth = seq["gt_xy"][:, pick]
    # Draw the truth whenever it is in the image, also behind occluders:
    # that is where a predicted (hidden) position should be.
    height, width_px = seq["frames"][0].shape[:2]
    visible = (
        (truth[..., 0] >= 0)
        & (truth[..., 0] < width_px)
        & (truth[..., 1] >= 0)
        & (truth[..., 1] < height)
    )
    video = render_frames(
        seq["frames"],
        recs,
        tgts,
        names,
        width=width,
        truth=truth,
        truth_visible=visible,
    )
    if focus:
        video = render_focus(
            video,
            seq["frames"],
            recs,
            tgts,
            names,
            list(range(len(pick))),
            half=32.0,
            truth=truth,
            truth_visible=visible,
        )
    frames = list(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(frames, path.with_suffix(".mp4"), fps=8)
    LOGGER.info("demo written to %s", path.with_suffix(".mp4"))


def demo(work: Path, out_dir: Path, width=720):
    """Held-out synthetic demos with the truth drawn (white ring in the
    frame and in the stabilized focus panels, dotted error in the
    timelines).

    * ``test``: one target per surface category, default settings.
    * ``occl``: three targets that pass behind pillars, once per occlusion
      policy (predict / drop).
    """
    seq = load(work, "test")
    with open(work / "test_tap.pkl", "rb") as f:
        raws = pickle.load(f)
    targets = make_targets(seq)
    records = SiftapTracker(targets, None, TrackerOptions()).run(
        seq["frames"], tap_raw=raws
    )
    cats = np.array(seq["categories"])
    pick = []
    for c in CATEGORIES:
        idx = np.flatnonzero(cats == c)
        # Longest visible; ties: the first sampled.
        pick += sorted(idx, key=lambda k: -int(seq["gt_vis"][:, k].sum()))[:1]
    _render_demo(
        seq,
        records,
        targets,
        pick,
        [str(cats[k]) for k in pick],
        out_dir / "siftap_synthetic_test",
        width,
        focus=True,
    )
    seq = load(work, "occl")
    with open(work / "occl_tap.pkl", "rb") as f:
        raws = pickle.load(f)
    targets = make_targets(seq)
    vis = seq["gt_vis"]
    # Targets with the most hide-and-reappear episodes (one per category
    # where possible).
    episodes = (vis[:-1] & ~vis[1:]).sum(0)
    cats = np.array(seq["categories"])
    pick = []
    for c in ("poster", "facade", "ground", "sky"):
        idx = [k for k in np.flatnonzero(cats == c) if episodes[k] > 0]
        if idx and len(pick) < 3:
            pick.append(max(idx, key=lambda k: episodes[k]))
    for policy in ("predict", "drop"):
        records = SiftapTracker(
            targets, None, TrackerOptions(occlusion=policy)
        ).run(seq["frames"], tap_raw=raws)
        _render_demo(
            seq,
            records,
            targets,
            pick,
            [str(cats[k]) for k in pick],
            out_dir / f"siftap_occlusion_{policy}",
            width,
            focus=True,
        )


def occlusion_eval(work: Path, thresholds=(0.0, 0.25, 0.5, 0.75, 0.9)):
    """Occlusion policies on the held-out sequences.

    One tracker run per policy; the reporting threshold only decides which
    frames are reported, so every threshold is evaluated from that run.
    Frames are split by the truth: visible, or hidden but inside the image.
    """
    from siftap.tracker import frame_status

    tap_cal, sift_par = calibrated_params(work)
    out = {}
    for name in [n for n in SEQUENCES if n not in CALIBRATION]:
        seq = load(work, name)
        with open(work / f"{name}_tap.pkl", "rb") as f:
            raws = pickle.load(f)
        height, width = seq["frames"][0].shape[:2]
        gt, vis = seq["gt_xy"], seq["gt_vis"]
        for policy in ("predict", "drop"):
            opts = TrackerOptions(fusion="kalman", occlusion=policy)
            records = SiftapTracker(
                make_targets(seq), None, opts, tap_cal, sift_par
            ).run(seq["frames"], tap_raw=raws)
            for x in thresholds:
                acc = {"visible": [], "hidden": []}
                resume = []
                for k, recs in enumerate(records):
                    gap = None
                    for r in recs[1:]:
                        t = r.frame
                        gx, gy = gt[t, k]
                        if not (0 <= gx < width and 0 <= gy < height):
                            gap = None
                            continue
                        st = frame_status(r.hidden, r.confidence, policy, x)
                        part = "visible" if vis[t, k] else "hidden"
                        e = float(np.linalg.norm(r.fused.position - gt[t, k]))
                        acc[part].append((st != "dropped", e, r.fused.radius()))
                        # Frames from reappearance to being reported again.
                        if not vis[t, k]:
                            gap = 0
                        elif gap is not None:
                            if st != "dropped":
                                resume.append(gap)
                                gap = None
                            else:
                                gap += 1
                row = {}
                for part, items in acc.items():
                    if not items:
                        continue
                    rep_ = [(e, r) for ok, e, r in items if ok]
                    errs = np.array([e for e, _ in rep_])
                    row[part] = {
                        "n": len(items),
                        "reported%": 100 * len(rep_) / len(items),
                        "median": float(np.median(errs)) if rep_ else None,
                        "p90": float(np.quantile(errs, 0.9)) if rep_ else None,
                        "fail%": float(100 * np.mean(errs > 8))
                        if rep_
                        else None,
                        "cover95%": float(
                            100 * np.mean([e <= r for e, r in rep_])
                        )
                        if rep_
                        else None,
                    }
                row["resume_frames"] = (
                    float(np.mean(resume)) if resume else None
                )
                out[f"{name}/{policy}/{x}"] = row
                LOGGER.info("%s %s x=%.2f %s", name, policy, x, row)
    (work / "occlusion.json").write_text(json.dumps(out, indent=1))
    return out


FUSION_LABELS = {
    "tap": "TAPNext++ alone, heatmap sigma only (no drift term)",
    "sift": "SIFT alone (no TAP to guide the search)",
    "best": "best-of (smaller radius wins)",
    "ivw": "naive Bayes (product of Gaussians)",
    "bayes": "robust Bayes (gate + mixture, no motion)",
    "kalman_nodrift": "Kalman, gate + mixture, white TAP noise",
    "kalman": "**Kalman, gate + mixture + TAP drift (default)**",
    "kalman_nogate": "Kalman without gate",
    "kalman_ci": "Kalman, covariance intersection",
    "kalman_taponly": "Kalman + drift, TAPNext++ only",
}


def tables(work: Path) -> str:
    """Markdown tables for the README from calibration.json/compare.json."""
    lines = []
    cal = json.loads((work / "calibration.json").read_text())
    cmp_ = json.loads((work / "compare.json").read_text())
    pct = "{:.0f} %".format
    lines += [
        "**Measurement calibration** (truth inside the drawn 95 % circle /"
        " median radius px):",
        "",
        "| model | sequence | all | sky | facade | poster | ground |",
        "|---|---|---|---|---|---|---|",
    ]

    def cells(d):
        return [
            f"{pct(100 * d[c]['circle95'])} / {d[c]['median_r95']:.1f}"
            if c in d and d[c]["n"] >= 5
            else "-"
            for c in ("all", *CATEGORIES)
        ]

    lines.append(
        "| SIFT, LS covariance only | calib | "
        + " | ".join(cells(cal["sift_raw"]))
        + " |"
    )
    lines.append(
        "| SIFT, calibrated + mismatch mixture | calib | "
        + " | ".join(cells(cal["sift_calibrated"]))
        + " |"
    )
    tf = cal["tap_fits"]["local"]
    lines.append(
        "| TAPNext++ raw heatmap | calib | "
        + " | ".join(cells(tf["raw"]))
        + " |"
    )
    for key, label in (
        ("white", "TAPNext++ heatmap, scaled"),
        ("drift", "TAPNext++ heatmap + texture drift"),
    ):
        for seq in ("calib", "test", "fast"):
            lines.append(
                f"| {label} | {seq} | " + " | ".join(cells(tf[key][seq])) + " |"
            )
    for seq in [
        n
        for n, c in SEQUENCES.items()
        if n not in CALIBRATION and "pillars" not in c
    ]:
        lines += [
            "",
            f"**Fusion on the held-out `{seq}` sequence** (all 40 targets):",
            "",
            "| fusion | median err px | fail % (> 8 px) "
            "| truth inside circle | median radius px |",
            "|---|---|---|---|---|",
        ]
        for f, label in FUSION_LABELS.items():
            key = f"{seq}/{f}"
            if key not in cmp_:
                continue
            a = cmp_[key]["all"]
            lines.append(
                f"| {label} | {a['median']:.2f} | {a['fail%']:.1f} | "
                f"{a['cover95%']:.0f} % | {a['r95']:.2f} |"
            )
        lines += [
            "",
            f"Per surface on `{seq}` (median error px / truth inside"
            " circle / median radius px):",
            "",
            "| fusion | sky | facade | poster | ground |",
            "|---|---|---|---|---|",
        ]
        for f in ("tap", "sift", "ivw", "kalman_nodrift", "kalman"):
            r = cmp_[f"{seq}/{f}"]
            row = [
                f"{r[c]['median']:.2f} / {r[c]['cover95%']:.0f} % / "
                f"{r[c]['r95']:.1f}"
                for c in CATEGORIES
            ]
            lines.append(f"| {FUSION_LABELS[f]} | " + " | ".join(row) + " |")
    occ_path = work / "occlusion.json"
    if occ_path.exists():
        occ = json.loads(occ_path.read_text())
        lines += [
            "",
            "**Occlusion policies on `occl`** (60 hide-and-reappear episodes, "
            "617 hidden frames; confidence = P(error <= 4 px)):",
            "",
            "| policy | x | visible frames reported | visible: fail % "
            "| hidden frames reported | hidden: median err px | hidden: fail % "
            "| hidden: truth in circle | frames to resume |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        def num(v, fmt):
            return "-" if v is None else fmt.format(v)

        for policy in ("predict", "drop"):
            for x in (0.0, 0.5, 0.9):
                r = occ.get(f"occl/{policy}/{x}")
                if r is None:
                    continue
                v, h = r["visible"], r["hidden"]
                lines.append(
                    f"| {policy} | {100 * x:.0f} % | {v['reported%']:.0f} % "
                    f"| {num(v['fail%'], '{:.1f}')} "
                    f"| {h['reported%']:.0f} % | {num(h['median'], '{:.2f}')} "
                    f"| {num(h['fail%'], '{:.1f}')} "
                    f"| {num(h['cover95%'], '{:.0f} %')} "
                    f"| {num(r['resume_frames'], '{:.2f}')} |"
                )
    text = "\n".join(lines)
    (work / "tables.md").write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_path", required=True)
    parser.add_argument("--poster_path", required=True)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output_path", default="assets")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s"
    )
    work = Path(args.work_path)
    work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("render", "all"):
        render(work, load_posters(args.poster_path))
    if args.stage in ("tap", "all"):
        run_tap(work, args.precision)
    if args.stage in ("calibrate", "all"):
        calibrate(work)
    if args.stage in ("compare", "all"):
        compare(work)
    if args.stage in ("occlusion", "all"):
        occlusion_eval(work)
    if args.stage in ("demo", "all"):
        demo(work, Path(args.output_path))
    if args.stage in ("tables", "all"):
        print(tables(work))


if __name__ == "__main__":
    main()
