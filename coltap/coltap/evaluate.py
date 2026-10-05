# SPDX-License-Identifier: BSD-3-Clause
"""Reconstruction statistics and pose accuracy against ground truth."""

from __future__ import annotations

import numpy as np

import pycolmap


def reconstruction_stats(rec: pycolmap.Reconstruction, num_images: int) -> dict:
    track_lengths = [p.track.length() for p in rec.points3D.values()]
    return {
        "registered_images": rec.num_reg_images(),
        "total_images": num_images,
        "points3D": rec.num_points3D(),
        "observations": int(sum(track_lengths)),
        "mean_track_length": float(np.mean(track_lengths))
        if track_lengths
        else 0,
        "mean_observations_per_image": (
            rec.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error": rec.compute_mean_reprojection_error(),
    }


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimizing ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / xs.var(0).sum()
    return scale, rot, mu_d - scale * rot @ mu_s


def pose_errors(
    rec: pycolmap.Reconstruction,
    gt_poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Absolute trajectory error and relative rotation error (RRE).

    ``gt_poses`` maps image name -> (R, t) of cam_from_world. Camera centers
    are aligned with a Sim3 (Umeyama); translation errors are reported in GT
    units and relative to the GT trajectory extent. Rotation accuracy uses
    relative rotations between all camera pairs, which needs no alignment.
    """
    names, est_c, gt_c, est_r, gt_r = [], [], [], [], []
    for image in rec.images.values():
        if not image.has_pose or image.name not in gt_poses:
            continue
        rot, trans = gt_poses[image.name]
        pose = image.cam_from_world()
        names.append(image.name)
        est_c.append(image.projection_center())
        est_r.append(pose.rotation.matrix())
        gt_c.append(-rot.T @ trans)
        gt_r.append(rot)
    if len(names) < 3:
        return {"aligned_images": len(names)}
    est_c, gt_c = np.array(est_c), np.array(gt_c)
    scale, rot, trans = umeyama(est_c, gt_c)
    aligned = scale * est_c @ rot.T + trans
    center_err = np.linalg.norm(aligned - gt_c, axis=1)
    extent = np.linalg.norm(gt_c - gt_c.mean(0), axis=1).max()
    # Gauge-free rotation accuracy: relative rotation between every pair of
    # registered cameras, compared with ground truth.
    est_r, gt_r = np.array(est_r), np.array(gt_r)
    i, j = np.triu_indices(len(names), k=1)
    rel_est = est_r[i] @ est_r[j].transpose(0, 2, 1)
    rel_gt = gt_r[i] @ gt_r[j].transpose(0, 2, 1)
    delta = rel_est @ rel_gt.transpose(0, 2, 1)
    cos = np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)
    rre = np.degrees(np.arccos(cos))
    return {
        "aligned_images": len(names),
        "ate_rmse": float(np.sqrt(np.mean(center_err**2))),
        "ate_rmse_rel": float(np.sqrt(np.mean(center_err**2)) / extent),
        "center_err_median": float(np.median(center_err)),
        "rre_mean_deg": float(rre.mean()),
        "rre_median_deg": float(np.median(rre)),
        "rre_max_deg": float(rre.max()),
    }


def largest_reconstruction(
    reconstructions: dict[int, pycolmap.Reconstruction],
) -> pycolmap.Reconstruction | None:
    if not reconstructions:
        return None
    return max(reconstructions.values(), key=lambda r: r.num_reg_images())
