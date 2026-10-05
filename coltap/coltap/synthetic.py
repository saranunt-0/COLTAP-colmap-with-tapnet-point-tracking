# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic static scenes with exact ground truth, for tests and benchmarks.

The scene is the inside of a textured box ("room") with a few textured
cuboids standing on the floor, which gives parallax and occlusions. A pinhole
camera moves smoothly through the room. Everything is ray-cast exactly, so
ground-truth camera poses, 3D points and 2D tracks are available.

Conventions match COLMAP: camera x right, y down, z forward; ``cam_from_world``
maps world to camera; pixel centers are at +0.5.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np


@dataclasses.dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    interior: bool = False  # True for the room (camera inside).


@dataclasses.dataclass
class Camera:
    width: int
    height: int
    focal: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.focal, 0, self.width / 2],
                [0, self.focal, self.height / 2],
                [0, 0, 1],
            ]
        )


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)):
    """cam_from_world rotation R and translation t for a camera at center."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])  # rows = camera axes in world coordinates
    return rot, -rot @ center


def procedural_texture(size: int = 512, seed: int = 0) -> np.ndarray:
    """Multi-scale colored noise with blobs: textured and corner-rich."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size, 3), np.float32)
    for octave in (4, 8, 16, 32, 64):
        noise = rng.random((octave, octave, 3)).astype(np.float32)
        tex += cv2.resize(
            noise, (size, size), interpolation=cv2.INTER_CUBIC
        ) / (octave**0.3)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    for _ in range(60):
        c = rng.integers(0, size, 2)
        r = int(rng.integers(4, size // 12))
        color = rng.random(3).tolist()
        cv2.circle(tex, (int(c[0]), int(c[1])), r, color, -1)
    return (np.clip(tex, 0, 1) * 255).astype(np.uint8)


class Scene:
    """Textured room with cuboids. Textures are uint8 RGB images."""

    def __init__(
        self,
        textures: list[np.ndarray],
        room_size=(8.0, 4.0, 8.0),
        boxes: list[tuple[tuple, tuple]] | None = None,
        texels_per_unit: float = 160.0,
        seed: int = 0,
    ):
        half = np.array(room_size) / 2
        # y is down: the floor is at y = +half[1].
        self.boxes = [Box(-half, half, interior=True)]
        if boxes is None:
            floor = half[1]
            boxes = [
                ((-1.6, floor - 1.4, 0.4), (-0.6, floor, 1.4)),
                ((0.5, floor - 0.9, -0.2), (1.7, floor, 0.8)),
                ((-0.4, floor - 2.0, 1.8), (0.4, floor, 2.6)),
            ]
        for lo, hi in boxes:
            self.boxes.append(Box(np.array(lo, float), np.array(hi, float)))
        self.textures = textures
        self.texels_per_unit = texels_per_unit
        rng = np.random.default_rng(seed)
        num_faces = 6 * len(self.boxes)
        self.face_texture = rng.integers(0, len(textures), num_faces)
        self.face_offset = rng.random((num_faces, 2)) * 1000.0
        self.face_shade = rng.uniform(0.75, 1.0, num_faces)

    def raycast(self, origins: np.ndarray, dirs: np.ndarray):
        """Nearest hit per ray. Returns (distance, face_id, hit_points)."""
        n = len(dirs)
        best_t = np.full(n, np.inf)
        best_face = np.full(n, -1, np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dirs
            for b, box in enumerate(self.boxes):
                t1 = (box.lo - origins) * inv
                t2 = (box.hi - origins) * inv
                if box.interior:
                    t_far = np.maximum(t1, t2)
                    axis = np.argmin(t_far, axis=1)
                    t_hit = t_far[np.arange(n), axis]
                    side = dirs[np.arange(n), axis] > 0
                else:
                    t_near = np.minimum(t1, t2)
                    t_far = np.maximum(t1, t2)
                    axis = np.argmax(t_near, axis=1)
                    t_hit = t_near[np.arange(n), axis]
                    exit_t = t_far.min(axis=1)
                    side = dirs[np.arange(n), axis] < 0
                    t_hit = np.where(
                        (t_hit <= exit_t) & (t_hit > 1e-6), t_hit, np.inf
                    )
                better = t_hit < best_t
                best_t[better] = t_hit[better]
                best_face[better] = (b * 6 + axis * 2 + side)[better]
        points = origins + best_t[:, None] * dirs
        return best_t, best_face, points

    def shade(self, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Bilinear texture lookup for hit points, float RGB in [0, 255]."""
        colors = np.zeros((len(faces), 3), np.float32)
        for face in np.unique(faces):
            if face < 0:
                continue
            sel = faces == face
            axis = (face % 6) // 2
            uv_axes = [a for a in range(3) if a != axis]
            uv = points[sel][:, uv_axes] * self.texels_per_unit
            uv = uv + self.face_offset[face]
            tex = self.textures[self.face_texture[face]]
            mapx = np.mod(uv[:, 0], tex.shape[1]).astype(np.float32)
            mapy = np.mod(uv[:, 1], tex.shape[0]).astype(np.float32)
            # cv2.remap needs < 32767 rows/cols: lay the samples out in rows.
            n, cols = len(mapx), 1024
            pad = (-n) % cols
            mapx = np.pad(mapx, (0, pad)).reshape(-1, cols)
            mapy = np.pad(mapy, (0, pad)).reshape(-1, cols)
            sampled = cv2.remap(
                tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
            ).reshape(-1, 3)[:n]
            colors[sel] = sampled * self.face_shade[face]
        return colors

    def render(self, camera: Camera, rot, trans, supersample: int = 2):
        w, h, s = camera.width, camera.height, supersample
        u, v = np.meshgrid(
            (np.arange(w * s) + 0.5) / s, (np.arange(h * s) + 0.5) / s
        )
        rays = np.stack(
            [
                (u - camera.width / 2) / camera.focal,
                (v - camera.height / 2) / camera.focal,
                np.ones_like(u),
            ],
            axis=-1,
        ).reshape(-1, 3)
        dirs = rays @ rot  # camera -> world: R^T * ray
        center = -rot.T @ trans
        origins = np.broadcast_to(center, dirs.shape)
        _, faces, points = self.raycast(origins, dirs)
        image = self.shade(faces, points).reshape(h * s, w * s, 3)
        if s > 1:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return np.clip(image, 0, 255).astype(np.uint8)

    def unproject(self, camera: Camera, rot, trans, xy: np.ndarray):
        """3D points seen at pixels ``xy`` (COLMAP convention)."""
        rays = np.stack(
            [
                (xy[:, 0] - camera.width / 2) / camera.focal,
                (xy[:, 1] - camera.height / 2) / camera.focal,
                np.ones(len(xy)),
            ],
            axis=-1,
        )
        dirs = rays @ rot
        center = -rot.T @ trans
        _, _, points = self.raycast(np.broadcast_to(center, dirs.shape), dirs)
        return points

    def project(self, camera: Camera, rot, trans, points: np.ndarray):
        """Pixel coordinates and visibility (in view and not occluded)."""
        cam = points @ rot.T + trans
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = camera.focal * cam[:, :2] / z[:, None]
        xy = xy + np.array([camera.width / 2, camera.height / 2])
        inside = (
            (z > 1e-6)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < camera.width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < camera.height)
        )
        center = -rot.T @ trans
        to_point = points - center
        dist = np.linalg.norm(to_point, axis=1)
        dirs = to_point / np.maximum(dist, 1e-12)[:, None]
        hit_t, _, _ = self.raycast(np.broadcast_to(center, dirs.shape), dirs)
        unoccluded = hit_t >= dist * (1 - 1e-4) - 1e-4
        return xy, inside & unoccluded


def orbit_trajectory(
    num_frames: int,
    radius: float = 2.3,
    height: float = -0.2,
    arc_degrees: float = 100.0,
    target=(0.0, 0.8, 1.0),
    start_degrees: float = 230.0,
):
    """Smooth arc around the cuboids, always looking at ``target``."""
    poses = []
    for i in range(num_frames):
        a = np.deg2rad(start_degrees + arc_degrees * i / max(num_frames - 1, 1))
        bob = 0.15 * np.sin(2 * np.pi * i / max(num_frames - 1, 1))
        center = np.array(
            [radius * np.cos(a), height + bob, 1.0 + radius * np.sin(a)]
        )
        poses.append(look_at(center, np.asarray(target, float)))
    return poses


def write_sequence(
    out_dir: str | Path,
    scene: Scene,
    camera: Camera,
    poses,
    supersample: int = 2,
    noise_sigma: float = 1.0,
    seed: int = 0,
) -> Path:
    """Render frames to ``out_dir/images`` and ground truth to gt.json."""
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gt = {
        "camera": dataclasses.asdict(camera),
        "images": [],
    }
    for i, (rot, trans) in enumerate(poses):
        image = scene.render(camera, rot, trans, supersample).astype(np.float32)
        image += rng.normal(0, noise_sigma, image.shape)
        name = f"frame_{i:06d}.png"
        cv2.imwrite(
            str(image_dir / name),
            cv2.cvtColor(
                np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )
        gt["images"].append(
            {"name": name, "R": rot.tolist(), "t": trans.tolist()}
        )
    (out_dir / "gt.json").write_text(json.dumps(gt, indent=1))
    return out_dir


def load_ground_truth(path: str | Path):
    gt = json.loads(Path(path).read_text())
    camera = Camera(**gt["camera"])
    poses = {
        im["name"]: (np.array(im["R"]), np.array(im["t"]))
        for im in gt["images"]
    }
    return camera, poses


def ground_truth_tracks(tracks, scene: Scene, camera: Camera, poses):
    """GT 2D positions/visibility for every track, lifted at its query frame.

    Only the scene geometry is used (textures are irrelevant), so ``scene``
    can be rebuilt with any textures from the same box layout.
    Returns ``(xy [N, T, 2], visible [N, T])``.
    """
    names = tracks.image_names
    gt_xy = np.full(tracks.xy.shape, np.nan)
    visible = np.zeros(tracks.visibility.shape, bool)
    for q in np.unique(tracks.query_frame):
        rows = np.nonzero(tracks.query_frame == q)[0]
        rot, trans = poses[names[q]]
        points = scene.unproject(camera, rot, trans, tracks.xy[rows, q])
        for f, name in enumerate(names):
            rot, trans = poses[name]
            gt_xy[rows, f], visible[rows, f] = scene.project(
                camera, rot, trans, points
            )
    return gt_xy, visible


def track_errors(tracks, gt_xy, gt_visible, min_confidence=0.5) -> dict:
    """Pixel error and visibility precision/recall of kept observations,
    excluding each track's query frame (where it is exact by definition)."""
    observed = tracks.observed(min_confidence)
    observed &= tracks.query_frame[:, None] != np.arange(tracks.num_frames)
    err = np.linalg.norm(tracks.xy - gt_xy, axis=-1)[observed & gt_visible]
    return {
        "observations": int(observed.sum()),
        "visibility_precision": float(gt_visible[observed].mean()),
        "error_median_px": float(np.median(err)),
        "error_mean_px": float(np.mean(err)),
        "within_1px": float(np.mean(err < 1)),
        "within_2px": float(np.mean(err < 2)),
    }
