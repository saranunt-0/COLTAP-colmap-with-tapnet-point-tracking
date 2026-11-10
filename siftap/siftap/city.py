# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic street scene with exact ground truth for tracker evaluation.

Built on coltap.synthetic's ray caster. Surfaces are chosen to stress the
two trackers differently:

* ``sky``: a dome with a soft gradient and faint clouds (almost no texture,
  SIFT finds nothing);
* ``facade``: building walls with a repetitive window grid (SIFT matches are
  ambiguous);
* ``poster``: building walls covered with photographs (distinctive, SIFT's
  best case);
* ``ground``: fine random texture.

Buildings at different depths occlude each other as the camera moves; an
optional motion blur averages sub-frame renders.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from coltap import synthetic

CATEGORIES = ("sky", "facade", "poster", "ground")


def sky_texture(size=1024, seed=0, vertical=True) -> np.ndarray:
    """Pale horizon to blue zenith with faint large clouds (low texture)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, size)[:, None]  # 0 = top of texture
    zenith = np.array([95, 140, 205], np.float32)
    horizon = np.array([200, 215, 230], np.float32)
    grad = zenith + (horizon - zenith) * t[..., None] ** 0.7
    tex = np.broadcast_to(grad, (size, size, 3)).copy()
    clouds = np.zeros((size, size), np.float32)
    for octave in (3, 6, 12):
        noise = rng.random((octave, octave)).astype(np.float32)
        clouds += (
            cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
            / octave**0.5
        )
    clouds = (clouds - clouds.mean()) / (clouds.std() + 1e-6)
    tex += 6.0 * clouds[..., None]  # a few grey levels: barely visible
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return tex if vertical else tex.transpose(1, 0, 2).copy()


def facade_texture(size=512, seed=0, windows=6) -> np.ndarray:
    """Wall with a regular grid of identical windows (repetitive)."""
    rng = np.random.default_rng(seed)
    wall = np.array(rng.uniform(150, 210, 3), np.float32)
    tex = np.empty((size, size, 3), np.float32)
    tex[:] = wall
    tex += rng.normal(0, 3, tex.shape)
    cell = size // windows
    for i in range(windows):
        for j in range(windows):
            x0, y0 = j * cell + cell // 4, i * cell + cell // 5
            x1, y1 = x0 + cell // 2, y0 + int(cell * 0.55)
            cv2.rectangle(
                tex, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (90, 85, 80), -1
            )
            cv2.rectangle(tex, (x0, y0), (x1, y1), (60, 75, 95), -1)
            cv2.line(
                tex,
                ((x0 + x1) // 2, y0),
                ((x0 + x1) // 2, y1),
                (200, 200, 200),
                2,
            )
    return np.clip(tex, 0, 255).astype(np.uint8)


def ground_texture(size=512, seed=0) -> np.ndarray:
    return synthetic.procedural_texture(size, seed=seed + 100)


@dataclasses.dataclass
class CityScene:
    scene: synthetic.Scene
    camera: synthetic.Camera
    face_category: dict  # face id -> category name

    def category_of(self, faces: np.ndarray) -> np.ndarray:
        return np.array([self.face_category.get(int(f), "") for f in faces])


def make_city(
    posters: list[np.ndarray],
    seed: int = 0,
    width: int = 640,
    height: int = 360,
    focal: float = 420.0,
    pillars: int = 0,
) -> CityScene:
    """Street with 8 buildings in front of a sky dome (alternating window
    facades and posters). ``pillars`` thin columns close to the camera
    sweep across the view during the lateral dolly: targets behind them
    disappear for a few frames and reappear (occlusion tests)."""
    rng = np.random.default_rng(seed)
    half = np.array([40.0, 15.0, 40.0])  # dome; y is down, ground at +15
    ground = half[1]
    boxes = []
    for k in range(8):
        depth = rng.uniform(8, 26)
        x = rng.uniform(-6, 6) + (k - 3.5) * 3.5
        w, d = rng.uniform(2.5, 5.0), rng.uniform(2.0, 4.0)
        h = rng.uniform(4.0, 11.0)
        boxes.append(
            (
                (x - w / 2, ground - h, depth - d / 2),
                (x + w / 2, ground, depth + d / 2),
            )
        )
    first_pillar = len(boxes) + 1  # box 0 is the dome
    for x in np.linspace(-7.0, 7.0, pillars) + rng.uniform(-0.6, 0.6, pillars):
        w, z = rng.uniform(0.25, 0.35), rng.uniform(1.0, 3.0)
        boxes.append(
            ((x - w / 2, ground - 16.0, z), (x + w / 2, ground, z + w))
        )
    textures = [
        sky_texture(seed=seed, vertical=True),
        sky_texture(seed=seed, vertical=False),
        ground_texture(seed=seed),
    ]
    facade_ids = []
    for k in range(3):
        facade_ids.append(len(textures))
        textures.append(facade_texture(seed=seed * 10 + k, windows=5 + k))
    poster_ids = []
    for poster in posters:
        poster_ids.append(len(textures))
        textures.append(poster)
    scene = synthetic.Scene(
        textures, room_size=tuple(2 * half), boxes=boxes, seed=seed
    )
    category = {}
    # Dome (box 0): ceiling and walls are sky, the floor is ground.
    for axis in range(3):
        for side in range(2):
            face = 2 * axis + side
            if axis == 1 and side == 1:  # floor (y = +half)
                scene.face_texture[face] = 2
                scene.face_texels_per_unit[face] = 40.0
                category[face] = "ground"
                continue
            # One texture covers the whole wall: no tiling seams in the sky.
            extent = 2 * half[[a for a in range(3) if a != axis]]
            tpu = 1024 / extent.max()
            scene.face_texels_per_unit[face] = tpu
            scene.face_offset[face] = (
                half[[a for a in range(3) if a != axis]] * tpu
            )
            scene.face_shade[face] = 1.0
            # Gradient must follow world y: on x-walls y is the texture's
            # first uv axis (columns), on z-walls the second (rows).
            scene.face_texture[face] = 1 if axis == 0 else 0
            category[face] = "sky"
    for b in range(first_pillar, len(scene.boxes)):
        for f in range(6):
            scene.face_texture[6 * b + f] = facade_ids[0]
            scene.face_texels_per_unit[6 * b + f] = 60.0
            category[6 * b + f] = "occluder"
    for b in range(1, first_pillar):
        is_poster = bool(posters) and b % 2 == 0
        for f in range(6):
            face = 6 * b + f
            if is_poster:
                scene.face_texture[face] = poster_ids[
                    rng.integers(len(poster_ids))
                ]
                scene.face_texels_per_unit[face] = 140.0
                category[face] = "poster"
            else:
                scene.face_texture[face] = facade_ids[
                    rng.integers(len(facade_ids))
                ]
                scene.face_texels_per_unit[face] = 110.0
                category[face] = "facade"
    return CityScene(scene, synthetic.Camera(width, height, focal), category)


def street_trajectory(
    num_frames: int, speed: float = 0.12, seed: int = 0, subframes: int = 1
):
    """Lateral dolly with a slow pan and slight bob, looking down the street.

    Returns ``num_frames * subframes`` poses; with subframes > 1 consecutive
    groups are averaged into one motion-blurred frame.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    poses = []
    total = num_frames * subframes
    for i in range(total):
        s = (i / subframes) - (subframes - 1) / (2 * subframes)
        x = -0.5 * speed * num_frames + speed * s
        bob = 0.08 * np.sin(0.35 * s + phase)
        center = np.array([x, 13.3 + bob, -4.0 + 0.02 * s])
        look = np.array([0.4 * np.sin(0.05 * s + phase) + 0.5 * x, 10.5, 18.0])
        poses.append(synthetic.look_at(center, look))
    return poses


def render_sequence(
    city: CityScene, poses, subframes: int = 1, noise=1.5, seed=0
):
    """Frames (RGB uint8) and the GT pose of each frame (mid-exposure)."""
    rng = np.random.default_rng(seed)
    frames, frame_poses = [], []
    for start in range(0, len(poses), subframes):
        group = poses[start : start + subframes]
        acc = None
        for rot, trans in group:
            img = city.scene.render(city.camera, rot, trans).astype(np.float32)
            acc = img if acc is None else acc + img
        img = acc / len(group) + rng.normal(0, noise, acc.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        frame_poses.append(group[len(group) // 2])
    return frames, frame_poses


def ground_truth(city: CityScene, poses, points_xy0: np.ndarray):
    """GT image positions [T, N, 2] and visibility [T, N] of frame-0 pixels."""
    rot, trans = poses[0]
    world = city.scene.unproject(city.camera, rot, trans, points_xy0)
    xy, vis = [], []
    for rot, trans in poses:
        p, v = city.scene.project(city.camera, rot, trans, world)
        xy.append(p)
        vis.append(v)
    return np.array(xy), np.array(vis)


def face_at(city: CityScene, pose, xy: np.ndarray) -> np.ndarray:
    rot, trans = pose
    cam = city.camera
    rays = np.stack(
        [
            (xy[:, 0] - cam.width / 2) / cam.focal,
            (xy[:, 1] - cam.height / 2) / cam.focal,
            np.ones(len(xy)),
        ],
        axis=-1,
    )
    dirs = rays @ rot
    center = -rot.T @ trans
    _, faces, _ = city.scene.raycast(np.broadcast_to(center, dirs.shape), dirs)
    return faces


def sample_targets(
    city: CityScene,
    frame0: np.ndarray,
    pose0,
    per_category: int = 10,
    margin: int = 40,
    seed: int = 0,
    homogeneous_radius: int = 12,
):
    """Target pixels per category whose neighbourhood lies on one category.

    Poster and facade targets are taken at Shi-Tomasi corners (what a user
    would click on); sky and ground targets anywhere.
    """
    rng = np.random.default_rng(seed)
    h, w = frame0.shape[:2]
    gray = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 2000, 0.01, 8, blockSize=7)
    corners = (
        corners.reshape(-1, 2) + 0.5
        if corners is not None
        else np.zeros((0, 2))
    )
    grid = np.stack(
        np.meshgrid(
            np.arange(margin, w - margin, 6) + 0.5,
            np.arange(margin, h - margin, 6) + 0.5,
        ),
        -1,
    ).reshape(-1, 2)
    out = {}
    for cat in CATEGORIES:
        cand = corners if cat in ("poster", "facade") else grid
        cand = cand[
            (cand[:, 0] > margin)
            & (cand[:, 0] < w - margin)
            & (cand[:, 1] > margin)
            & (cand[:, 1] < h - margin)
        ]
        rng.shuffle(cand)
        chosen = []
        for p in cand:
            ring = p + homogeneous_radius * np.array(
                [
                    [1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [0.7, 0.7],
                    [-0.7, -0.7],
                    [0.7, -0.7],
                    [-0.7, 0.7],
                    [0, 0],
                ]
            )
            cats = city.category_of(face_at(city, pose0, ring))
            if (cats == cat).all() and all(
                np.hypot(*(p - c)) > 20 for c in chosen
            ):
                chosen.append(p)
            if len(chosen) >= per_category:
                break
        out[cat] = np.array(chosen).reshape(-1, 2)
    return out
