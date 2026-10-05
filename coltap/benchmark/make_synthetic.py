# SPDX-License-Identifier: BSD-3-Clause
"""Render a static synthetic sequence with exact ground truth.

    python benchmark/make_synthetic.py --output_path data/synth_room \
        --num_frames 48 --arc_degrees 60 [--texture_path some/photos]

Without ``--texture_path`` procedural textures are used. The published
numbers used photos from the VGGT ``kitchen`` example and the instant-ngp
``fox`` images as textures.
"""

import argparse
from pathlib import Path

import cv2

from coltap import synthetic
from coltap.frames import list_images, read_image


def load_textures(texture_path: str, max_textures: int = 12, size: int = 768):
    if not texture_path:
        return [synthetic.procedural_texture(512, seed=s) for s in range(8)]
    names = list_images(texture_path)
    step = max(1, len(names) // max_textures)
    textures = []
    for name in names[::step][:max_textures]:
        image = read_image(Path(texture_path) / name)
        scale = size / max(image.shape[:2])
        textures.append(
            cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        )
    return textures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_frames", type=int, default=48)
    parser.add_argument("--arc_degrees", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--focal", type=float, default=500.0)
    parser.add_argument("--texture_path", default="")
    args = parser.parse_args()
    scene = synthetic.Scene(load_textures(args.texture_path))
    camera = synthetic.Camera(args.width, args.height, args.focal)
    poses = synthetic.orbit_trajectory(
        args.num_frames, arc_degrees=args.arc_degrees
    )
    synthetic.write_sequence(args.output_path, scene, camera, poses)
    print(f"Wrote {args.num_frames} frames + gt.json to {args.output_path}")


if __name__ == "__main__":
    main()
