# SPDX-License-Identifier: BSD-3-Clause
"""Image-sequence and video I/O with COLMAP-compatible conventions."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _natural_key(name: str):
    return [
        int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name)
    ]


def list_images(image_path: str | Path) -> list[str]:
    """Image names relative to ``image_path`` (COLMAP style, '/' separated),
    in natural sort order, which is the temporal order used for tracking."""
    root = Path(image_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    names = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(names, key=_natural_key)


def read_image(path: str | Path, apply_orientation: bool = False) -> np.ndarray:
    """Read an image as uint8 RGB [H, W, 3].

    By default EXIF orientation is ignored so that pixel coordinates match the
    stored raster, which is what COLMAP's ImageReader records as width/height.
    """
    flags = cv2.IMREAD_COLOR
    if not apply_orientation:
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imread(str(path), flags)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(
    mask_path: str | Path | None, image_name: str, shape: tuple[int, int]
) -> np.ndarray | None:
    """Read a COLMAP-style mask (``<mask_path>/<image_name>.png``).

    Returns a bool [H, W] array that is True where features are allowed, or
    None when no mask exists. As in COLMAP, black (0) pixels are excluded.
    """
    if not mask_path:
        return None
    path = Path(mask_path) / f"{image_name}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"Failed to read mask: {path}")
    if mask.shape[:2] != tuple(shape):
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask > 0


class ImageSequence:
    """Ordered image collection with lazy loading."""

    def __init__(
        self,
        image_path: str | Path,
        image_names: Sequence[str] | None = None,
        mask_path: str | Path | None = None,
        apply_orientation: bool = False,
    ):
        self.root = Path(image_path)
        self.names = (
            list(image_names) if image_names else list_images(self.root)
        )
        if not self.names:
            raise ValueError(f"No images found in {self.root}")
        self.mask_path = mask_path
        self.apply_orientation = apply_orientation

    def __len__(self) -> int:
        return len(self.names)

    def image(self, index: int) -> np.ndarray:
        return read_image(self.root / self.names[index], self.apply_orientation)

    def mask(self, index: int, shape: tuple[int, int]) -> np.ndarray | None:
        return read_mask(self.mask_path, self.names[index], shape)


def extract_video_frames(
    video_path: str | Path,
    image_path: str | Path,
    stride: int = 1,
    max_frames: int = 0,
    start: int = 0,
    max_size: int = 0,
    prefix: str = "frame_",
) -> list[str]:
    """Decode a video into ``image_path/frame_XXXXXX.jpg`` files.

    Args:
        stride: keep every ``stride``-th frame.
        max_frames: stop after this many written frames (0 = all).
        start: index of the first decoded frame to keep.
        max_size: downscale so that max(width, height) <= max_size (0 = off).
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Failed to open video: {video_path}")
    out_dir = Path(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index < start or (index - start) % max(stride, 1) != 0:
            continue
        if max_size > 0 and max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            size = (
                round(frame.shape[1] * scale),
                round(frame.shape[0] * scale),
            )
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        name = f"{prefix}{len(names):06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
        if max_frames > 0 and len(names) >= max_frames:
            break
    capture.release()
    LOGGER.info("Extracted %d frames from %s", len(names), video_path)
    return names
