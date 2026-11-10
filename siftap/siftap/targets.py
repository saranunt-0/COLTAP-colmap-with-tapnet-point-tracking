# SPDX-License-Identifier: BSD-3-Clause
"""User-selected targets: a single point or an area (box)."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Target:
    """What to track, defined on the first frame.

    Coordinates follow COLMAP's convention (pixel centers at +0.5), so a
    click on pixel (i, j) is the point (i + 0.5, j + 0.5).

    ``radius`` is the size of the neighbourhood around a point target that
    SIFT uses for keypoints and TAPNext++ for support points. For an area
    target the box itself is the neighbourhood.
    """

    center: np.ndarray  # [2]
    box: np.ndarray | None = None  # [x0, y0, x1, y1] for area targets
    radius: float = 24.0
    name: str = ""

    @classmethod
    def point(cls, x: float, y: float, radius: float = 24.0, name: str = ""):
        return cls(np.array([x, y], float), None, radius, name)

    @classmethod
    def area(cls, x0, y0, x1, y1, name: str = ""):
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        radius = 0.5 * float(np.hypot(x1 - x0, y1 - y0))
        return cls(center, np.array([x0, y0, x1, y1]), radius, name)

    def scaled(self, s: float) -> Target:
        """The same target in an image resized by ``s``."""
        box = None if self.box is None else self.box * s
        return Target(self.center * s, box, self.radius * s, self.name)

    @property
    def is_area(self) -> bool:
        return self.box is not None

    def corner_offsets(self) -> np.ndarray:
        """Box corners relative to the center (area targets)."""
        if self.box is None:
            return np.zeros((0, 2))
        x0, y0, x1, y1 = self.box
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return corners - self.center

    def region_mask(self, shape: tuple[int, int], scale: float = 1.0):
        """Bool mask of the target neighbourhood (for SIFT keypoints)."""
        height, width = shape
        yy, xx = np.mgrid[0:height, 0:width] + 0.5
        if self.box is None:
            r = self.radius * scale
            return (xx - self.center[0]) ** 2 + (
                yy - self.center[1]
            ) ** 2 <= r * r
        half = 0.5 * (self.box[2:] - self.box[:2]) * scale
        return (np.abs(xx - self.center[0]) <= half[0]) & (
            np.abs(yy - self.center[1]) <= half[1]
        )

    def support_offsets(self, num_ring: int = 8) -> np.ndarray:
        """TAPNext++ query offsets relative to the center (first = center).

        Point targets: the point plus two rings (r/3 and 2r/3). Area targets:
        the center plus a 4x4 grid inside the box.
        """
        offsets = [np.zeros(2)]
        if self.box is None:
            for frac in (1 / 3, 2 / 3):
                angles = np.linspace(0, 2 * np.pi, num_ring, endpoint=False)
                if frac > 0.5:
                    angles = angles + np.pi / num_ring
                r = frac * self.radius
                offsets += [
                    np.array([np.cos(a), np.sin(a)]) * r for a in angles
                ]
        else:
            half = 0.5 * (self.box[2:] - self.box[:2])
            for u in np.linspace(-0.75, 0.75, 4):
                for v in np.linspace(-0.75, 0.75, 4):
                    offsets.append(np.array([u * half[0], v * half[1]]))
        return np.array(offsets)


def parse_target(text: str, radius: float = 24.0, name: str = "") -> Target:
    """ "x,y" -> point target; "x,y,w,h" -> area target.

    Continuous pixel coordinates: (0, 0) is the top-left corner of the
    image, pixel (i, j) covers [i, i + 1) x [j, j + 1).
    """
    values = [float(v) for v in text.replace(" ", "").split(",")]
    if len(values) == 2:
        return Target.point(values[0], values[1], radius=radius, name=name)
    if len(values) == 4:
        x, y, w, h = values
        return Target.area(x, y, x + w, y + h, name=name)
    raise ValueError(f"Expected 'x,y' or 'x,y,w,h', got {text!r}")
