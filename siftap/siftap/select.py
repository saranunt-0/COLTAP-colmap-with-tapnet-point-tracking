# SPDX-License-Identifier: BSD-3-Clause
"""Interactive target selection on the first frame (needs GUI OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from .targets import Target

HELP = "click: point | drag: area | u: undo | enter/space: start | esc: cancel"


def select_targets(image_rgb: np.ndarray, radius: float = 24.0) -> list[Target]:
    """Let the user click points / drag boxes; returns the targets.

    Raises RuntimeError when OpenCV has no GUI support (the headless wheel);
    install ``opencv-python`` or pass --point / --box instead.
    """
    window = "SIFTAP: select targets"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        raise RuntimeError(
            "Interactive selection needs OpenCV with GUI support: "
            "pip uninstall opencv-python-headless && pip install "
            "opencv-python. Or use --point x,y / --box x,y,w,h."
        ) from exc
    selection = Selection(radius)

    def on_mouse(event, x, y, flags, _):
        selection.mouse(event, x, y)

    cv2.setMouseCallback(window, on_mouse)
    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    while True:
        view = base.copy()
        for t in selection.targets:
            if t.is_area:
                x0, y0, x1, y1 = (t.box - 0.5).astype(int)
                cv2.rectangle(view, (x0, y0), (x1, y1), (100, 230, 70), 2)
            else:
                c = tuple((t.center - 0.5).astype(int))
                cv2.circle(view, c, 4, (100, 230, 70), -1)
                cv2.circle(view, c, int(t.radius), (100, 230, 70), 1)
            cv2.putText(
                view,
                t.name,
                tuple((t.center - 0.5 + [6, -6]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 230, 70),
                1,
            )
        if selection.start is not None:
            cv2.rectangle(
                view, selection.start, selection.current, (0, 200, 255), 1
            )
        cv2.putText(
            view,
            HELP,
            (8, view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.imshow(window, view)
        if selection.key(cv2.waitKey(20) & 0xFF):
            break
    cv2.destroyWindow(window)
    return selection.targets


class Selection:
    """Click / drag / key logic of the selection window (GUI-free)."""

    def __init__(self, radius: float = 24.0, min_drag: int = 4):
        self.radius = radius
        self.min_drag = min_drag
        self.targets: list[Target] = []
        self.start = None
        self.current = None

    def mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            (x0, y0), (x1, y1) = self.start, (x, y)
            name = f"T{len(self.targets) + 1}"
            # Pixel (i, j) has its centre at (i + 0.5, j + 0.5) (COLMAP).
            if abs(x1 - x0) < self.min_drag and abs(y1 - y0) < self.min_drag:
                target = Target.point(x0 + 0.5, y0 + 0.5, self.radius, name)
            else:
                target = Target.area(
                    x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5, name
                )
            self.targets.append(target)
            self.start = self.current = None

    def key(self, key: int) -> bool:
        """Handle a key; True when the selection is finished."""
        if key in (13, 32) and self.targets:
            return True
        if key == 27:
            self.targets = []
            return True
        if key == ord("u") and self.targets:
            self.targets.pop()
        return False
