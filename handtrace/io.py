"""Random-access frame reading for the local labelling tools (eval/label.py, eval/label_events.py).

The pipeline itself streams video with `handtrace.wilor.video_loader`. Nothing here writes or logs pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frame(path: str | Path, frame_index: int) -> np.ndarray | None:
    """One BGR frame by index, or None past the end."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


class VideoCursor:
    """Sequential reading with cheap random jumps, for interactive tools."""

    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = -1

    def read(self, frame_index: int) -> np.ndarray | None:
        if frame_index != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, img = self.cap.read()
        self.pos = frame_index if ok else self.pos
        return img if ok else None

    def close(self) -> None:
        self.cap.release()
