"""Shared test helpers. Every test uses synthetic data only (generated videos, fake models)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_video(path: Path, n: int = 30, fps: float = 15.0, size=(64, 48)) -> Path:
    """Write a small synthetic MP4 (moving grey gradient, no real footage)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        img = np.full((h, w, 3), (i * 7) % 255, np.uint8)
        cv2.rectangle(img, (i % w, 5), (i % w + 5, 15), (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path
