"""Compact object masks: a grid x grid bitmap of the mask inside the object's box, base64-packed.

Storing a 32 x 32 bitmap per object (172 characters) keeps object_detections.jsonl small and lets
contact analysis be re-run without the segmentation model. It matters for objects whose box is a
poor fit: a long thin LED strip lying diagonally, a chain, or an L-shaped machine part, where most of
the box is background and a fingertip inside the box is not necessarily on the object.
"""
from __future__ import annotations

import base64
import math

import numpy as np


def mask_to_grid(mask: np.ndarray, bbox, n: int = 32) -> str:
    """Full-frame boolean mask + its box -> base64 of an n x n bitmap of the box region."""
    import cv2
    H, W = mask.shape[:2]
    x1, y1 = max(0, int(math.floor(bbox[0]))), max(0, int(math.floor(bbox[1])))
    x2, y2 = min(W, int(math.ceil(bbox[2]))), min(H, int(math.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return encode_grid(np.zeros((n, n), bool))
    crop = mask[y1:y2, x1:x2].astype(np.float32)
    small = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA) > 0.25    # thin parts survive
    return encode_grid(small)


def encode_grid(grid: np.ndarray) -> str:
    return base64.b64encode(np.packbits(grid.astype(bool).ravel())).decode("ascii")


def decode_grid(s: str, n: int = 32) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))[: n * n]
    return bits.reshape(n, n).astype(bool)


def point_near_mask(grid: np.ndarray, bbox, p, pad_px: float) -> bool:
    """Is point p (frame pixels) within pad_px of the mask? Approximated on the grid (cell-level)."""
    n = grid.shape[0]
    x1, y1, x2, y2 = bbox
    cw, ch = max((x2 - x1) / n, 1e-6), max((y2 - y1) / n, 1e-6)
    gx, gy = (p[0] - x1) / cw, (p[1] - y1) / ch
    rx, ry = pad_px / cw, pad_px / ch
    ys, xs = np.nonzero(grid)
    if not len(xs):
        return False
    # distance from p to each occupied cell (cell = [x, x+1) x [y, y+1)), in cell units per axis
    dx = np.maximum(np.maximum(xs - gx, gx - (xs + 1)), 0) / max(rx, 1e-6)
    dy = np.maximum(np.maximum(ys - gy, gy - (ys + 1)), 0) / max(ry, 1e-6)
    return bool(np.any(dx * dx + dy * dy <= 1.0)) if pad_px > 0 else bool(np.any((dx == 0) & (dy == 0)))
