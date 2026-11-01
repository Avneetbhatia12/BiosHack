"""Annotated-frame drawing (OpenCV only). Left and right hands use distinct colour families;
interpolated (gap-filled) hands are drawn thinner with an 'interp' tag so judges can tell them
apart from real detections."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

# 21-joint skeleton (OpenPose hand topology: 0 wrist, tips 4/8/12/16/20)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_OF_EDGE = [0, 0, 0, 0, 1, 1, 1, 1, 5, 2, 2, 2, 5, 3, 3, 3, 5, 5, 4, 4, 4]
FINGER_OF_JOINT = [5, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

# BGR. Left = blue/green family, right = orange/red family. Finger shades vary within a family.
SIDE_BASE = {"left": (255, 160, 40), "right": (40, 120, 255)}
FINGER_SHADES = {
    "left": [(255, 120, 0), (255, 200, 0), (200, 255, 0), (120, 255, 60), (255, 90, 160), (255, 160, 40)],
    "right": [(0, 90, 255), (0, 170, 255), (0, 220, 255), (80, 80, 255), (160, 60, 255), (40, 120, 255)],
}


def _scale(img_h: int) -> float:
    return max(0.6, img_h / 1080.0)


def draw_hand(img: np.ndarray, hand: Dict) -> None:
    side = hand["side"]
    s = _scale(img.shape[0])
    interp = bool(hand.get("interpolated", False))
    thick = max(1, int(round((1 if interp else 3) * s)))
    base = SIDE_BASE.get(side, (200, 200, 200))
    x1, y1, x2, y2 = [int(round(v)) for v in hand["bbox"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), base, max(1, int(2 * s)), cv2.LINE_AA)
    kp = np.asarray(hand["kp2d"], np.float32)
    if kp.shape == (21, 2) and np.isfinite(kp).all():
        pts = np.round(kp).astype(int)
        shades = FINGER_SHADES[side] if side in FINGER_SHADES else [base] * 6
        for (a, b), f in zip(HAND_EDGES, FINGER_OF_EDGE):
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), shades[f], thick, cv2.LINE_AA)
        r = max(2, int(round((2 if interp else 4) * s)))
        for j, p in enumerate(pts):
            cv2.circle(img, tuple(p), r, shades[FINGER_OF_JOINT[j]], -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), r, (255, 255, 255), 1, cv2.LINE_AA)
    label = f"{'L' if side == 'left' else 'R'} id{hand['id']} {hand['score']:.2f}" + (" interp" if interp else "")
    fs = 0.7 * s
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(2 * s)))
    ty = max(th + 4, y1 - 6)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + bl), base, -1)
    cv2.putText(img, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                max(1, int(2 * s)), cv2.LINE_AA)


def draw_record(img: np.ndarray, record: Dict, hud: Optional[str] = None) -> np.ndarray:
    for hand in record["hands"]:
        draw_hand(img, hand)
    s = _scale(img.shape[0])
    text = f"frame {record['frame']}  t={record['t']:.2f}s  hands={len(record['hands'])}"
    if hud:
        text += "  " + hud
    cv2.putText(img, text, (int(12 * s), int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                (0, 0, 0), max(2, int(4 * s)), cv2.LINE_AA)
    cv2.putText(img, text, (int(12 * s), int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return img


def contact_sheet(frames: List[np.ndarray], cols: int = 3, width: int = 640) -> np.ndarray:
    """Small grid for in-notebook inspection (never publish: contains private footage)."""
    tiles = [cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1]))) for f in frames]
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, 0)
