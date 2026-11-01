"""PS2 drawing: PS1 hand skeletons + object boxes + contact state (OpenCV only).

Styles: portable objects orange, fixed objects blue (thin), unknown grey (thin). An object in
active contact gets a thick box, a filled corner tag and a line from the fingertips to it. Each
hand gets one short label block: "RIGHT ID 4 / CONTACT: SCREWDRIVER / PORTABLE  3/5".
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from handtrace.wilor.visualization import draw_hand

TYPE_COLOUR = {"portable": (0, 140, 255), "fixed": (230, 160, 0), "unknown": (160, 160, 160)}
CONTACT_COLOUR = (60, 220, 60)


def _scale(h: int) -> float:
    return max(0.6, h / 1080.0)


def _text_block(img, lines: List[str], org, colour, s: float):
    fs, th = 0.55 * s, max(1, int(round(1.5 * s)))
    sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for l in lines]
    w = max(sz[0] for sz in sizes) + 8
    lh = max(sz[1] for sz in sizes) + 6
    x, y = int(org[0]), int(org[1])
    y = max(lh * len(lines) + 2, y)
    x = min(max(0, x), max(0, img.shape[1] - w))
    cv2.rectangle(img, (x, y - lh * len(lines) - 2), (x + w, y + 2), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y - lh * len(lines) - 2), (x + w, y + 2), colour, 1)
    for k, l in enumerate(lines):
        cv2.putText(img, l, (x + 4, y - lh * (len(lines) - 1 - k) - 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)


def draw_ps2(img: np.ndarray, hand_rec: Dict, obj_rec: Dict, contact_rec: Dict, hud: str = "") -> np.ndarray:
    s = _scale(img.shape[0])
    active = {c["object_id"] for h in contact_rec["hands"] for c in h.get("contacts", [])}
    for o in obj_rec["objects"]:
        col = TYPE_COLOUR.get(o["object_type"], TYPE_COLOUR["unknown"])
        on = o["object_id"] in active
        x1, y1, x2, y2 = [int(round(v)) for v in o["bbox"]]
        thick = max(1, int((4 if on else (2 if o["object_type"] == "portable" else 1)) * s))
        cv2.rectangle(img, (x1, y1), (x2, y2), CONTACT_COLOUR if on else col, thick, cv2.LINE_AA)
        tag = f"{o['object_id']} {o['object_name']} {o['confidence']:.2f}"
        fs = 0.45 * s
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (x1, max(0, y1 - tht - 6)), (x1 + tw + 6, max(tht + 6, y1)), col, -1 if on else 1)
        cv2.putText(img, tag, (x1 + 3, max(tht + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
    contacts = {h["hand_id"]: h for h in contact_rec["hands"]}
    objs = {o["object_id"]: o for o in obj_rec["objects"]}
    for hand in hand_rec["hands"]:
        draw_hand(img, hand)
        c = contacts.get(hand["id"], {})
        x1, y1 = hand["bbox"][0], hand["bbox"][1]
        lines = [f"{str(hand['side']).upper()} HAND ID {hand['id']}"]
        col = (200, 200, 200)
        if c.get("contacts"):
            first = c["contacts"][0]
            n = first.get("fingertip_count")
            lines.append(f"CONTACT: {first['object_name'].upper()}")
            lines.append(f"TYPE: {first['object_type'].upper()}" + (f"  TIPS {n}/5" if n else "  (grace)"))
            col = CONTACT_COLOUR
            o = objs.get(first["object_id"])
            if o is not None:
                kp = np.asarray(hand["kp2d"], float)
                tip = kp[[4, 8, 12, 16, 20]].mean(0)
                ox, oy = (o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2
                cv2.line(img, (int(tip[0]), int(tip[1])), (int(ox), int(oy)), CONTACT_COLOUR,
                         max(1, int(2 * s)), cv2.LINE_AA)
        elif c.get("state") == "candidate":
            lines.append("candidate (unconfirmed)")
        _text_block(img, lines, (x1, y1 - 4 * s - 30 * s), col, s)
    text = f"frame {hand_rec['frame']}  t={hand_rec['t']:.2f}s  objects={len(obj_rec['objects'])}"
    if hud:
        text += "  " + hud
    org = (int(12 * s), int(img.shape[0] - 16 * s))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), max(2, int(4 * s)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return img
