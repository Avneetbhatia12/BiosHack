"""PS5 training-video generator: cut the key steps of a task out of a clip and caption them.

For each PS3 step (optionally only some verbs): a short title card ("Step 3 / 9 · fasten screw · left hand"), then
the step itself with the hands, the touched object and the step caption drawn on it. Also writes subtitles (.srt) and
a step list (.csv), so the same cut can be used with any player. Contains company footage: private use only.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .. import results as R
from ..review import ClipView, _fmt_t, hand_words

SKIP_VERBS = ("hold", "carry", "other")


def select_steps(actions: Sequence[dict], min_step_s: float = 1.0, skip_verbs: Iterable[str] = SKIP_VERBS,
                 max_steps: Optional[int] = None) -> List[dict]:
    skip = set(skip_verbs)
    steps = [a for a in sorted(actions, key=lambda x: x["start_s"])
             if a["verb"] not in skip and a["end_s"] - a["start_s"] >= min_step_s]
    return steps[:max_steps] if max_steps else steps


def _card(w: int, h: int, lines: List[str]) -> np.ndarray:
    img = np.full((h, w, 3), 24, np.uint8)
    s = max(0.6, h / 720.0)
    y = int(h * 0.42)
    for k, l in enumerate(lines):
        fs = (1.3 if k == 0 else 0.85) * s
        th = max(1, int((3 if k == 0 else 2) * s))
        (tw, tht), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.putText(img, l, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255) if k == 0 else (190, 190, 190), th, cv2.LINE_AA)
        y += int(tht * 2.0)
    return img


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def make_training_video(clip_path: str, out_root: str, out_mp4: str, min_step_s: float = 1.0, pad_s: float = 0.4,
                        card_s: float = 1.2, max_total_s: float = 120.0, width: int = 1280,
                        skip_verbs: Iterable[str] = SKIP_VERBS, layers=("ps1", "ps2", "ps3"), log=print) -> dict:
    from ..wilor.video_loader import VideoWriter
    view = ClipView(clip_path, out_root)
    steps = select_steps(view.actions, min_step_s, skip_verbs)
    if not steps:
        raise ValueError(f"{view.stem}: no PS3 steps to cut (run PS3 first, or lower min_step_s)")
    probe = view.compose(view.f_of(steps[0]["start_s"]), layers, width=width)
    W, H = probe.shape[1] // 2 * 2, probe.shape[0] // 2 * 2
    writer = VideoWriter(out_mp4, view.fps, (W, H))
    srt, rows, t_out, k_used = [], [], 0.0, 0
    for k, a in enumerate(steps, 1):
        seg_s = (a["end_s"] - a["start_s"]) + 2 * pad_s + card_s
        if t_out + seg_s > max_total_s:
            log(f"[training] stopping at step {k - 1}: the {max_total_s:.0f} s limit would be exceeded")
            break
        title = f"Step {k} / {len(steps)}: {a['verb']} {a['object']}"
        sub = f"{hand_words(a['hand'])}  |  {_fmt_t(a['start_s'])} - {_fmt_t(a['end_s'])} in {view.stem}"
        card = _card(W, H, [title, sub] + ([f"edges refined by PS2: {', '.join(a['refined'])}"] if a.get("refined") else []))
        for _ in range(int(round(card_s * view.fps))):
            writer.write(card)
        f0 = max(0, view.f_of(a["start_s"] - pad_s))
        f1 = min(view.n_frames - 1, view.f_of(a["end_s"] + pad_s))
        start_out = t_out + card_s
        for f in range(f0, f1 + 1):
            img = view.compose(f, layers, width=width)
            if img is None:
                break
            img = img[:H, :W]
            s = max(0.6, img.shape[0] / 720.0)
            cap = f"{k}. {a['verb']} {a['object']} ({a['hand']})"
            (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, max(1, int(2 * s)))
            y = img.shape[0] - int(24 * s)
            cv2.rectangle(img, (int(16 * s) - 8, y - th - 10), (int(16 * s) + tw + 8, y + 10), (0, 0, 0), -1)
            cv2.putText(img, cap, (int(16 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255),
                        max(1, int(2 * s)), cv2.LINE_AA)
            writer.write(img)
        t_out = start_out + (f1 - f0 + 1) / view.fps
        srt.append(f"{len(srt) + 1}\n{_srt_time(start_out - card_s)} --> {_srt_time(t_out)}\n{title} ({hand_words(a['hand'])})\n")
        rows.append({"step": k, "verb": a["verb"], "object": a["object"], "hand": a["hand"], "clip": view.stem,
                     "src_start_s": a["start_s"], "src_end_s": a["end_s"], "video_start_s": round(start_out, 2),
                     "video_end_s": round(t_out, 2), "refined": "; ".join(a.get("refined", []))})
        k_used = k
    path = writer.close()
    view.close()
    base = os.path.splitext(out_mp4)[0]
    with open(base + ".srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    R.write_csv(base + "_steps.csv", rows)
    log(f"[training] {k_used} step(s), {t_out:.0f} s -> {path}")
    return {"video": path, "srt": base + ".srt", "steps_csv": base + "_steps.csv", "steps": rows, "duration_s": round(t_out, 1)}
