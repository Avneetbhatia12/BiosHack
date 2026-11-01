"""Rendering helpers (OpenCV only): demo video segments and private review contact sheets.

Everything produced here contains the company's footage. Keep it in the working folder and upload it
only to the event's Drive folder; never commit or post it.

    render_hands(...)       PS1 skeletons on a time range, optionally raw vs final side by side
    render_ps2_segment(...) PS1 hands + object boxes + contact state on a time range
    review_sheet(...)       a grid of frames with hands drawn, for eyeballing sides / IDs
    ps2_sheet(...)          a grid of contact-positive and contact-negative frames
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .wilor.video_loader import VideoReader, VideoWriter, probe
from .wilor.visualization import contact_sheet, draw_record


def _load(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _range(fps: float, start_s: float, end_s: Optional[float], total: int) -> tuple:
    start = max(0, int(round(start_s * fps)))
    stop = total if end_s is None else min(total, int(round(end_s * fps)))
    return start, max(0, stop - start)


def _panel_size(w: int, h: int, out_w: Optional[int], panels: int) -> tuple:
    pw = ((out_w or w * panels) // panels) // 2 * 2          # even sizes: required by yuv420p
    ph = int(round(h * pw / w)) // 2 * 2
    return pw, ph


def render_hands(video: str, jsonl: str, out: str, start_s: float = 0.0, end_s: Optional[float] = None,
                 compare_jsonl: Optional[str] = None, width: Optional[int] = None,
                 labels: Sequence[str] = ("before", "after"), crf: int = 23, log=print) -> str:
    """Draw `jsonl` (hackathon format) on `video` between start_s and end_s. With `compare_jsonl` the left
    panel shows that file (e.g. raw output) and the right panel `jsonl` (e.g. final)."""
    info = probe(video)
    start, n = _range(info.fps, start_s, end_s, info.frame_count)
    by = {r["frame"]: r for r in _load(jsonl)}
    cmp_by = {r["frame"]: r for r in _load(compare_jsonl)} if compare_jsonl else None
    panels = 2 if cmp_by is not None else 1
    pw, ph = _panel_size(info.width, info.height, width, panels)
    writer = VideoWriter(out, info.fps, (pw * panels, ph), crf)
    sources = [(cmp_by, labels[0]), (by, labels[1])] if cmp_by is not None else [(by, None)]
    count = 0
    for fidx, t, img in VideoReader(video, start=start, max_frames=n):
        tiles = []
        for recs, title in sources:
            rec = {**recs.get(fidx, {"hands": []}), "frame": fidx, "t": t}
            tile = draw_record(img.copy() if panels > 1 else img, rec, hud=title)
            tiles.append(cv2.resize(tile, (pw, ph), interpolation=cv2.INTER_AREA))
        writer.write(np.concatenate(tiles, axis=1) if panels > 1 else tiles[0])
        count += 1
    path = writer.close()
    log(f"[render] {count} frames -> {path}")
    return path


def render_ps2_segment(video: str, hands_jsonl: str, objects_jsonl: str, contacts_jsonl: str, out: str,
                       start_s: float = 0.0, end_s: Optional[float] = None, width: Optional[int] = None,
                       crf: int = 23, log=print) -> str:
    """PS1 hands + PS2 objects and contact state on a time range (PS2 files cover the whole clip)."""
    from .ps2.visualization import draw_ps2
    info = probe(video)
    start, n = _range(info.fps, start_s, end_s, info.frame_count)
    hands = {r["frame"]: r for r in _load(hands_jsonl)}
    objs = {r["frame"]: r for r in _load(objects_jsonl)}
    cons = {r["frame"]: r for r in _load(contacts_jsonl)}
    pw, ph = _panel_size(info.width, info.height, width, 1)
    writer = VideoWriter(out, info.fps, (pw, ph), crf)
    count = 0
    for fidx, t, img in VideoReader(video, start=start, max_frames=n):
        if fidx in hands and fidx in objs and fidx in cons:
            img = draw_ps2(img, hands[fidx], objs[fidx], cons[fidx])
        writer.write(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA) if (pw, ph) != (info.width, info.height)
                     else img)
        count += 1
    path = writer.close()
    log(f"[render] {count} frames -> {path}")
    return path


def _grab(video: str, idxs: Sequence[int]) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    out = {}
    try:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if ok:
                out[int(i)] = img
    finally:
        cap.release()
    return out


def _spread(items: list, n: int) -> list:
    return items[:: max(1, len(items) // n)][:n] if items else []


def review_sheet(video: str, records: List[Dict], n: int = 6, min_hands: int = 2, hud: str = "",
                 width: int = 640) -> Optional[np.ndarray]:
    """Grid of up to n frames with hands drawn. Prefers frames with at least `min_hands` hands (left/right
    check), then any frame with a hand, then any frame. Returns None if nothing could be read."""
    pool = [r for r in records if len(r["hands"]) >= min_hands] or [r for r in records if r["hands"]] or records
    chosen = _spread(pool, n)
    imgs = _grab(video, [r["frame"] for r in chosen])
    tiles = [draw_record(imgs[r["frame"]], r, hud=hud) for r in chosen if r["frame"] in imgs]
    return contact_sheet(tiles, width=width) if tiles else None


def ps2_sheet(video: str, hands: List[Dict], objects: List[Dict], contacts: List[Dict], n_pos: int = 6,
              n_neg: int = 3, width: int = 640) -> Optional[np.ndarray]:
    """Grid of contact-positive (POS) and contact-negative-with-hands (NEG) frames with PS2 drawn."""
    from .ps2.visualization import draw_ps2
    pos = [i for i, c in enumerate(contacts) if any(h["state"].startswith("touching") for h in c["hands"])]
    pos_set = set(pos)
    neg = [i for i, c in enumerate(contacts) if c["hands"] and i not in pos_set]
    picks = [(i, "POS") for i in _spread(pos, n_pos)] + [(i, "NEG") for i in _spread(neg, n_neg)]
    imgs = _grab(video, [hands[i]["frame"] for i, _ in picks])
    tiles = [draw_ps2(imgs[hands[i]["frame"]], hands[i], objects[i], contacts[i], hud=tag)
             for i, tag in picks if hands[i]["frame"] in imgs]
    return contact_sheet(tiles, width=width) if tiles else None
