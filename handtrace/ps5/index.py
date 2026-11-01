"""One searchable event table for every clip, built from whatever PS1 / PS2 / PS3 / PS5-safety produced.

event = {clip, source, kind, t_start, t_end, duration_s, hand, verb, object, object_type, confidence, detail}
    source ps1  kind "hands_visible" (per side), "both_hands", "no_hands"
    source ps2  kind "contact"       (verb "touch", object + portable/fixed)
    source ps3  kind "action"        (verb from the fixed vocabulary)
    source ps5  kind "safety"        (hand near a hazard object), "quality" (blurry / dark stretch)
"""
from __future__ import annotations

import os
from typing import List, Sequence

from .. import results as R

EVENT_FIELDS = ["clip", "source", "kind", "t_start", "t_end", "duration_s", "hand", "verb", "object", "object_type",
                "confidence", "detail"]


def _ev(clip, source, kind, t0, t1, hand="", verb="", obj="", otype="", conf=None, detail=""):
    return {"clip": clip, "source": source, "kind": kind, "t_start": round(float(t0), 2), "t_end": round(float(t1), 2),
            "duration_s": round(float(t1) - float(t0), 2), "hand": hand, "verb": verb, "object": obj,
            "object_type": otype, "confidence": conf, "detail": detail}


def _segments(flags: List[bool], times: List[float], max_gap_s: float, min_len_s: float) -> List[tuple]:
    """Runs of True in a per-frame flag list -> [(t0, t1)], bridging gaps up to max_gap_s."""
    out, start, last = [], None, None
    for f, t in zip(flags, times):
        if f:
            if start is None:
                start = t
            elif last is not None and t - last > max_gap_s:
                out.append((start, last)); start = t
            last = t
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if b - a >= min_len_s]


def ps1_events(clip: str, hands: List[dict], max_gap_s: float = 0.3, min_len_s: float = 0.5) -> List[dict]:
    if not hands:
        return []
    times = [r["t"] for r in hands]
    sides = [{h.get("side") for h in r["hands"]} for r in hands]
    out = []
    for side in ("left", "right"):
        for a, b in _segments([side in s for s in sides], times, max_gap_s, min_len_s):
            out.append(_ev(clip, "ps1", "hands_visible", a, b, hand=side))
    for a, b in _segments([{"left", "right"} <= s for s in sides], times, max_gap_s, min_len_s):
        out.append(_ev(clip, "ps1", "both_hands", a, b, hand="both"))
    for a, b in _segments([not s for s in sides], times, max_gap_s, 1.0):
        out.append(_ev(clip, "ps1", "no_hands", a, b, detail="no hand found"))
    return out


def ps2_events(clip: str, contacts: List[dict]) -> List[dict]:
    out = []
    for c in contacts:
        try:
            conf = float(c.get("mean_contact_score") or 0) or None
        except ValueError:
            conf = None
        out.append(_ev(clip, "ps2", "contact", c["t_start"], c["t_end"], hand=c.get("hand", ""), verb="touch",
                       obj=c.get("object_name", ""), otype=c.get("object_type", ""), conf=conf,
                       detail=c.get("uncertainty", "")))
    return out


def ps3_events(clip: str, actions: List[dict]) -> List[dict]:
    return [_ev(clip, "ps3", "action", a["start_s"], a["end_s"], hand=a.get("hand", ""), verb=a.get("verb", ""),
                obj=a.get("object", ""), conf=a.get("confidence"), detail=";".join(a.get("refined", []) or []))
            for a in actions]


def ps5_events(clip: str, out_root: str, stem: str) -> List[dict]:
    p = R.paths(out_root, stem)
    out = []
    for s in R.read_csv(p["safety"]):
        out.append(_ev(clip, "ps5", "safety", s["t_start"], s["t_end"], hand=s.get("hand", ""), verb="near",
                       obj=s.get("hazard", ""), otype="fixed", detail=f"{s.get('level', '')} {s.get('note', '')}".strip()))
    q = R.read_json(p["quality"])
    for seg in q.get("bad_segments", []):
        out.append(_ev(clip, "ps5", "quality", seg["t_start"], seg["t_end"], detail=seg.get("reason", "")))
    return out


def clip_events(out_root: str, stem: str) -> List[dict]:
    ev = (ps1_events(stem, R.load_hands(out_root, stem)) + ps2_events(stem, R.load_contacts(out_root, stem))
          + ps3_events(stem, R.load_actions(out_root, stem)) + ps5_events(stem, out_root, stem))
    return sorted(ev, key=lambda e: (e["t_start"], e["source"]))


def build_index(out_root: str, stems: Sequence[str], write: bool = True) -> List[dict]:
    """Every clip's events -> OUT_ROOT/ps5_events_all.csv. Returns the rows."""
    rows = []
    for s in stems:
        rows += clip_events(out_root, s)
    if write:
        R.write_csv(os.path.join(out_root, "ps5_events_all.csv"), rows, EVENT_FIELDS)
    return rows


def known_objects(events: List[dict]) -> List[str]:
    return sorted({e["object"] for e in events if e["object"] and e["object"] != "unknown"})
