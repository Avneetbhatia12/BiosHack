"""Merge duplicate / overlapping descriptions of the same action (overlapping windows, split activities)."""
from __future__ import annotations

from typing import List

from .config import PS3Config


def _tokens(s: str) -> set:
    return {t for t in s.replace("-", " ").split() if t}


def same_object(a: str, b: str) -> bool:
    if a == b or "unknown" in (a, b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.5 or ta <= tb or tb <= ta     # whole words: "screw" != "screwdriver"


def _hand(a: str, b: str) -> str:
    if a == b:
        return a
    if "unknown" in (a, b):
        return b if a == "unknown" else a
    return "both"


def can_merge(a: dict, b: dict, gap_s: float) -> bool:
    if a["verb"] != b["verb"] or not same_object(a["object"], b["object"]):
        return False
    if b["start_s"] - a["end_s"] > gap_s:
        return False
    # different hands acting separately at the same time are two actions, not one
    if a["hand"] != b["hand"] and "unknown" not in (a["hand"], b["hand"]) and "both" not in (a["hand"], b["hand"]):
        overlap = min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"])
        if overlap > 0.5:
            return False
    return True


def merge_actions(actions: List[dict], cfg: PS3Config) -> List[dict]:
    items = sorted((dict(a) for a in actions), key=lambda a: (a["start_s"], a["end_s"]))
    merged: List[dict] = []
    for act in items:
        target = None
        for prev in reversed(merged[-6:]):                      # compare with the last few only
            if can_merge(prev, act, cfg.merge_gap_s):
                target = prev
                break
        if target is None:
            merged.append(act)
            continue
        better = act if act["confidence"] > target["confidence"] else target
        target["start_s"] = min(target["start_s"], act["start_s"])
        target["end_s"] = max(target["end_s"], act["end_s"])
        target["object"] = better["object"] if better["object"] != "unknown" else (
            target["object"] if target["object"] != "unknown" else act["object"])
        target["hand"] = _hand(target["hand"], act["hand"])
        target["confidence"] = round(max(target["confidence"], act["confidence"]), 3)
        target["windows"] = sorted(set(target.get("windows", [])) | set(act.get("windows", [])))
    merged.sort(key=lambda a: (a["start_s"], a["end_s"]))
    return merged
