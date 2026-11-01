"""Temporal contact state machine (per hand x object ID): turns per-frame contact candidates into intervals.

none -> candidate (consecutive candidate frames) -> touching once `min_contact_frames` (and `min_contact_s`) are
reached; the event starts at the FIRST candidate frame. It survives `release_grace_frames` frames without support
and ends at the LAST supported frame. Interpolated hands cannot start an event unless `interpolated_can_start`.
Different object IDs are never merged; without `allow_multiple`, opening a new event closes the hand's previous
one unless the pair is listed in `co_pairs` (e.g. a cloth held while wiping a slab).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _Key:
    streak: list = field(default_factory=list)       # [(frame, t, cand)] before opening
    event: Optional[dict] = None
    missed: int = 0


class ContactStateMachine:
    def __init__(self, min_contact_frames: int = 3, min_contact_s: Optional[float] = None,
                 release_grace_frames: int = 3, interpolated_can_start: bool = False,
                 allow_multiple: bool = False, co_pairs: Optional[set] = None):
        self.min_frames, self.min_s = min_contact_frames, min_contact_s
        self.grace, self.interp_start, self.multi = release_grace_frames, interpolated_can_start, allow_multiple
        self.co_pairs = co_pairs or set()
        self.keys: Dict[tuple, _Key] = {}
        self.events: List[dict] = []
        self.rejected_candidates = 0                     # candidate runs that never became an event

    def active_objects(self, hand_id) -> set:
        return {k[1] for k, s in self.keys.items() if k[0] == hand_id and s.event}

    def _open(self, key, s: _Key):
        hand_id, obj_id = key
        if not self.multi:
            name = s.streak[-1][2]["object_name"]
            for k2, s2 in self.keys.items():
                if k2[0] == hand_id and k2 != key and s2.event:
                    other = s2.event["names"].most_common(1)[0][0]
                    if frozenset((name, other)) not in self.co_pairs:
                        self._close(k2, s2, "switched_object")
        f0, t0, c0 = s.streak[0]
        s.event = {"hand_id": hand_id, "object_id": obj_id, "start_frame": f0, "t_start": t0,
                   "end_frame": f0, "t_end": t0, "supported_frames": 0, "interpolated_frames": 0,
                   "carried_frames": 0, "names": Counter(), "types": Counter(), "sides": Counter(),
                   "scores": [], "tips": [], "confs": [], "grasps": Counter(), "co_contact": False}
        for f, t, c in s.streak:
            self._support(s.event, f, t, c)
        s.streak = []

    @staticmethod
    def _support(ev: dict, frame: int, t: float, c: dict):
        ev["end_frame"], ev["t_end"] = frame, t
        ev["supported_frames"] += 1
        ev["interpolated_frames"] += bool(c.get("interpolated"))
        ev["carried_frames"] += bool(c.get("carried"))
        ev["names"][c["object_name"]] += 1
        ev["types"][c["object_type"]] += 1
        ev["sides"][c.get("side", "?")] += 1
        ev["scores"].append(c["score"]); ev["tips"].append(c["fingertip_count"]); ev["confs"].append(c["object_conf"])
        ev["grasps"][c.get("grasp", "partial")] += 1
        ev["co_contact"] = ev["co_contact"] or bool(c.get("co_contact_with"))

    def _close(self, key, s: _Key, reason: str):
        if s.event:
            s.event["end_reason"] = reason
            self.events.append(s.event)
        s.event, s.missed = None, 0

    def step(self, frame: int, t: float, selected: Dict[object, List[dict]]):
        """selected: hand_id -> chosen candidates this frame (each with side, interpolated)."""
        seen = set()
        for hand_id, cands in selected.items():
            for c in cands:
                key = (hand_id, c["object_id"])
                seen.add(key)
                s = self.keys.setdefault(key, _Key())
                if s.event:
                    self._support(s.event, frame, t, c)
                    s.missed = 0
                    continue
                if c.get("interpolated") and not self.interp_start:
                    continue                             # neutral: neither starts nor breaks a streak
                s.streak.append((frame, t, c))
                long_enough = len(s.streak) >= self.min_frames
                if long_enough and self.min_s is not None:
                    long_enough = (t - s.streak[0][1]) >= self.min_s
                if long_enough:
                    self._open(key, s)
        for key, s in list(self.keys.items()):
            if key in seen:
                continue
            if s.event:
                s.missed += 1
                if s.missed > self.grace:
                    self._close(key, s, "released")
                    del self.keys[key]
            else:
                if s.streak:
                    self.rejected_candidates += 1
                del self.keys[key]
        # interpolated-only keys with an empty streak and no event are dropped too
        for key in [k for k, s in self.keys.items() if not s.event and not s.streak and k not in seen]:
            del self.keys[key]

    def finish(self) -> List[dict]:
        for key, s in list(self.keys.items()):
            if s.event:
                self._close(key, s, "clip_end")
            elif s.streak:
                self.rejected_candidates += 1
        self.keys.clear()
        self.events.sort(key=lambda e: (e["start_frame"], str(e["hand_id"]), e["object_id"]))
        return self.events
