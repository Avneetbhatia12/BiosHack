"""PS2 -> PS3 link: make action start / end times precise with the grab and release moments from PS2.

Gemini sees the video at a few frames per second and rounds its times; PS2 sees every frame and knows the exact
frame a hand's fingertips closed on an object (grab = contact t_start) and opened again (release = contact t_end).
An action edge is moved onto such a moment only when ALL of these hold:
  * the moment is within `snap_tol_s` of the edge the model gave,
  * it belongs to a compatible hand (same side; "both"/"unknown" accept either),
  * the PS2 object is compatible with the action's object (same words, or one side is "unknown"),
  * the action stays at least `min_action_s` long.
Which moment an edge may snap to depends on the verb:
  pick   END   -> grab     (a pick is the reach that ends when the fingers close)
  place  END   -> release  (a place ends when the fingers open)
  others START -> grab, END -> release   (a fasten that starts when the screwdriver is picked up, ...)
Unknown hands / objects are filled from the PS2 contact that overlaps the action most. Every change is recorded in
`refined` on the action, so the before/after can be shown and the change reverted.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from .merge import same_object

GRAB_EDGE = {"pick": "end"}          # verb -> which edge snaps to a grab (default: start)
RELEASE_EDGE = {"pick": None}        # verb -> which edge snaps to a release (default: end); None = no release snap


def load_contacts(path: str) -> List[dict]:
    """PS2 contacts.csv -> list of {t_start, t_end, hand, object_name, object_type}. [] if missing."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append({"t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
                            "hand": (r.get("hand") or "unknown").lower(),
                            "object_name": (r.get("object_name") or "unknown").replace("_", " ").lower(),
                            "object_type": (r.get("object_type") or "unknown").lower()})
            except (KeyError, ValueError):
                continue
    return out


def _hand_ok(action_hand: str, contact_hand: str) -> bool:
    return action_hand in ("both", "unknown") or contact_hand in ("unknown",) or action_hand == contact_hand


def _obj_ok(action_obj: str, contact_obj: str) -> bool:
    return same_object(action_obj.replace("_", " "), contact_obj)


def _nearest(t: float, moments: List[Tuple[float, dict]], tol: float):
    best = None
    for m, c in moments:
        d = abs(m - t)
        if d <= tol and (best is None or d < best[0]):
            best = (d, m, c)
    return best


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def refine_actions(actions: List[dict], contacts: List[dict], cfg) -> Tuple[List[dict], Dict]:
    tol = float(getattr(cfg, "snap_tol_s", 1.5))
    min_len = float(getattr(cfg, "min_action_s", 0.4))
    out, snapped, hands_filled, objects_filled = [], 0, 0, 0
    for a in actions:
        a = dict(a)
        notes = []
        compat = [c for c in contacts if _hand_ok(a["hand"], c["hand"]) and _obj_ok(a["object"], c["object_name"])]
        # 1. fill unknown hand / object from the PS2 contact overlapping the action most
        ov = sorted(((_overlap(a["start_s"], a["end_s"], c["t_start"], c["t_end"]), c) for c in compat),
                    key=lambda x: -x[0])
        if ov and ov[0][0] > 0:
            c = ov[0][1]
            if a["hand"] == "unknown" and c["hand"] in ("left", "right"):
                a["hand"] = c["hand"]; hands_filled += 1; notes.append(f"hand<-ps2:{c['hand']}")
            if a["object"] == "unknown" and c["object_name"] != "unknown":
                a["object"] = c["object_name"]; objects_filled += 1; notes.append(f"object<-ps2:{c['object_name']}")
            compat = [c for c in contacts if _hand_ok(a["hand"], c["hand"]) and _obj_ok(a["object"], c["object_name"])]
        # 2. snap edges
        grabs = [(c["t_start"], c) for c in compat]
        releases = [(c["t_end"], c) for c in compat]
        new_s, new_e = a["start_s"], a["end_s"]
        g_edge = GRAB_EDGE.get(a["verb"], "start")
        r_edge = RELEASE_EDGE.get(a["verb"], "end")
        g = _nearest(a["end_s"] if g_edge == "end" else a["start_s"], grabs, tol)
        if g:
            if g_edge == "end":
                new_e = g[1]
            else:
                new_s = g[1]
        if r_edge:
            r = _nearest(a["end_s"], releases, tol)
            if r and not (g and g_edge == "end"):
                new_e = r[1]
        if new_e - new_s >= min_len and (new_s, new_e) != (a["start_s"], a["end_s"]):
            if new_s != a["start_s"]:
                notes.append(f"start {a['start_s']:.2f}->{new_s:.2f} (grab)")
                snapped += 1
            if new_e != a["end_s"]:
                notes.append(f"end {a['end_s']:.2f}->{new_e:.2f} ({'grab' if g and g_edge == 'end' else 'release'})")
                snapped += 1
            a["start_s"], a["end_s"] = round(new_s, 2), round(new_e, 2)
        if notes:
            a["refined"] = notes
        out.append(a)
    out.sort(key=lambda x: (x["start_s"], x["end_s"]))
    # 3. PS2 contacts with no action at all: listed so a reviewer can see what the model may have missed
    unexplained = [c for c in contacts if c["object_type"] == "portable" and c["t_end"] - c["t_start"] >= 1.0
                   and not any(_overlap(c["t_start"], c["t_end"], x["start_s"], x["end_s"]) > 0 for x in out)]
    rep = {"ps2_contacts": len(contacts), "edges_snapped": snapped, "hands_filled": hands_filled,
           "objects_filled": objects_filled, "snap_tol_s": tol,
           "portable_contacts_without_action": [
               {"t_start": c["t_start"], "t_end": c["t_end"], "hand": c["hand"], "object": c["object_name"]}
               for c in unexplained[:50]]}
    return out, rep
