"""Hand-object contact from PS1 hands + tracked object boxes (pure: records in, records out).

Per frame and hand (the hackathon baseline rule):
  pad = pad_frac * max(hand box width, height); expand every object box by pad; a contact
  CANDIDATE needs >= fingertip_threshold of the fingertips [4, 8, 12, 16, 20] (kp2d, pixels) inside
  the expanded box. Candidates rank by object type (portable > unknown > fixed, `type_priority`), then
  fingertip count, then the smallest fingertip-to-box distance, then object confidence. The object the hand touched in the previous frame is kept if it is within
  `sticky_margin` fingertips of the best. A second simultaneous contact (allow_multiple) needs
  `multi_min_fingertips` fingertips not already used by the first.

Over time (state machine per hand x object ID):
  none -> candidate (consecutive candidate frames) -> touching_<portable|fixed|unknown> once
  `min_contact_frames` (and `min_contact_s`, if set) are reached; the event starts at the FIRST
  candidate frame. A touching event survives up to `release_grace_frames` frames without support and
  ends at the LAST supported frame. Interpolated PS1 hands cannot start an event (unless
  interpolated_can_start) but can continue one; that is flagged as uncertainty. Different object IDs
  are never merged. With allow_multiple=False, opening a new event closes the hand's previous one.

These are 2D box-proximity estimates of touching, not verified physical grasps.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .state_machine import ContactStateMachine  # noqa: F401  (re-exported)

TIPS = (4, 8, 12, 16, 20)
TIP_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
START_EVENT = {"portable": "grab", "fixed": "touch_start", "unknown": "contact_start"}
END_EVENT = {"portable": "release", "fixed": "touch_end", "unknown": "contact_end"}


def point_box_distance(p, box) -> float:
    """0 inside the box, else Euclidean distance to it."""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float(np.hypot(dx, dy))


def grasp_type(tips: List[int]) -> str:
    """Coarse grasp label from which fingertips touch the object (dataset card, section 5.2):
    pinch = thumb + index only; power = thumb + >= 3 fingers; hook = fingers without the thumb."""
    s = set(tips)
    if s == {4, 8}:
        return "pinch"
    if 4 in s and len(s) >= 4:
        return "power"
    if 4 not in s and len(s) >= 2:
        return "hook"
    return "partial"


def frame_candidates(hand: dict, objects: List[dict], fingertip_threshold: int = 2,
                     pad_frac: float = 0.15, min_object_conf: float = 0.0,
                     type_priority: Optional[tuple] = ("portable", "unknown", "fixed"),
                     use_masks: bool = True, mask_n: int = 32) -> List[dict]:
    """All objects with >= fingertip_threshold fingertips inside the padded box, best first.

    With `type_priority`, a qualifying portable object outranks a fixed one even with fewer fingertips:
    a hand holding a tool above a table has all five tips inside the large table box, and ranking by
    count alone would report "touching table" and never the grab. None = rank by fingertip count only.
    If an object carries a mask (`mask_grid`, sam3 backend) and use_masks, a fingertip must also be
    within `pad` of the mask itself, not just of its box."""
    from .masks import decode_grid, point_near_mask
    kp = hand.get("kp2d")
    if not kp or len(kp) < 21:
        return []
    x1, y1, x2, y2 = hand["bbox"]
    pad = pad_frac * max(x2 - x1, y2 - y1)
    out = []
    for o in objects:
        if o.get("confidence", 1.0) < min_object_conf:
            continue
        b = o["bbox"]
        tips = [i for i in TIPS if b[0] - pad <= kp[i][0] <= b[2] + pad and b[1] - pad <= kp[i][1] <= b[3] + pad]
        masked = bool(use_masks and o.get("mask_grid"))
        if masked and len(tips) >= fingertip_threshold:
            grid = decode_grid(o["mask_grid"], mask_n)
            tips = [i for i in tips if point_near_mask(grid, b, kp[i], pad)]
        if len(tips) < fingertip_threshold:
            continue
        dists = [point_box_distance(kp[i], b) for i in tips]
        closeness = 1.0 - min(1.0, float(np.mean(dists)) / max(pad, 1e-6))
        out.append({"object_id": o["object_id"], "object_name": o["object_name"],
                    "object_type": o["object_type"], "object_conf": float(o.get("confidence", 1.0)),
                    "tips": tips, "fingertip_count": len(tips), "min_tip_dist_px": round(min(dists), 1),
                    "score": round(0.5 * len(tips) / 5 + 0.3 * float(o.get("confidence", 1.0)) + 0.2 * closeness, 4),
                    "carried": bool(o.get("carried", False)), "grasp": grasp_type(tips), "mask_checked": masked})
    rank = {t: i for i, t in enumerate(type_priority or ())}
    out.sort(key=lambda c: (rank.get(c["object_type"], len(rank)) if type_priority else 0,
                            -c["fingertip_count"], c["min_tip_dist_px"], -c["object_conf"]))
    return out


def select_contacts(cands: List[dict], prev_ids: set, sticky_margin: int = 1,
                    allow_multiple: bool = False, multi_min_fingertips: int = 2,
                    co_pairs: Optional[set] = None) -> List[dict]:
    """Primary contact, plus (a) one co-contact partner listed in `co_pairs` (e.g. cloth held while
    wiping a slab: the fingertips are shared, so the disjoint-fingertip rule cannot apply), and (b)
    with allow_multiple, contacts using >= multi_min_fingertips fingertips not used before."""
    if not cands:
        return []
    best = cands[0]
    primary = next((c for c in cands if c["object_id"] in prev_ids and c["object_type"] == best["object_type"]
                    and c["fingertip_count"] >= best["fingertip_count"] - sticky_margin), best)
    chosen, used = [primary], set(primary["tips"])
    if co_pairs:
        partner = next((c for c in cands if c is not primary
                        and frozenset((primary["object_name"], c["object_name"])) in co_pairs), None)
        if partner is not None:
            chosen.append(dict(partner, co_contact_with=primary["object_id"]))
            used |= set(partner["tips"])
    if allow_multiple:
        for c in cands:
            if c is primary or any(c["object_id"] == x["object_id"] for x in chosen):
                continue
            free = set(c["tips"]) - used
            if len(free) >= multi_min_fingertips:
                chosen.append(c)
                used |= set(c["tips"])
    return chosen


def analyze_contacts(hand_records: List[dict], object_records: List[dict], cfg,
                     co_pairs: Optional[list] = None) -> dict:
    """hand_records: PS1 jsonl records; object_records: {"frame", "t", "objects": [...]} with object_id.
    Both must cover the same frames in the same order (checked by the pipeline). `co_pairs`: object
    name pairs one hand may touch together (vocabulary co_contact_pairs; used if cfg.co_contact).
    Returns {"frames": per-frame rows, "events": closed events, "rejected_candidates": n}."""
    allow = set(cfg.hand_ids) if cfg.hand_ids else None
    pairs = {frozenset(p) for p in (co_pairs or [])} if getattr(cfg, "co_contact", True) else set()
    sm = ContactStateMachine(cfg.min_contact_frames, cfg.min_contact_s, cfg.release_grace_frames,
                             cfg.interpolated_can_start, cfg.allow_multiple, pairs)
    priority = {t: i for i, t in enumerate(getattr(cfg, "type_priority", None) or ())}
    per_frame = []
    last_choice: Dict[object, set] = defaultdict(set)
    for hr, orr in zip(hand_records, object_records):
        selected, rows = {}, []
        for h in hr["hands"]:
            hid = h["id"]
            row = {"hand_id": hid, "side": h.get("side"), "interpolated": bool(h.get("interpolated"))}
            if allow is not None and hid not in allow:
                row.update(state="not_analyzed", contacts=[], n_candidates=0)
                rows.append(row)
                continue
            cands = frame_candidates(h, orr["objects"], cfg.fingertip_threshold, cfg.pad_frac,
                                     cfg.min_contact_object_conf, getattr(cfg, "type_priority", None),
                                     getattr(cfg, "use_masks", True), getattr(cfg, "mask_grid", 32))
            prev = sm.active_objects(hid) | last_choice[hid]
            chosen = select_contacts(cands, prev, cfg.sticky_margin, cfg.allow_multiple, cfg.multi_min_fingertips,
                                     pairs)
            for c in chosen:
                c["side"], c["interpolated"] = h.get("side"), bool(h.get("interpolated"))
            last_choice[hid] = {c["object_id"] for c in chosen}
            selected[hid] = chosen
            row.update(n_candidates=len(cands), _chosen=chosen)
            rows.append(row)
        sm.step(hr["frame"], hr["t"], selected)
        per_frame.append({"frame": hr["frame"], "t": hr["t"], "hands": rows})
    events = sm.finish()
    # back-fill per-frame states from the confirmed events
    by_hand = defaultdict(list)
    for i, e in enumerate(events):
        by_hand[e["hand_id"]].append((i, e))
    for fr in per_frame:
        for row in fr["hands"]:
            chosen = row.pop("_chosen", None)
            if row.get("state") == "not_analyzed":
                continue
            contacts = []
            for i, e in by_hand.get(row["hand_id"], []):
                if e["start_frame"] <= fr["frame"] <= e["end_frame"]:
                    c = next((c for c in chosen or [] if c["object_id"] == e["object_id"]), None)
                    otype = e["types"].most_common(1)[0][0]
                    item = {"event_index": i, "object_id": e["object_id"],
                            "object_name": e["names"].most_common(1)[0][0], "object_type": otype,
                            "in_grace": c is None}
                    if c:
                        item.update({k: c[k] for k in ("fingertip_count", "tips", "min_tip_dist_px",
                                                       "object_conf", "score", "carried", "grasp")})
                    contacts.append(item)
            contacts.sort(key=lambda c: priority.get(c["object_type"], len(priority)))
            if contacts:
                row["state"] = f"touching_{contacts[0]['object_type']}"
                row["uncertain"] = row["interpolated"] or any(c["in_grace"] or c.get("carried") for c in contacts)
            else:
                row["state"] = "candidate" if chosen else "none"
                if chosen:
                    row["rejected_candidate"] = {k: chosen[0][k] for k in
                                                 ("object_id", "object_name", "object_type", "fingertip_count")}
            row["contacts"] = contacts
    return {"frames": per_frame, "events": events, "rejected_candidates": sm.rejected_candidates}


CSV_COLUMNS = ["clip", "t_start", "t_end", "hand", "state", "object_name", "hand_id", "object_id",
               "object_type", "start_frame", "end_frame", "duration_s", "mean_contact_score",
               "fingertip_count_mean", "start_event", "end_event", "uncertainty", "grasp", "co_contact"]


def event_rows(events: List[dict], clip: str) -> List[dict]:
    rows = []
    for e in events:
        otype = e["types"].most_common(1)[0][0]
        unc = ["2d_box_proximity"]
        if e["interpolated_frames"]:
            unc.append(f"interpolated_hand_frames={e['interpolated_frames']}")
        if e["carried_frames"]:
            unc.append(f"carried_object_frames={e['carried_frames']}")
        if len(e["names"]) > 1:
            unc.append("object_name_varied:" + "|".join(n for n, _ in e["names"].most_common()))
        if e["end_reason"] != "released":
            unc.append(f"ended_by={e['end_reason']}")
        rows.append({
            "clip": clip, "t_start": round(e["t_start"], 3), "t_end": round(e["t_end"], 3),
            "hand": e["sides"].most_common(1)[0][0], "state": f"touching_{otype}",
            "object_name": e["names"].most_common(1)[0][0], "hand_id": e["hand_id"],
            "object_id": e["object_id"], "object_type": otype, "start_frame": e["start_frame"],
            "end_frame": e["end_frame"], "duration_s": round(e["t_end"] - e["t_start"], 3),
            "mean_contact_score": round(float(np.mean(e["scores"])), 4),
            "fingertip_count_mean": round(float(np.mean(e["tips"])), 2),
            "start_event": START_EVENT[otype],
            "end_event": END_EVENT[otype] if e["end_reason"] != "clip_end" else "clip_end",
            "uncertainty": ";".join(unc),
            "grasp": e["grasps"].most_common(1)[0][0] if e.get("grasps") else "",
            "co_contact": bool(e.get("co_contact"))})
    return rows
