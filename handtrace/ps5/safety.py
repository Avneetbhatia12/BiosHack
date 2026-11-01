"""PS5 / PS2-bonus safety monitor: flag stretches where a hand is inside or close to a hazard object.

Inputs (no GPU, no new model): PS1 hands_final.jsonl + PS2 object_detections.jsonl + PS2 contacts.jsonl.
A *hazard* is a detected object whose name contains one of `hazard_words` (machines, spindles, blades, presses,
conveyors, clamps, ...). Per frame and hand:
    touching   PS2 says this hand is in contact with the hazard object
    inside     a fingertip lies inside the hazard box
    near       some hand keypoint is within `near_frac` x hand size of the hazard box
A stretch is reported when it lasts at least `min_s` (short flickers are ignored). With `check_motion`, the hazard
box is compared between two frames of the stretch: a large pixel change means the part is probably MOVING, the
case that matters most ("hand close to a moving machine part").

Honest limits: this is 2D image proximity. A hand in front of a machine (closer to the camera) looks "inside" its
box. It is a screening tool that points a safety reviewer at moments to check, not a measured distance.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results as R

TIPS = (4, 8, 12, 16, 20)
LEVEL_RANK = {"near": 1, "inside": 2, "touching": 3}


@dataclass
class SafetyConfig:
    hazard_words: List[str] = field(default_factory=lambda: [
        "machine", "spindle", "lathe", "blade", "saw", "press", "conveyor", "grinder", "polishing",
        "cutter", "clamp", "motor", "crane", "chain", "roller", "belt", "gear", "fan"])
    exclude_words: List[str] = field(default_factory=lambda: ["surface", "bench", "table", "shelf", "controller"])
    only_fixed: bool = False            # True: only objects PS2 typed "fixed" can be hazards
    near_frac: float = 0.35             # keypoint within this x hand size of the box = near
    min_s: float = 0.3
    grace_frames: int = 3
    min_level: str = "near"             # report stretches whose worst level is at least this (risk LOW for "near")
    check_motion: bool = True
    motion_thr: float = 12.0            # mean |pixel change| (0..255) inside the box between two frames


def is_hazard(name: str, otype: str, cfg: SafetyConfig) -> bool:
    n = (name or "").lower().replace("_", " ")
    if cfg.only_fixed and otype != "fixed":
        return False
    if any(w in n for w in cfg.exclude_words):
        return False
    return any(w in n for w in cfg.hazard_words)


def _box_dist(p, b) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def frame_level(hand: dict, box, in_contact: bool, near_frac: float) -> Optional[str]:
    kp = hand.get("kp2d")
    if not kp or len(kp) < 21:
        return None
    if in_contact:
        return "touching"
    if any(box[0] <= kp[i][0] <= box[2] and box[1] <= kp[i][1] <= box[3] for i in TIPS):
        return "inside"
    x1, y1, x2, y2 = hand["bbox"]
    size = max(x2 - x1, y2 - y1, 1.0)
    if min(_box_dist(p, box) for p in kp) <= near_frac * size:
        return "near"
    return None


def _motion(video: str, f_a: int, f_b: int, box) -> Optional[float]:
    import cv2
    cap = cv2.VideoCapture(video)
    imgs = []
    for f in (f_a, f_b):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            cap.release(); return None
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = imgs[0].shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    a, b = imgs[0][y1:y2, x1:x2].astype(np.float32), imgs[1][y1:y2, x1:x2].astype(np.float32)
    return float(np.abs(a - b).mean())


def analyse_clip(clip_path: str, out_root: str, cfg: Optional[SafetyConfig] = None, write: bool = True,
                 log=print) -> List[Dict]:
    cfg = cfg or SafetyConfig()
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    p = R.paths(out_root, stem)
    hands = R.load_hands(out_root, stem)
    objs = {r["frame"]: r["objects"] for r in R.read_jsonl(p["objects"])}
    cons = {r["frame"]: r for r in R.read_jsonl(p["contact_frames"])}
    if not hands or not objs:
        log(f"[safety] {stem}: needs PS1 hands and PS2 object detections; skipped")
        return []
    fps = R.video_info(out_root, stem, clip_path)["fps"]
    open_: Dict[tuple, dict] = {}
    events: List[dict] = []

    def close(key):
        e = open_.pop(key)
        if (e["last"] - e["first"] + 1) / fps >= cfg.min_s and LEVEL_RANK[e["worst"]] >= LEVEL_RANK[cfg.min_level]:
            events.append(e)

    for r in hands:
        f = r["frame"]
        contact_ids = {}
        for hr in (cons.get(f) or {}).get("hands", []):
            contact_ids[hr["hand_id"]] = {c["object_id"] for c in hr.get("contacts", [])}
        seen = set()
        for o in objs.get(f, []):
            if not is_hazard(o.get("object_name", ""), o.get("object_type", ""), cfg):
                continue
            for h in r["hands"]:
                lvl = frame_level(h, o["bbox"], o["object_id"] in contact_ids.get(h.get("id"), set()), cfg.near_frac)
                if lvl is None:
                    continue
                key = (h.get("id"), o["object_id"])
                seen.add(key)
                e = open_.get(key)
                if e is None:
                    e = open_[key] = {"hand_id": h.get("id"), "hand": h.get("side"), "object_id": o["object_id"],
                                      "hazard": o.get("object_name", ""), "first": f, "last": f, "worst": lvl,
                                      "levels": {}, "box": o["bbox"]}
                e["last"] = f
                e["levels"][lvl] = e["levels"].get(lvl, 0) + 1
                if LEVEL_RANK[lvl] > LEVEL_RANK[e["worst"]]:
                    e["worst"], e["box"] = lvl, o["bbox"]
        for key in [k for k, e in open_.items() if k not in seen and f - e["last"] > cfg.grace_frames]:
            close(key)
    for key in list(open_):
        close(key)
    rows = []
    for e in sorted(events, key=lambda x: x["first"]):
        moving = None
        if cfg.check_motion and e["last"] - e["first"] >= 2:
            mid = (e["first"] + e["last"]) // 2
            moving = _motion(clip_path, mid, min(e["last"], mid + max(2, int(fps // 5))), e["box"])
        note = []
        if moving is not None:
            note.append(f"motion {moving:.1f}" + (" MOVING" if moving >= cfg.motion_thr else ""))
        rows.append({"clip": stem, "t_start": round(e["first"] / fps, 2), "t_end": round(e["last"] / fps, 2),
                     "duration_s": round((e["last"] - e["first"] + 1) / fps, 2), "hand": e["hand"],
                     "hand_id": e["hand_id"], "hazard": e["hazard"], "object_id": e["object_id"], "level": e["worst"],
                     "frames_touching": e["levels"].get("touching", 0), "frames_inside": e["levels"].get("inside", 0),
                     "frames_near": e["levels"].get("near", 0),
                     "moving": None if moving is None else bool(moving >= cfg.motion_thr),
                     "risk": ("HIGH" if moving is not None and moving >= cfg.motion_thr and e["worst"] != "near"
                              else "MEDIUM" if e["worst"] != "near" else "LOW"),
                     "note": "; ".join(note)})
    if write:
        R.write_csv(p["safety"], rows, list(rows[0].keys()) if rows else
                    ["clip", "t_start", "t_end", "duration_s", "hand", "hand_id", "hazard", "object_id", "level",
                     "frames_touching", "frames_inside", "frames_near", "moving", "risk", "note"])
    log(f"[safety] {stem}: {len(rows)} stretch(es) " + str({k: sum(r['risk'] == k for r in rows)
                                                           for k in ('HIGH', 'MEDIUM', 'LOW')}))
    return rows


def analyse_all(clips: Sequence[str], out_root: str, cfg: Optional[SafetyConfig] = None, log=print) -> List[Dict]:
    rows = []
    for c in clips:
        try:
            rows += analyse_clip(c, out_root, cfg, log=log)
        except Exception as e:                                   # noqa: BLE001
            log(f"[safety] {os.path.basename(c)}: FAILED {e!r}")
    R.write_csv(os.path.join(out_root, "ps5_safety_all.csv"), rows,
                list(rows[0].keys()) if rows else ["clip", "t_start", "t_end", "hand", "hazard", "level", "risk"])
    return rows


def config_dict(cfg: SafetyConfig) -> dict:
    return asdict(cfg)
