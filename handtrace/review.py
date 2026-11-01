"""Manual review of PS1 + PS2 + PS3 + PS5 on the original video: frame composition, jump-to-hard-moment targets,
a manual audit log that turns into real accuracy numbers, and all-layer segment rendering.

    v = ClipView(clip_path, OUT_ROOT)
    img = v.compose(frame, layers={"ps1", "ps2", "ps3", "ps5", "strip"})     # BGR image with every layer drawn
    v.next_target("ps1_low_score", frame)                                    # next hard frame of a kind
    sample_audit_frames(v, 100)                                              # stratified frames to check by hand
    AuditLog(OUT_ROOT, stem).save({...})  ->  audit_report(OUT_ROOT, stems)  # hand recall / precision / side accuracy
    render_review_segment(v, "demo.mp4", 30, 45)                             # private demo video, all layers

The interactive widget lives in review_widgets.py, the standalone HTML player in review_html.py.
Everything drawn here contains company footage: keep it private (event Drive folder only).
"""
from __future__ import annotations

import os
import random
import time
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from . import results as R
from .wilor.visualization import draw_hand

ALL_LAYERS = ("ps1", "ps2", "objects_all", "ps3", "ps5", "strip", "hud")
DEFAULT_LAYERS = {"ps1", "ps2", "ps3", "ps5", "strip", "hud"}
JUMP_KINDS = {
    "ps1_no_hand": "PS1: frame without any hand",
    "ps1_low_score": "PS1: low-confidence detection (< 0.4)",
    "ps1_interpolated": "PS1: gap-filled (interpolated) hand",
    "ps1_side_corrected": "PS1: left/right corrected by post-processing",
    "ps1_two_close": "PS1: two hands overlapping",
    "ps1_edge": "PS1: hand at the frame edge (leaving / entering)",
    "ps2_grab": "PS2: grab (contact start)",
    "ps2_release": "PS2: release (contact end)",
    "ps3_action": "PS3: action start",
    "ps3_refined": "PS3: action edge moved by PS2 refinement",
    "ps5_safety": "PS5: safety stretch",
    "ps5_quality": "PS5: blurry / dark / no-hand stretch",
    "audit_sample": "Audit: next sampled frame not yet reviewed",
}
VERB_COLOURS = {  # BGR, stable per verb
    "pick": (80, 200, 80), "place": (60, 160, 230), "carry": (200, 160, 60), "hold": (150, 150, 150),
    "insert": (200, 90, 200), "align": (230, 200, 80), "fasten": (40, 90, 230), "tighten": (40, 40, 200),
    "loosen": (120, 60, 200), "press": (0, 200, 200), "push": (100, 220, 220), "pull": (180, 120, 60),
    "turn": (220, 120, 160), "cut": (30, 30, 160), "wipe": (200, 200, 120), "inspect": (250, 130, 40),
    "adjust": (140, 200, 160), "other": (120, 120, 120)}


def hand_words(h: str) -> str:
    return "both hands" if h == "both" else f"{h} hand"


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _text(img, s, org, scale=0.6, col=(255, 255, 255), thick=1, bg=None):
    if bg is not None:
        (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (org[0] - 3, org[1] - h - 4), (org[0] + w + 3, org[1] + b + 1), bg, -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


class ClipView:
    """Everything known about one clip, indexed by frame, plus a frame reader that is fast for sequential access."""

    def __init__(self, clip_path: str, out_root: str, load_objects: bool = True):
        self.clip, self.out_root = clip_path, out_root
        self.stem = os.path.splitext(os.path.basename(clip_path))[0]
        self.p = R.paths(out_root, self.stem)
        self.info = R.video_info(out_root, self.stem, clip_path)
        self.fps = self.info["fps"]
        cap = cv2.VideoCapture(clip_path)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.info["frames"] or 0)
        self.W, self.H = int(cap.get(3)), int(cap.get(4))
        cap.release()
        self.hands = {r["frame"]: r for r in R.load_hands(out_root, self.stem, final=True)}
        self._raw = None
        self.objects = {r["frame"]: r for r in R.read_jsonl(self.p["objects"])} if load_objects else {}
        self.contact_frames = {r["frame"]: r for r in R.read_jsonl(self.p["contact_frames"])}
        self.contacts = R.load_contacts(out_root, self.stem)
        self.actions = R.load_actions(out_root, self.stem)
        self.safety = [dict(s, t_start=float(s["t_start"]), t_end=float(s["t_end"])) for s in R.read_csv(self.p["safety"])]
        self.quality = R.read_json(self.p["quality"])
        from .ps5.index import ps1_events
        self.ps1_segments = ps1_events(self.stem, list(self.hands.values())) if self.hands else []
        self._cap, self._pos, self._last = None, -1, None
        self._targets: Dict[str, List[int]] = {}
        self.audit = AuditLog(out_root, self.stem)
        self.samples: List[int] = []

    # ------------------------------------------------------------------------------------------- frames
    @property
    def hands_raw(self) -> Dict[int, dict]:
        if self._raw is None:
            self._raw = {r["frame"]: r for r in R.load_hands(self.out_root, self.stem, final=False)}
        return self._raw

    def read(self, f: int) -> Optional[np.ndarray]:
        f = int(min(max(0, f), max(0, self.n_frames - 1)))
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.clip)
        if f == self._pos and self._last is not None:
            return self._last.copy()
        if f != self._pos + 1 or not (0 < f - self._pos <= 30):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        else:
            while self._pos + 1 < f:
                self._cap.grab(); self._pos += 1
        ok, img = self._cap.read()
        self._pos, self._last = (f, img) if ok else (-1, None)
        return img.copy() if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release(); self._cap = None

    def t_of(self, f: int) -> float:
        return f / self.fps if self.fps else 0.0

    def f_of(self, t: float) -> int:
        return int(round(t * self.fps))

    # ------------------------------------------------------------------------------------------- state at a time
    def actions_at(self, t: float) -> List[dict]:
        return [a for a in self.actions if a["start_s"] <= t <= a["end_s"]]

    def contacts_at(self, t: float) -> List[dict]:
        return [c for c in self.contacts if c["t_start"] <= t <= c["t_end"]]

    def safety_at(self, t: float) -> List[dict]:
        return [s for s in self.safety if s["t_start"] <= t <= s["t_end"]]

    def quality_at(self, t: float) -> List[dict]:
        return [s for s in self.quality.get("bad_segments", []) if s["t_start"] <= t <= s["t_end"]]

    def info_at(self, f: int) -> Dict:
        t = self.t_of(f)
        rec = self.hands.get(f, {"hands": []})
        hands = [{"id": h.get("id"), "side": h.get("side"), "score": round(float(h.get("score", 0)), 3),
                  "interpolated": bool(h.get("interpolated")), "side_corrected": bool(h.get("side_corrected")),
                  "side_raw": h.get("side_raw"), "wearer_score": h.get("wearer_score")} for h in rec["hands"]]
        cf = self.contact_frames.get(f, {})
        states = {h["hand_id"]: (h.get("state"), [c["object_name"] for c in h.get("contacts", [])])
                  for h in cf.get("hands", [])}
        for h in hands:
            st = states.get(h["id"])
            if st:
                h["ps2_state"], h["ps2_objects"] = st
        return {"clip": self.stem, "frame": f, "t": round(t, 3), "time": _fmt_t(t), "hands": hands,
                "actions": [f"{a['verb']} {a['object']} ({a['hand']}) {_fmt_t(a['start_s'])}-{_fmt_t(a['end_s'])}"
                            + (f" [refined: {'; '.join(a['refined'])}]" if a.get("refined") else "")
                            for a in self.actions_at(t)],
                "safety": [f"{s['hand']} near {s['hazard']} ({s['level']}, risk {s['risk']})" for s in self.safety_at(t)],
                "quality": [s["reason"] for s in self.quality_at(t)],
                "audited": self.audit.get(f)}

    # ------------------------------------------------------------------------------------------- drawing
    def compose(self, f: int, layers: Iterable[str] = DEFAULT_LAYERS, raw: bool = False, width: Optional[int] = None,
                img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        layers = set(layers)
        img = self.read(f) if img is None else img
        if img is None:
            return None
        t = self.t_of(f)
        src = self.hands_raw if raw else self.hands
        hrec = {**src.get(f, {"hands": []}), "frame": f, "t": t}
        if "ps1" not in layers:
            hrec = {**hrec, "hands": []}
        orec, crec = self.objects.get(f), self.contact_frames.get(f)
        if "ps2" in layers and orec is not None and crec is not None and not raw:
            from .ps2.visualization import draw_ps2
            if "objects_all" not in layers:
                active = {c["object_id"] for h in crec["hands"] for c in h.get("contacts", [])}
                orec = {**orec, "objects": [o for o in orec["objects"] if o["object_id"] in active]}
            img = draw_ps2(img, hrec, orec, crec)
        else:
            for h in hrec["hands"]:
                draw_hand(img, h)
        s = max(0.6, img.shape[0] / 1080.0)
        if "ps5" in layers:
            sf = self.safety_at(t)
            if sf:
                worst = max(sf, key=lambda x: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(x["risk"], 0))
                col = (0, 0, 255) if worst["risk"] == "HIGH" else (0, 140, 255)
                cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, max(4, int(10 * s)))
                _text(img, f"SAFETY {worst['risk']}: {worst['hand']} hand {worst['level']} {worst['hazard']}",
                      (int(14 * s), int(img.shape[0] - 60 * s)), 0.8 * s, (255, 255, 255), max(1, int(2 * s)), bg=col)
            qa = self.quality_at(t)
            if qa:
                _text(img, "QUALITY: " + qa[0]["reason"], (int(14 * s), int(img.shape[0] - 100 * s)), 0.7 * s,
                      (0, 0, 0), max(1, int(2 * s)), bg=(0, 220, 255))
        if "ps3" in layers:
            acts = self.actions_at(t)
            bar_h = int(46 * s)
            cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (25, 25, 25), -1)
            if acts:
                x = int(12 * s)
                for a in acts[:3]:
                    col = VERB_COLOURS.get(a["verb"], (200, 200, 200))
                    label = f"{a['verb'].upper()} {a['object']} [{a['hand']}] {_fmt_t(a['start_s'])}-{_fmt_t(a['end_s'])}"
                    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, max(1, int(2 * s)))
                    cv2.rectangle(img, (x - 6, int(6 * s)), (x + w + 6, bar_h - int(6 * s)), col, -1)
                    _text(img, label, (x, int(32 * s)), 0.75 * s, (255, 255, 255), max(1, int(2 * s)))
                    x += w + int(24 * s)
            else:
                _text(img, "PS3: no action" if self.actions else "PS3: not run", (int(12 * s), int(32 * s)), 0.7 * s,
                      (170, 170, 170), max(1, int(2 * s)))
        if "hud" in layers:
            n = len(hrec["hands"])
            _text(img, f"{self.stem}  frame {f}  {_fmt_t(t)}  hands {n}" + ("  RAW" if raw else ""),
                  (int(12 * s), int(img.shape[0] - 18 * s)), 0.65 * s, (255, 255, 255), max(1, int(2 * s)), bg=(0, 0, 0))
        if width and img.shape[1] != width:
            w2, h2 = width // 2 * 2, int(round(img.shape[0] * width / img.shape[1])) // 2 * 2
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        if "strip" in layers:
            img = np.concatenate([img, self.timeline_strip(img.shape[1], t)], axis=0)
        return img

    def timeline_strip(self, width: int, t: float, window_s: float = 30.0, lane_h: int = 0) -> np.ndarray:
        """Scrolling lanes under the frame: PS1 L/R visible, PS2 L/R contact, PS3 actions, PS5 safety. Playhead centred."""
        lane_h = lane_h or max(14, width // 80)
        lanes = [("L hand", "ps1", "left"), ("R hand", "ps1", "right"), ("L touch", "ps2", "left"),
                 ("R touch", "ps2", "right"), ("action", "ps3", None), ("safety", "ps5", None)]
        lab_w = int(lane_h * 4.2)
        h = lane_h * len(lanes) + 6
        strip = np.full((h + lane_h, width, 3), 18, np.uint8)
        t0, t1 = t - window_s / 2, t + window_s / 2
        span = width - lab_w - 4
        X = lambda tt: int(lab_w + (min(max(tt, t0), t1) - t0) / window_s * span)  # noqa: E731
        if t0 < 0:                                                   # before the clip starts: hatched
            cv2.rectangle(strip, (lab_w, 0), (X(0), h), (28, 28, 28), -1)
        fs = lane_h / 30.0
        for k, (name, src, side) in enumerate(lanes):
            y = 3 + k * lane_h
            cv2.rectangle(strip, (lab_w, y + 1), (width - 4, y + lane_h - 2), (35, 35, 35), -1)
            _text(strip, name, (4, y + lane_h - 4), fs, (200, 200, 200), 1)
            if src == "ps1":
                items = [(e["t_start"], e["t_end"], (255, 160, 40) if side == "left" else (40, 120, 255), "")
                         for e in self.ps1_segments if e["kind"] == "hands_visible" and e["hand"] == side]
            elif src == "ps2":
                items = [(c["t_start"], c["t_end"], (0, 140, 255) if c.get("object_type") == "portable" else (230, 160, 0),
                          c["object_name"]) for c in self.contacts if c.get("hand") == side]
            elif src == "ps3":
                items = [(a["start_s"], a["end_s"], VERB_COLOURS.get(a["verb"], (200, 200, 200)),
                          f"{a['verb']} {a['object']}") for a in self.actions]
            else:
                items = [(s["t_start"], s["t_end"], (0, 0, 255) if s["risk"] == "HIGH" else (0, 140, 255), s["hazard"])
                         for s in self.safety]
            for a, b, col, label in items:
                if b < t0 or a > t1:
                    continue
                xa, xb = X(a), max(X(b), X(a) + 2)
                cv2.rectangle(strip, (xa, y + 2), (xb, y + lane_h - 3), col, -1)
                if label and xb - xa > 40:
                    _text(strip, label[: max(1, (xb - xa) // max(1, int(9 * fs * 1.6)))], (xa + 2, y + lane_h - 5),
                          fs * 0.9, (255, 255, 255), 1)
        xp = X(t)
        cv2.line(strip, (xp, 0), (xp, h), (255, 255, 255), 2)
        for k in range(-int(window_s // 2), int(window_s // 2) + 1, 5):
            tt = t + k - (t % 5)
            if t0 <= tt <= t1 and tt >= 0:
                cv2.line(strip, (X(tt), h), (X(tt), h + 4), (150, 150, 150), 1)
                _text(strip, _fmt_t(tt), (X(tt) + 2, h + lane_h - 2), fs * 0.9, (150, 150, 150), 1)
        return strip

    # ------------------------------------------------------------------------------------------- jump targets
    def targets(self, kind: str) -> List[int]:
        if kind == "audit_sample":
            done = set(self.audit.rows)
            return [f for f in sorted(self.samples) if f not in done]
        if kind in self._targets:
            return self._targets[kind]
        F = self.f_of
        out: List[int] = []
        recs = [self.hands[k] for k in sorted(self.hands)]
        if kind == "ps1_no_hand":
            had_hand = True                                   # first frame of each no-hand stretch
            for r in recs:
                if not r["hands"] and had_hand:
                    out.append(r["frame"])
                had_hand = bool(r["hands"])
        elif kind == "ps1_low_score":
            out = [r["frame"] for r in recs if any(not h.get("interpolated") and float(h.get("score", 1)) < 0.4
                                                    for h in r["hands"])]
        elif kind == "ps1_interpolated":
            out = [r["frame"] for r in recs if any(h.get("interpolated") for h in r["hands"])]
        elif kind == "ps1_side_corrected":
            out = [r["frame"] for r in recs if any(h.get("side_corrected") for h in r["hands"])]
        elif kind == "ps1_two_close":
            for r in recs:
                hs = r["hands"]
                if len(hs) >= 2 and _iou(hs[0]["bbox"], hs[1]["bbox"]) > 0.05:
                    out.append(r["frame"])
        elif kind == "ps1_edge":
            m = 0.02
            for r in recs:
                if any(h["bbox"][0] < m * self.W or h["bbox"][2] > (1 - m) * self.W or h["bbox"][3] > (1 - m) * self.H
                       for h in r["hands"]):
                    out.append(r["frame"])
        elif kind == "ps2_grab":
            out = sorted(F(c["t_start"]) for c in self.contacts)
        elif kind == "ps2_release":
            out = sorted(F(c["t_end"]) for c in self.contacts)
        elif kind == "ps3_action":
            out = sorted(F(a["start_s"]) for a in self.actions)
        elif kind == "ps3_refined":
            out = sorted(F(a["start_s"]) for a in self.actions if a.get("refined"))
        elif kind == "ps5_safety":
            out = sorted(F(s["t_start"]) for s in self.safety)
        elif kind == "ps5_quality":
            out = sorted(F(s["t_start"]) for s in self.quality.get("bad_segments", []))
        out = _thin(out, int(self.fps // 2) if kind.startswith("ps1") else 0)
        self._targets[kind] = out
        return out

    def next_target(self, kind: str, f: int, direction: int = 1) -> Optional[int]:
        ts = self.targets(kind)
        if not ts:
            return None
        if direction > 0:
            i = bisect_right(ts, f)
            return ts[i] if i < len(ts) else None
        i = bisect_left(ts, f) - 1
        return ts[i] if i >= 0 else None

    def target_counts(self) -> Dict[str, int]:
        return {k: len(self.targets(k)) for k in JUMP_KINDS}


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _thin(frames: List[int], min_gap: int) -> List[int]:
    """Keep the first frame of each run (frames closer than min_gap belong to one moment)."""
    if min_gap <= 1:
        return sorted(set(frames))
    out: List[int] = []
    for f in sorted(set(frames)):
        if not out or f - out[-1] > min_gap:
            out.append(f)
    return out


# ============================================================================================ manual audit
AUDIT_FIELDS = ["clip", "frame", "t", "split", "pred_hands", "missed_hands", "false_hands", "side_swaps",
                "bad_keypoints", "ps2_contact", "ps3_action", "safety_flag", "note", "reviewer", "saved_at"]


class AuditLog:
    """One row per reviewed frame in OUT_ROOT/<stem>/review/manual_audit.csv (private: no footage, numbers only).

    For each frame the reviewer counts, among the worker's OWN hands that are visible:
        missed_hands   visible but not found          false_hands   found but not a wearer hand / not a hand
        side_swaps     found with the wrong side      bad_keypoints found but the 21 points are clearly off
    and judges, if shown: ps2_contact / ps3_action / safety_flag = ok | wrong | na."""

    def __init__(self, out_root: str, stem: str):
        self.path = R.paths(out_root, stem)["audit"]
        self.stem = stem
        self.rows: Dict[int, dict] = {}
        for r in R.read_csv(self.path):
            try:
                self.rows[int(r["frame"])] = r
            except (KeyError, ValueError):
                continue

    def get(self, f: int) -> Optional[dict]:
        return self.rows.get(int(f))

    def save(self, row: dict) -> dict:
        row = {k: row.get(k, "") for k in AUDIT_FIELDS}
        row["clip"] = self.stem
        row["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.rows[int(row["frame"])] = row
        R.write_csv(self.path, [self.rows[k] for k in sorted(self.rows)], AUDIT_FIELDS)
        return row

    def delete(self, f: int) -> None:
        if self.rows.pop(int(f), None) is not None:
            R.write_csv(self.path, [self.rows[k] for k in sorted(self.rows)], AUDIT_FIELDS)

    def merge_csv(self, path: str) -> int:
        """Import rows exported by the HTML player (same columns). Later rows win. Returns the number imported."""
        n = 0
        for r in R.read_csv(path):
            if r.get("clip") and r["clip"] != self.stem:
                continue
            try:
                self.rows[int(r["frame"])] = {k: r.get(k, "") for k in AUDIT_FIELDS} | {"clip": self.stem}
                n += 1
            except (KeyError, ValueError):
                continue
        R.write_csv(self.path, [self.rows[k] for k in sorted(self.rows)], AUDIT_FIELDS)
        return n


def sample_audit_frames(view: ClipView, n: int = 100, seed: int = 0, hard_share: float = 0.6) -> List[int]:
    """Stratified frames to review: `hard_share` from the hard kinds (blur shows up as low score / interpolation,
    plus side corrections, overlaps, frame edges, grabs), the rest uniformly at random, so the result reflects both
    the hard moments judges look at and the clip as a whole. Alternate frames are marked dev / test."""
    rng = random.Random(seed)
    hard_kinds = ["ps1_low_score", "ps1_interpolated", "ps1_side_corrected", "ps1_two_close", "ps1_edge",
                  "ps2_grab", "ps1_no_hand"]
    pools = {k: [f for f in view.targets(k) if 0 <= f < view.n_frames] for k in hard_kinds}
    n_hard = int(n * hard_share)
    picked: List[int] = []
    per = max(1, n_hard // max(1, sum(bool(v) for v in pools.values())))
    for k in hard_kinds:
        pool = [f for f in pools[k] if all(abs(f - p) > view.fps for p in picked)]
        picked += rng.sample(pool, min(per, len(pool)))
    pool = [f for f in range(0, view.n_frames, max(1, int(view.fps // 2))) if all(abs(f - p) > view.fps / 2 for p in picked)]
    picked += rng.sample(pool, min(max(0, n - len(picked)), len(pool)))
    picked = sorted(picked)[:n]
    view.samples = picked
    return picked


def _num(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def audit_report(out_root: str, stems: Sequence[str], split: Optional[str] = None) -> Dict:
    """Manual-audit rows -> hand-tracking accuracy numbers (the honest ones to put on a slide, with frame counts)."""
    per_clip, tot = [], {"frames": 0, "pred": 0, "missed": 0, "false": 0, "swaps": 0, "badkp": 0, "perfect": 0,
                         "ps2_ok": 0, "ps2_n": 0, "ps3_ok": 0, "ps3_n": 0, "safety_ok": 0, "safety_n": 0}
    for stem in stems:
        rows = list(AuditLog(out_root, stem).rows.values())
        if split:
            rows = [r for r in rows if r.get("split") == split]
        if not rows:
            continue
        c = {k: 0 for k in tot}
        for r in rows:
            pred, miss, fal, sw, bk = (_num(r.get(k)) for k in ("pred_hands", "missed_hands", "false_hands",
                                                                 "side_swaps", "bad_keypoints"))
            c["frames"] += 1; c["pred"] += pred; c["missed"] += miss; c["false"] += fal; c["swaps"] += sw; c["badkp"] += bk
            c["perfect"] += int(miss == fal == sw == bk == 0)
            for key in ("ps2", "ps3", "safety"):
                v = (r.get({"ps2": "ps2_contact", "ps3": "ps3_action", "safety": "safety_flag"}[key]) or "").lower()
                if v in ("ok", "wrong"):
                    c[f"{key}_n"] += 1; c[f"{key}_ok"] += int(v == "ok")
        per_clip.append({"clip": stem, **_rates(c)})
        for k in tot:
            tot[k] += c[k]
    return {"split": split or "all", "clips": per_clip, "total": _rates(tot) if tot["frames"] else {},
            "definitions": {
                "hand_recall": "wearer hands found / wearer hands visible (found + missed)",
                "hand_precision": "found wearer hands / all predicted hands",
                "side_accuracy": "found hands with the correct left/right / found hands",
                "keypoint_ok_rate": "found hands whose 21 points sit on the right joints / found hands",
                "frames_fully_correct": "frames with no miss, no false hand, no swap and no bad keypoints"}}


def _rates(c: dict) -> dict:
    found = max(0, c["pred"] - c["false"])
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"frames_reviewed": c["frames"], "predicted_hands": c["pred"], "found_wearer_hands": found,
            "missed_hands": c["missed"], "false_hands": c["false"], "side_swaps": c["swaps"],
            "bad_keypoints": c["badkp"], "hand_recall": div(found, found + c["missed"]),
            "hand_precision": div(found, c["pred"]), "side_accuracy": div(found - c["swaps"], found),
            "keypoint_ok_rate": div(found - c["badkp"], found), "frames_fully_correct": div(c["perfect"], c["frames"]),
            "ps2_contact_accuracy": div(c["ps2_ok"], c["ps2_n"]), "ps2_judged": c["ps2_n"],
            "ps3_action_accuracy": div(c["ps3_ok"], c["ps3_n"]), "ps3_judged": c["ps3_n"],
            "safety_flag_accuracy": div(c["safety_ok"], c["safety_n"]), "safety_judged": c["safety_n"]}


# ============================================================================================ rendering
def render_review_segment(view: ClipView, out: str, start_s: float = 0.0, end_s: Optional[float] = None,
                          layers: Iterable[str] = DEFAULT_LAYERS, width: Optional[int] = 1280, compare_raw: bool = False,
                          crf: int = 23, log=print) -> str:
    """Private demo video of a time range with every chosen layer (and optionally raw PS1 side by side)."""
    from .wilor.video_loader import VideoReader, VideoWriter
    start = max(0, int(round(start_s * view.fps)))
    stop = view.n_frames if end_s is None else min(view.n_frames, int(round(end_s * view.fps)))
    writer, count = None, 0
    layers = set(layers)
    for fidx, _, img in VideoReader(view.clip, start=start, max_frames=max(0, stop - start)):
        tile = view.compose(fidx, layers, img=img.copy(), width=(width // 2 if compare_raw else width) if width else None)
        if compare_raw:
            left = view.compose(fidx, (layers - {"ps2", "ps3", "ps5", "strip"}) | {"hud"}, raw=True, img=img,
                                width=tile.shape[1])
            pad = np.zeros((tile.shape[0] - left.shape[0], left.shape[1], 3), np.uint8)
            tile = np.concatenate([np.concatenate([left, pad], 0), tile], 1)
        tile = tile[: tile.shape[0] // 2 * 2, : tile.shape[1] // 2 * 2]
        if writer is None:
            writer = VideoWriter(out, view.fps, (tile.shape[1], tile.shape[0]), crf)
        writer.write(tile)
        count += 1
    path = writer.close() if writer else out
    log(f"[review] {count} frames -> {path}")
    return path
