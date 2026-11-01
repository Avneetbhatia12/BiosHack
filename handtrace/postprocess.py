"""Offline post-processing of a whole clip's hand records (pure: records in, records out).

Input/output records use the hackathon format written by the WiLoR pipeline:
    {"frame": 120, "t": 4.0, "hands": [{"id": 3, "side": "left", "bbox": [x1, y1, x2, y2],
      "score": 0.93, "kp2d": [[x, y] * 21], "kp3d": [...], "interpolated": false}]}

Because the whole clip is available, decisions can use the future as well as the past:
  1. dedup        drop a second box on the same hand (overlap / smaller-box area)
  2. wearer       track-level score from hand size, height in the frame and forearm direction;
                  background workers' tracks are removed (or only flagged)
  3. sides        windowed majority per track, then the whole-track majority when it is clear
  4. relink       a track that starts shortly after a same-side track ended, near where it ended
                  (or both at a frame edge = left and came back), continues that track's ID
  5. cap          at most `max_hands_per_side` wearer hands per side per frame; two same-side
                  hands with no opposite one -> the less certain one is flipped
  6. fill         linear interpolation of short gaps inside a (relinked) track
  7. renumber     IDs 1..n in order of first appearance
Every changed hand keeps evidence: `side_raw`, `side_corrected`, `wearer_score`, `interpolated`.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PostConfig:
    frame_wh: tuple[int, int]
    # 1. dedup
    dedup_overlap: float = 0.6            # intersection / smaller box area above this = same hand
    # 2. wearer filter
    wearer_filter: bool = True            # False = only flag hands with is_wearer / wearer_score
    wearer_min_score: float = 0.45
    wearer_w_size: float = 0.45
    wearer_w_low: float = 0.30
    wearer_w_arm: float = 0.25
    size_lo: float = 0.04                 # bbox diagonal / frame diagonal scoring 0 ...
    size_hi: float = 0.14                 # ... and 1
    # 3. sides
    side_window: int = 15                 # frames, centred majority window inside a track
    side_track_majority: float = 0.75     # weighted share above which the whole track takes it
    # 4. relink
    relink: bool = True
    relink_max_gap_s: float = 2.0
    relink_max_dist: float = 0.30         # wrist distance / frame diagonal
    edge_margin: float = 0.08             # fraction of width/height counted as "at the edge"
    # 5. cap
    max_hands_per_side: int = 1
    # 6. fill
    max_fill_gap: int = 8                 # frames
    # 7.
    renumber: bool = True


# ----------------------------------------------------------------------------- helpers
def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def overlap_small(a, b) -> float:
    """Intersection over the smaller box's area (1.0 = one box inside the other)."""
    return _inter(a, b) / max(1e-9, min(_area(a), _area(b)))


def _wrist(h) -> np.ndarray:
    return np.asarray(h["kp2d"][0], float)


def _rank(h) -> tuple:
    """Sort key: real detections before interpolated ones, then by score."""
    return (bool(h.get("interpolated")), -float(h.get("score", 0.0)))


def wearer_frame_score(hand: dict, cfg: PostConfig) -> float:
    """0..1, higher = more likely the camera wearer's hand (from one frame)."""
    W, H = cfg.frame_wh
    diag = float(np.hypot(W, H))
    x1, y1, x2, y2 = hand["bbox"]
    size = float(np.hypot(x2 - x1, y2 - y1)) / diag
    s_size = float(np.clip((size - cfg.size_lo) / max(1e-9, cfg.size_hi - cfg.size_lo), 0, 1))
    kp = np.asarray(hand["kp2d"], float)
    s_low = float(np.clip(kp[0, 1] / H, 0, 1))
    v = kp[0] - kp[9]                          # middle-finger knuckle -> wrist = forearm direction
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        s_arm = 0.5
    else:
        d = v / n
        outward = 1.0 if kp[0, 0] >= W / 2 else -1.0     # towards the nearer side edge
        s_arm = float(np.clip(max(d[1], 0.7 * d[0] * outward), 0, 1))
    tot = cfg.wearer_w_size + cfg.wearer_w_low + cfg.wearer_w_arm
    return (cfg.wearer_w_size * s_size + cfg.wearer_w_low * s_low + cfg.wearer_w_arm * s_arm) / tot


def _tracks(records: list[dict]) -> dict:
    """id -> list of (record index, hand) in time order."""
    out = defaultdict(list)
    for ri, r in enumerate(records):
        for h in r["hands"]:
            out[h["id"]].append((ri, h))
    return out


def _majority(hands: list[dict]) -> tuple[str, float]:
    w = defaultdict(float)
    for h in hands:
        w[h["side"]] += max(1e-3, float(h.get("score", 1.0)))
    side = max(w, key=w.get)
    return side, w[side] / sum(w.values())


def _near_edge(p, cfg: PostConfig) -> bool:
    W, H = cfg.frame_wh
    mx, my = cfg.edge_margin * W, cfg.edge_margin * H
    return p[0] < mx or p[0] > W - mx or p[1] > H - my


# ----------------------------------------------------------------------------- steps
def dedup(records: list[dict], cfg: PostConfig) -> int:
    removed = 0
    for r in records:
        keep = []
        for h in sorted(r["hands"], key=_rank):
            if all(overlap_small(h["bbox"], k["bbox"]) <= cfg.dedup_overlap for k in keep):
                keep.append(h)
            else:
                removed += 1
        r["hands"] = keep
    return removed


def filter_wearer(records: list[dict], cfg: PostConfig) -> dict:
    tracks = _tracks(records)
    scores = {}
    for tid, items in tracks.items():
        real = [h for _, h in items if not h.get("interpolated")] or [h for _, h in items]
        scores[tid] = float(np.median([wearer_frame_score(h, cfg) for h in real]))
    removed_tracks = removed_hands = 0
    for tid, items in tracks.items():
        wearer = scores[tid] >= cfg.wearer_min_score
        removed_tracks += not wearer
        for _, h in items:
            h["wearer_score"] = round(scores[tid], 3)
            h["is_wearer"] = wearer
    if cfg.wearer_filter:
        for r in records:
            n = len(r["hands"])
            r["hands"] = [h for h in r["hands"] if h["is_wearer"]]
            removed_hands += n - len(r["hands"])
    return {"tracks": len(tracks), "background_tracks": removed_tracks,
            "background_hands_removed": removed_hands}


def resolve_sides(records: list[dict], cfg: PostConfig) -> int:
    changed = 0
    half = cfg.side_window // 2
    for tid, items in _tracks(records).items():
        hands = [h for _, h in items]
        for h in hands:
            h.setdefault("side_raw", h["side"])
        orig = [h["side"] for h in hands]
        side, share = _majority(hands)
        if share >= cfg.side_track_majority:
            new = [side] * len(hands)
        else:
            new = [_majority(hands[max(0, i - half): i + half + 1])[0] for i in range(len(hands))]
        for h, o, s in zip(hands, orig, new):
            h["side"] = s
            if s != o:
                changed += 1
            h["side_corrected"] = h["side"] != h["side_raw"]
    return changed


def relink(records: list[dict], cfg: PostConfig) -> int:
    W, H = cfg.frame_wh
    diag = float(np.hypot(W, H))
    spans = []
    for tid, items in _tracks(records).items():
        side, _ = _majority([h for _, h in items])
        (r0, h0), (r1, h1) = items[0], items[-1]
        spans.append({"id": tid, "side": side, "t0": records[r0]["t"], "t1": records[r1]["t"],
                      "f0": records[r0]["frame"], "f1": records[r1]["frame"],
                      "p0": _wrist(h0), "p1": _wrist(h1)})
    spans.sort(key=lambda s: s["f0"])
    chain_end: dict = {}          # root id -> span currently ending the chain
    root_of: dict = {}
    links = 0
    for s in spans:
        best, best_key = None, None
        for root, e in chain_end.items():
            gap = s["t0"] - e["t1"]
            if e["side"] != s["side"] or s["f0"] <= e["f1"] or gap > cfg.relink_max_gap_s:
                continue
            dist = float(np.linalg.norm(s["p0"] - e["p1"])) / diag
            if dist <= cfg.relink_max_dist or (_near_edge(e["p1"], cfg) and _near_edge(s["p0"], cfg)):
                key = (gap, dist)
                if best_key is None or key < best_key:
                    best, best_key = root, key
        if best is None:
            root_of[s["id"]] = s["id"]
            chain_end[s["id"]] = s
        else:
            root_of[s["id"]] = best
            chain_end[best] = {**s, "side": chain_end[best]["side"]}
            links += 1
    for r in records:
        for h in r["hands"]:
            h["id"] = root_of.get(h["id"], h["id"])
    return links


def cap_per_side(records: list[dict], cfg: PostConfig) -> dict:
    conf = {}                      # id -> share of the track's majority side
    for tid, items in _tracks(records).items():
        conf[tid] = _majority([h for _, h in items])[1]
    flipped = dropped = 0
    for r in records:
        hs = r["hands"]
        sides = {h["side"] for h in hs}
        if len(hs) == 2 and len(sides) == 1:
            a, b = hs
            weaker = min((a, b), key=lambda h: (conf.get(h["id"], 0.0), -_rank(h)[1]))
            other = b if weaker is a else a
            if conf.get(a["id"], 0) == conf.get(b["id"], 0):   # tie: left one in the image is "left"
                weaker = max((a, b), key=lambda h: _wrist(h)[0]) if a["side"] == "left" else \
                    min((a, b), key=lambda h: _wrist(h)[0])
                other = b if weaker is a else a
            weaker["side"] = "right" if other["side"] == "left" else "left"
            weaker.setdefault("side_raw", other["side"])
            weaker["side_corrected"] = weaker["side"] != weaker["side_raw"]
            flipped += 1
        keep, per_side = [], defaultdict(int)
        for h in sorted(hs, key=_rank):
            if per_side[h["side"]] < cfg.max_hands_per_side:
                keep.append(h)
                per_side[h["side"]] += 1
            else:
                dropped += 1
        r["hands"] = sorted(keep, key=lambda h: h["id"])
    return {"pair_side_flips": flipped, "dropped_over_cap": dropped}


def _lerp(a, b, w):
    return np.round((1 - w) * np.asarray(a, float) + w * np.asarray(b, float), 3).tolist()


def fill_gaps(records: list[dict], cfg: PostConfig) -> int:
    filled = 0
    for tid, items in _tracks(records).items():
        for (ra, ha), (rb, hb) in zip(items, items[1:]):
            gap = rb - ra - 1
            if not (1 <= gap <= cfg.max_fill_gap) or ha["side"] != hb["side"]:
                continue
            for ri in range(ra + 1, rb):
                if any(h["id"] == tid for h in records[ri]["hands"]):
                    continue
                w = (ri - ra) / (rb - ra)
                new = {"id": tid, "side": hb["side"], "bbox": _lerp(ha["bbox"], hb["bbox"], w),
                       "score": round(min(ha["score"], hb["score"]), 4),
                       "kp2d": _lerp(ha["kp2d"], hb["kp2d"], w), "interpolated": True}
                for k in ("kp3d", "kp3d_root_cam"):
                    if k in ha and k in hb:
                        new[k] = _lerp(ha[k], hb[k], w)
                for k in ("wearer_score", "is_wearer"):
                    if k in hb:
                        new[k] = hb[k]
                records[ri]["hands"].append(new)
                records[ri]["hands"].sort(key=lambda h: h["id"])
                filled += 1
    return filled


def renumber(records: list[dict]) -> int:
    new_id: dict = {}
    for r in records:
        for h in r["hands"]:
            if h["id"] not in new_id:
                new_id[h["id"]] = len(new_id) + 1
            h["id"] = new_id[h["id"]]
    return len(new_id)


def postprocess(records: list[dict], cfg: PostConfig) -> tuple[list[dict], dict]:
    """Run all steps on a copy of `records` (must be sorted by frame). Returns (records, report)."""
    recs = copy.deepcopy(sorted(records, key=lambda r: r["frame"]))
    ids_in = len(_tracks(recs))
    rep = {"config": asdict(cfg), "frames": len(recs), "ids_in": ids_in,
           "hands_in": sum(len(r["hands"]) for r in recs)}
    rep["duplicates_removed"] = dedup(recs, cfg)
    rep["wearer"] = filter_wearer(recs, cfg)
    rep["side_changes"] = resolve_sides(recs, cfg)
    rep["relinks"] = relink(recs, cfg) if cfg.relink else 0
    rep.update(cap_per_side(recs, cfg))
    rep["gap_fills"] = fill_gaps(recs, cfg)
    rep["ids_out"] = renumber(recs) if cfg.renumber else len(_tracks(recs))
    rep["hands_out"] = sum(len(r["hands"]) for r in recs)
    n = max(1, len(recs))
    rep["frames_with_ge1_hand"] = round(sum(bool(r["hands"]) for r in recs) / n, 4)
    rep["frames_with_2_hands"] = round(sum(len(r["hands"]) == 2 for r in recs) / n, 4)
    rep["frames_with_gt2_hands"] = sum(len(r["hands"]) > 2 for r in recs)
    return recs, rep
