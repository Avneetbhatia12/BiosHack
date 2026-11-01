"""PS2 manual evaluation: annotation template, frame-level contact scoring, event timing error.

    # 1. sample ~50 hand-frames (half predicted contact, half not) into a sheet to fill by hand
    python eval/contact_metrics.py template --contacts <ps2>/contacts.jsonl --n 50 --out eval/labels/<clip>_contacts.csv
    # 2. fill gt_state (none / portable / fixed / unknown_object) and gt_object_name, then:
    python eval/contact_metrics.py score --sheet eval/labels/<clip>_contacts.csv
    # 3. optional: hand-made event list (clip, hand_id, object_name, t_start, t_end) vs contacts.csv
    python eval/contact_metrics.py timing --pred <ps2>/contacts.csv --gt eval/labels/<clip>_events.csv

Labels stay in eval/labels/ (gitignored). Report numbers only from filled sheets, with the count.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

SHEET_COLUMNS = ["clip", "frame", "t", "hand_id", "side", "interpolated", "pred_state", "pred_object_id",
                 "pred_object_name", "pred_fingertips", "gt_state", "gt_object_name", "notes"]
GT_STATES = {"none", "portable", "fixed", "unknown_object"}


def make_template(contact_frames: list, clip: str, n: int = 50, seed: int = 0) -> list:
    pos, neg = [], []
    for fr in contact_frames:
        for h in fr["hands"]:
            if h.get("state") == "not_analyzed":
                continue
            c = (h.get("contacts") or [{}])[0]
            row = {"clip": clip, "frame": fr["frame"], "t": fr["t"], "hand_id": h["hand_id"], "side": h.get("side"),
                   "interpolated": h.get("interpolated"), "pred_state": h["state"],
                   "pred_object_id": c.get("object_id", ""), "pred_object_name": c.get("object_name", ""),
                   "pred_fingertips": c.get("fingertip_count", ""), "gt_state": "", "gt_object_name": "", "notes": ""}
            (pos if h["state"].startswith("touching") else neg).append(row)
    rng = random.Random(seed)
    k = min(len(pos), n // 2)
    rows = rng.sample(pos, k) + rng.sample(neg, min(len(neg), n - k))
    return sorted(rows, key=lambda r: (r["frame"], str(r["hand_id"])))


def score_sheet(rows: list) -> dict:
    rows = [r for r in rows if r.get("gt_state", "").strip()]
    bad = [r["gt_state"] for r in rows if r["gt_state"].strip() not in GT_STATES]
    if bad:
        raise SystemExit(f"unknown gt_state values {sorted(set(bad))}; use {sorted(GT_STATES)}")
    tp = fp = fn = tn = type_ok = name_ok = name_n = 0
    for r in rows:
        pred = r["pred_state"].startswith("touching")
        gt = r["gt_state"].strip() != "none"
        tp += pred and gt; fp += pred and not gt; fn += gt and not pred; tn += not pred and not gt
        if pred and gt:
            pt = r["pred_state"].replace("touching_", "")
            type_ok += pt == r["gt_state"].strip() or (pt == "unknown" and r["gt_state"].strip() == "unknown_object")
            if r.get("gt_object_name", "").strip():
                name_n += 1
                name_ok += r["pred_object_name"].strip().lower() == r["gt_object_name"].strip().lower()
    return {"labelled_hand_frames": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "contact_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "contact_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "type_accuracy_on_tp": round(type_ok / tp, 4) if tp else None,
            "object_name_accuracy_on_tp": round(name_ok / name_n, 4) if name_n else None,
            "object_names_checked": name_n}


def timing_error(pred: list, gt: list, min_overlap: float = 0.3, tolerance_s: float = 0.5) -> dict:
    """Match GT events to predicted intervals of the same hand by temporal IoU (greedy). The hand is
    matched by hand_id when both sides have one, else by side (`hand`: left / right, as written by
    eval/label_events.py). Reports the judges' criterion: start and end within +-tolerance_s."""
    def same_hand(g, p):
        if g.get("hand_id") not in (None, "") and p.get("hand_id") not in (None, ""):
            return str(g["hand_id"]) == str(p["hand_id"])
        return str(g.get("hand", "")).lower() == str(p.get("hand", "")).lower()

    def tiou(a, b):
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if same_hand(g, p):
                v = tiou((float(g["t_start"]), float(g["t_end"])), (float(p["t_start"]), float(p["t_end"])))
                if v >= min_overlap:
                    pairs.append((v, gi, pi))
    used_g, used_p, ds, de, names = set(), set(), [], [], []
    for v, gi, pi in sorted(pairs, reverse=True):
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        ds.append(abs(float(pred[pi]["t_start"]) - float(gt[gi]["t_start"])))
        de.append(abs(float(pred[pi]["t_end"]) - float(gt[gi]["t_end"])))
        if gt[gi].get("object_name"):
            names.append(gt[gi]["object_name"].strip().lower() == pred[pi].get("object_name", "").strip().lower())
    mean = lambda x: round(sum(x) / len(x), 3) if x else None
    return {"gt_events": len(gt), "pred_events": len(pred), "matched": len(used_g),
            "event_recall": round(len(used_g) / len(gt), 4) if gt else None,
            "event_precision": round(len(used_p) / len(pred), 4) if pred else None,
            "mean_abs_start_error_s": mean(ds), "mean_abs_end_error_s": mean(de), "min_temporal_iou": min_overlap,
            f"start_within_{tolerance_s}s": mean([d <= tolerance_s for d in ds]),
            f"end_within_{tolerance_s}s": mean([d <= tolerance_s for d in de]),
            "object_name_accuracy": mean(names), "object_names_checked": len(names)}


def _read_csv(p) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--contacts", type=Path, required=True)
    t.add_argument("--clip", default=None)
    t.add_argument("--n", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--sheet", type=Path, required=True)
    m = sub.add_parser("timing")
    m.add_argument("--pred", type=Path, required=True)
    m.add_argument("--gt", type=Path, required=True)
    m.add_argument("--min-overlap", type=float, default=0.3)
    a = ap.parse_args(argv)
    if a.cmd == "template":
        frames = [json.loads(l) for l in a.contacts.read_text(encoding="utf-8").splitlines() if l.strip()]
        clip = a.clip or a.contacts.parent.parent.name
        rows = make_template(frames, clip, a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            w.writeheader(); w.writerows(rows)
        print(f"[contact-eval] {len(rows)} rows -> {a.out}. gt_state: none | portable | fixed | unknown_object")
    elif a.cmd == "score":
        print(json.dumps(score_sheet(_read_csv(a.sheet)), indent=1))
    else:
        print(json.dumps(timing_error(_read_csv(a.pred), _read_csv(a.gt), a.min_overlap), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
