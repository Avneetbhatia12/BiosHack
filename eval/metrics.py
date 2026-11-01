"""Score a hands.jsonl against hand-made ground-truth labels (from eval/label.py).

Label file (eval/labels/<video_stem>.json, gitignored, private):
    {"video": "clip.mp4", "width": 854, "height": 480,
     "frames": {"120": {"split": "dev",
                        "hands": [{"bbox": [x1, y1, x2, y2], "side": "left", "wearer": true,
                                   "kp": {"0": [x, y], "4": [x, y], ...}}]}}}
Labels may come from clips_480p; they are scaled to the prediction resolution.

Metrics (wearer hands only; background-worker hands count as false positives if predicted):
    recall, precision        box match at IoU >= --iou (greedy, highest IoU first)
    side_accuracy            on matched hands
    pck                      labelled keypoints within alpha * max(gt box w, h) of the prediction
    mean_kp_error_px         at prediction resolution
    id_switches              the wearer's left (right) hand is one physical hand, so the GT side is
                             its identity: a switch = its matched predicted ID changes between two
                             labelled frames <= --segment-gap frames apart
Every report states the frame count. These are the only numbers to call "accuracy".

    python eval/metrics.py --labels eval/labels/clip.json --pred outputs/wilor/clip/hands_final.jsonl
    python eval/metrics.py --labels-dir eval/labels --pred-root outputs/wilor --pred-name hands_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KP_IDS = (0, 4, 8, 12, 16, 20)       # wrist + fingertips, the points the labeller clicks


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def scale_labels(labels: dict, width: int, height: int) -> dict:
    """Return a copy of `labels` in a (width, height) pixel space."""
    sx, sy = width / labels["width"], height / labels["height"]
    out = {**labels, "width": width, "height": height, "frames": {}}
    for f, fr in labels["frames"].items():
        hands = []
        for h in fr["hands"]:
            b = h["bbox"]
            hands.append({**h, "bbox": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                          "kp": {k: [p[0] * sx, p[1] * sy] for k, p in h.get("kp", {}).items()}})
        out["frames"][f] = {**fr, "hands": hands}
    return out


def _match(gt: list[dict], pred: list[dict], thr: float) -> list[tuple[int, int, float]]:
    pairs = sorted(((iou(g["bbox"], p["bbox"]), gi, pi) for gi, g in enumerate(gt)
                    for pi, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); out.append((gi, pi, v))
    return out


def score(labels: dict, records: list[dict], pred_wh: tuple[int, int] | None = None,
          iou_thr: float = 0.3, pck_alpha: float = 0.2, segment_gap: int = 15,
          include_interpolated: bool = True) -> dict:
    if pred_wh and (labels["width"], labels["height"]) != tuple(pred_wh):
        labels = scale_labels(labels, *pred_wh)
    by_frame = {r["frame"]: r for r in records}
    c = defaultdict(float)
    kp_err, kp_ok = [], []
    id_track = defaultdict(list)            # gt side -> [(frame, predicted id)]
    per_split = defaultdict(lambda: defaultdict(float))
    frames = sorted(int(f) for f in labels["frames"])
    missing = 0
    for f in frames:
        if f not in by_frame:
            missing += 1
            continue
        fr = labels["frames"][str(f)]
        split = fr.get("split", "all")
        gt = [h for h in fr["hands"] if h.get("wearer", True)]
        bg = [h for h in fr["hands"] if not h.get("wearer", True)]
        pred = [h for h in by_frame[f]["hands"] if include_interpolated or not h.get("interpolated")]
        m = _match(gt, pred, iou_thr)
        mp = {pi for _, pi, _ in m}
        fp = [p for pi, p in enumerate(pred) if pi not in mp]
        fp_bg = sum(any(iou(p["bbox"], b["bbox"]) >= iou_thr for b in bg) for p in fp)
        side_ok = sum(gt[gi]["side"] == pred[pi]["side"] for gi, pi, _ in m)
        for d in (c, per_split[split]):
            d["frames"] += 1; d["gt"] += len(gt); d["tp"] += len(m); d["fn"] += len(gt) - len(m)
            d["fp_background"] += fp_bg; d["fp_other"] += len(fp) - fp_bg; d["side_ok"] += side_ok
        for gi, pi, _ in m:
            g, p = gt[gi], pred[pi]
            id_track[g["side"]].append((f, p["id"]))
            norm = max(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1], 1.0)
            for k, xy in g.get("kp", {}).items():
                e = float(np.hypot(p["kp2d"][int(k)][0] - xy[0], p["kp2d"][int(k)][1] - xy[1]))
                kp_err.append(e); kp_ok.append(e <= pck_alpha * norm)
    switches = pairs = 0
    for seq in id_track.values():
        for (fa, ia), (fb, ib) in zip(seq, seq[1:]):
            if fb - fa <= segment_gap:
                pairs += 1; switches += ia != ib

    def summary(d):
        tp, gt, fp = d["tp"], d["gt"], d["fp_background"] + d["fp_other"]
        return {"frames": int(d["frames"]), "gt_wearer_hands": int(gt), "tp": int(tp), "fn": int(d["fn"]),
                "fp_background": int(d["fp_background"]), "fp_other": int(d["fp_other"]),
                "recall": round(tp / gt, 4) if gt else None,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "side_accuracy": round(d["side_ok"] / tp, 4) if tp else None}

    out = summary(c)
    out.update({"pck": round(float(np.mean(kp_ok)), 4) if kp_ok else None, "pck_alpha": pck_alpha,
                "keypoints_scored": len(kp_ok),
                "mean_kp_error_px": round(float(np.mean(kp_err)), 2) if kp_err else None,
                "id_switches": switches, "id_pairs_checked": pairs,
                "labelled_frames_missing_in_pred": missing, "iou_threshold": iou_thr,
                "by_split": {k: summary(v) for k, v in per_split.items()}})
    return out


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pred_size(pred: Path, override: str | None) -> tuple[int, int] | None:
    if override:
        w, h = override.lower().split("x")
        return int(w), int(h)
    meta = pred.parent / "hands_meta.json"            # written by the WiLoR pipeline
    if meta.exists():
        v = json.loads(meta.read_text())["video"]
        return int(v["width"]), int(v["height"])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, help="one label file")
    ap.add_argument("--pred", type=Path, help="hands.jsonl for that label file")
    ap.add_argument("--labels-dir", type=Path, help="score every <stem>.json in this folder ...")
    ap.add_argument("--pred-root", type=Path, help="... against <pred-root>/<stem>/<pred-name>")
    ap.add_argument("--pred-name", default="hands_final.jsonl")
    ap.add_argument("--pred-size", help="WxH of the predicted video, if hands_meta.json is absent")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--pck-alpha", type=float, default=0.2)
    ap.add_argument("--no-interpolated", action="store_true", help="ignore gap-filled hands")
    ap.add_argument("--out", type=Path, help="write the report JSON here")
    a = ap.parse_args(argv)
    if a.labels and a.pred:
        jobs = [(a.labels, a.pred)]
    elif a.labels_dir and a.pred_root:
        jobs = [(p, a.pred_root / p.stem / a.pred_name) for p in sorted(a.labels_dir.glob("*.json"))]
    else:
        ap.error("give --labels + --pred, or --labels-dir + --pred-root")
    report = {}
    for lab, pred in jobs:
        if not pred.exists():
            print(f"[metrics] missing prediction for {lab.stem}: {pred}", file=sys.stderr)
            report[lab.stem] = {"error": f"missing {pred}"}
            continue
        labels = json.loads(lab.read_text(encoding="utf-8"))
        report[lab.stem] = score(labels, _load_jsonl(pred), _pred_size(pred, a.pred_size), a.iou,
                                 a.pck_alpha, include_interpolated=not a.no_interpolated)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
