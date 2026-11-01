"""Coverage / speed / tracking-stability report.

IMPORTANT: there is no ground truth for this dataset. Everything here measures COVERAGE (how
often the system outputs a hand), SPEED and SELF-CONSISTENCY - not precision, recall or keypoint
accuracy. Those require manually annotated frames (see `annotation_template`)."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

DISCLAIMER = ("Coverage and stability statistics only. No ground-truth labels were used, so these "
              "numbers are NOT detection precision/recall or keypoint accuracy.")


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_metrics(records: List[Dict], timing: Dict, tracker_events: Optional[Dict] = None,
                    frame_wh=(1920, 1080), fps: float = 30.0) -> Dict:
    n = len(records)
    diag = float(np.hypot(*frame_wh))
    obs_counts = [sum(1 for h in r["hands"] if not h.get("interpolated")) for r in records]
    all_counts = [len(r["hands"]) for r in records]
    sides = defaultdict(int)
    scores, in_box = [], []
    track_frames = defaultdict(list)
    lr_ok = lr_total = 0
    for r in records:
        lefts = [h for h in r["hands"] if h["side"] == "left"]
        rights = [h for h in r["hands"] if h["side"] == "right"]
        if len(lefts) == 1 and len(rights) == 1:
            lr_total += 1
            lr_ok += int(lefts[0]["kp2d"][0][0] < rights[0]["kp2d"][0][0])
        for h in r["hands"]:
            track_frames[h["id"]].append((r["frame"], h["side"], h["kp2d"][0]))
            if h.get("interpolated"):
                continue
            sides[h["side"]] += 1
            scores.append(h["score"])
            x1, y1, x2, y2 = h["bbox"]
            pw, ph = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
            kp = np.asarray(h["kp2d"])
            in_box.append(float(np.mean((kp[:, 0] >= x1 - pw) & (kp[:, 0] <= x2 + pw)
                                        & (kp[:, 1] >= y1 - ph) & (kp[:, 1] <= y2 + ph))))

    # --- track stability (approximate, label-free) ---
    gaps, side_changes, lengths = 0, 0, []
    spans = {}
    for tid, fr in track_frames.items():
        fr.sort()
        frames = [f for f, _, _ in fr]
        lengths.append(len(frames))
        gaps += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
        side_changes += sum(1 for (_, s1, _), (_, s2, _) in zip(fr, fr[1:]) if s1 != s2)
        spans[tid] = (fr[0], fr[-1])
    # fragmentation: a new ID appears shortly after another ID of the same side vanished nearby
    frag = 0
    ends = sorted(((e[0], tid, e[1], e[2]) for tid, (_, e) in spans.items()))
    for tid, (s, _) in spans.items():
        f0, side0, p0 = s
        for fe, tid_e, side_e, pe in ends:
            if tid_e == tid or side_e != side0:
                continue
            if 0 < f0 - fe <= int(fps) and np.hypot(p0[0] - pe[0], p0[1] - pe[1]) < 0.15 * diag:
                frag += 1
                break

    proc = max(1, timing.get("processed_frames", n))
    det_t, rec_t, wall = timing.get("detector_s", 0.0), timing.get("wilor_s", 0.0), timing.get("wall_s", 0.0)
    m = {
        "disclaimer": DISCLAIMER,
        "total_frames": n,
        "processed_frames": timing.get("processed_frames", n),
        "frames_with_ge1_hand_observed": int(sum(c >= 1 for c in obs_counts)),
        "frames_with_2_hands_observed": int(sum(c == 2 for c in obs_counts)),
        "frames_with_gt2_hands_observed": int(sum(c > 2 for c in obs_counts)),
        "frames_with_ge1_hand_incl_interp": int(sum(c >= 1 for c in all_counts)),
        "total_hand_detections": int(sum(obs_counts)),
        "interpolated_hand_instances": int(sum(all_counts) - sum(obs_counts)),
        "detections_by_side": dict(sides),
        "frame_detection_rate": round(sum(c >= 1 for c in obs_counts) / max(1, n), 4),
        "two_hand_frame_rate": round(sum(c == 2 for c in obs_counts) / max(1, n), 4),
        "mean_detection_score": round(float(np.mean(scores)), 4) if scores else None,
        "low_score_detections_lt_0.4": int(sum(s < 0.4 for s in scores)),
        "avg_inference_ms_per_processed_frame": round(1000 * (det_t + rec_t) / proc, 2),
        "avg_detector_ms_per_processed_frame": round(1000 * det_t / proc, 2),
        "avg_wilor_ms_per_hand": round(1000 * rec_t / max(1, sum(obs_counts)), 2),
        "detector_only_fps": round(proc / det_t, 2) if det_t else None,
        "end_to_end_fps": round(n / wall, 2) if wall else None,
        "total_processing_time_s": round(wall, 1),
        "tracking_ids_generated": len(track_frames),
        "mean_track_length_frames": round(float(np.mean(lengths)), 1) if lengths else 0,
        "approx_id_fragmentations": frag,
        "approx_side_label_changes_within_track": side_changes,
        "approx_track_interruptions": gaps,
        "kp2d_inside_padded_bbox_ratio": round(float(np.mean(in_box)), 4) if in_box else None,
        "handedness_spatial_consistency": (round(lr_ok / lr_total, 4) if lr_total else None),
        "handedness_spatial_frames": lr_total,
    }
    if tracker_events:
        m["tracker_births"] = len(tracker_events.get("births", []))
        m["tracker_recoveries"] = len(tracker_events.get("recoveries", []))
        m["tracker_side_corrections"] = int(tracker_events.get("side_corrections", 0))
    m["notes"] = {
        "approx_id_fragmentations": "new ID born <=1 s after a same-side ID ended within 15% of the "
                                    "frame diagonal (likely the same physical hand re-acquired).",
        "approx_track_interruptions": "gaps inside a track's output history (after gap filling).",
        "handedness_spatial_consistency": "fraction of frames with one L + one R hand where the left "
                                          "wrist is left of the right wrist. A heuristic for "
                                          "egocentric views (arms can cross); very low values suggest "
                                          "a systematic L/R swap.",
        "kp2d_inside_padded_bbox_ratio": "keypoints inside the detector box padded 25%; low values "
                                         "flag reconstruction failures.",
    }
    return m


def annotation_template(records: List[Dict], n: int = 100, seed: int = 0) -> List[Dict]:
    """Pick n frames (stratified over the clip) for manual labelling of
    {hand present? side? wrist/fingertip positions?} to measure real accuracy."""
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(records) - 1, n).astype(int)
    idx = np.unique(np.clip(idx + rng.integers(-5, 6, len(idx)), 0, len(records) - 1))
    return [{"frame": records[i]["frame"], "gt_hands": [], "pred": records[i]["hands"]} for i in idx]
