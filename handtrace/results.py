"""One place that knows where every stage writes its results, so PS5 and the review tools read them the same way.

OUT_ROOT/<stem>/
    hands.jsonl, hands_final.jsonl, metrics.json, hands_meta.json, postprocess_report.json      PS1
    ps2_final/object_detections.jsonl, contacts.jsonl, contacts.csv, ps2_metrics.json           PS2
    ps3/actions.json, actions_final.json, actions_final.csv                                      PS3
    ps5/quality.json, safety_events.csv                                                          PS5
    review/manual_audit.csv                                                                       manual review
Every loader returns an empty result (never raises) when a stage has not run yet, so any subset of PS1/PS2/PS3
can be inspected.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional


def clip_dir(out_root: str, stem: str) -> str:
    return os.path.join(out_root, stem)


def paths(out_root: str, stem: str) -> Dict[str, str]:
    d = clip_dir(out_root, stem)
    p2, p3, p5 = os.path.join(d, "ps2_final"), os.path.join(d, "ps3"), os.path.join(d, "ps5")
    return {
        "dir": d, "hands_raw": os.path.join(d, "hands.jsonl"), "hands_final": os.path.join(d, "hands_final.jsonl"),
        "metrics": os.path.join(d, "metrics.json"), "meta": os.path.join(d, "hands_meta.json"),
        "post": os.path.join(d, "postprocess_report.json"),
        "objects": os.path.join(p2, "object_detections.jsonl"), "contact_frames": os.path.join(p2, "contacts.jsonl"),
        "contacts": os.path.join(p2, "contacts.csv"), "ps2_metrics": os.path.join(p2, "ps2_metrics.json"),
        "actions": os.path.join(p3, "actions.json"), "actions_final": os.path.join(p3, "actions_final.json"),
        "ps5": p5, "quality": os.path.join(p5, "quality.json"), "safety": os.path.join(p5, "safety_events.csv"),
        "review": os.path.join(d, "review"), "audit": os.path.join(d, "review", "manual_audit.csv"),
    }


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict], fields: Optional[List[str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)
    return path


def load_hands(out_root: str, stem: str, final: bool = True) -> List[dict]:
    p = paths(out_root, stem)
    recs = read_jsonl(p["hands_final"] if final else p["hands_raw"])
    if not recs and final:
        recs = read_jsonl(p["hands_raw"])                 # post-processing not run yet: fall back to raw PS1
    return recs


def load_contacts(out_root: str, stem: str) -> List[dict]:
    """PS2 contact intervals (contacts.csv) with numeric times."""
    out = []
    for r in read_csv(paths(out_root, stem)["contacts"]):
        try:
            r = dict(r)
            r["t_start"], r["t_end"] = float(r["t_start"]), float(r["t_end"])
            r["object_name"] = (r.get("object_name") or "unknown").replace("_", " ")
            out.append(r)
        except (KeyError, ValueError):
            continue
    return out


def load_actions(out_root: str, stem: str) -> List[dict]:
    """PS3 actions: the PS2-refined timeline when it exists, else the model timeline."""
    p = paths(out_root, stem)
    doc = read_json(p["actions_final"]) or read_json(p["actions"])
    return list(doc.get("actions", []))


def video_info(out_root: str, stem: str, clip_path: Optional[str] = None) -> dict:
    """{"fps", "width", "height", "frames", "duration_s"} from PS1 metadata, else from the video itself."""
    meta = read_json(paths(out_root, stem)["meta"]).get("video", {})
    if meta.get("fps"):
        fps = float(meta["fps"])
        n = int(meta.get("frame_count") or meta.get("frames") or 0)
        return {"fps": fps, "width": int(meta.get("width", 0)), "height": int(meta.get("height", 0)), "frames": n,
                "duration_s": n / fps if n else 0.0}
    if clip_path:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = {"fps": float(fps), "width": int(cap.get(3)), "height": int(cap.get(4)), "frames": n,
               "duration_s": n / fps if fps else 0.0}
        cap.release()
        return out
    return {"fps": 30.0, "width": 0, "height": 0, "frames": 0, "duration_s": 0.0}


def stages_done(out_root: str, stem: str) -> Dict[str, bool]:
    p = paths(out_root, stem)
    return {"ps1": os.path.exists(p["hands_final"]) or os.path.exists(p["hands_raw"]),
            "ps2": os.path.exists(p["contacts"]), "ps3": os.path.exists(p["actions"]),
            "ps5_quality": os.path.exists(p["quality"]), "ps5_safety": os.path.exists(p["safety"])}
