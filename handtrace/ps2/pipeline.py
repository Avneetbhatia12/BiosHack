"""PS2 pass for one clip, on top of an existing PS1 hands.jsonl (WiLoR is never re-run).

  1. load + check PS1 records       consecutive frames, t == frame / fps (within tolerance)
  2. object detection (GPU)         -> ps2/object_detections.jsonl (+ _meta.json); cached and reused
                                       when the detection settings are unchanged (per-clip resume)
  3. contact analysis (CPU, fast)   -> ps2/contacts.jsonl (per frame and hand), ps2/contacts.csv
                                       (one row per confirmed interval)
  4. metrics                        -> ps2/ps2_metrics.json
  5. annotated video (optional)     -> ps2/annotated_ps2.mp4   (PS1 annotated video is untouched)
Frames are streamed; nothing holds the whole video in memory. Step 3 can be re-run alone with
`run_contacts_only` after changing contact thresholds.
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Dict, List, Optional

from .config import ObjectVocabulary, PS2Config
from .contact_detector import CSV_COLUMNS, analyze_contacts, event_rows
from .object_tracker import ObjectTracker

DISCLAIMER = ("Contacts are estimated from 2D proximity of PS1 fingertip keypoints to detected object boxes "
              "(>= N fingertips inside the padded box for >= M frames). They are visual estimates of touching, "
              "not verified physical grasps, and object names come from an object detector that can be wrong. "
              "No ground truth was used for these numbers.")


class AlignmentError(ValueError):
    pass


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path: str, rows) -> None:
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, Counter):
        return dict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(type(o))


def check_hand_records(records: List[dict], fps: float, tol_frames: float = 0.5) -> None:
    """Fail clearly unless frames are consecutive and t matches frame / fps."""
    if not records:
        raise AlignmentError("PS1 hands.jsonl is empty")
    f0 = records[0]["frame"]
    for k, r in enumerate(records):
        if r["frame"] != f0 + k:
            raise AlignmentError(f"PS1 frames not consecutive at line {k + 1}: expected {f0 + k}, got {r['frame']}")
        if abs(r["t"] - r["frame"] / fps) > tol_frames / fps:
            raise AlignmentError(f"PS1 frame {r['frame']}: t={r['t']} but frame/fps={r['frame'] / fps:.4f} "
                                 f"(fps {fps}); is this hands.jsonl from a different video?")


def check_object_records(hand_records: List[dict], obj_records: List[dict], tol_s: float) -> None:
    if len(obj_records) != len(hand_records):
        raise AlignmentError(f"{len(obj_records)} object records vs {len(hand_records)} hand records")
    for h, o in zip(hand_records, obj_records):
        if h["frame"] != o["frame"] or abs(h["t"] - o["t"]) > tol_s:
            raise AlignmentError(f"object frame {o['frame']} t={o['t']} does not match hand frame {h['frame']} t={h['t']}")


def default_output_dir(cfg: PS2Config) -> str:
    return cfg.output_dir or os.path.join(os.path.dirname(os.path.abspath(cfg.hands_jsonl)), "ps2")


# ----------------------------------------------------------------------------- step 2
def run_object_detection(cfg: PS2Config, vocab: ObjectVocabulary, hand_records: List[dict], detector,
                         out_dir: str, log=print) -> dict:
    from handtrace.wilor.video_loader import VideoReader, probe
    path = os.path.join(out_dir, "object_detections.jsonl")
    meta_path = os.path.join(out_dir, "object_detections_meta.json")
    fp = cfg.detection_fingerprint(vocab.profile)
    if not cfg.overwrite_detections and os.path.exists(path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("fingerprint") == fp and meta.get("frames") == len(hand_records):
            log(f"[ps2] reusing cached object detections ({meta['frames']} frames, fingerprint {fp})")
            return {"path": path, "meta": meta, "cached": True}
    if detector is None:
        raise RuntimeError("object detections are not cached for these settings; a detector is required")
    info = probe(cfg.video_path)
    W, H = info.width, info.height
    tracker = ObjectTracker((W, H), cfg.obj_iou_gate, cfg.obj_center_gate, cfg.obj_max_missed, cfg.obj_type_mismatch)
    by_frame = {r["frame"]: r for r in hand_records}
    f0, n = hand_records[0]["frame"], len(hand_records)
    tol = cfg.t_tolerance_frames / info.fps
    tmp = path + ".partial"
    det_s, n_det_frames, n_dets, last, done = 0.0, 0, 0, [], 0
    t_start = time.perf_counter()
    batch: list = []

    def flush(f):
        nonlocal det_s, n_det_frames, n_dets, last
        todo = [b for b in batch if b[2]]
        results = detector([img for _, _, _, img in todo]) if todo else []
        det_s += getattr(detector, "last_time_s", 0.0)
        res_iter = iter(results)
        for fidx, t, do_det, _ in batch:
            if do_det:
                objs = tracker.update(next(res_iter), fidx)
                n_det_frames += 1; n_dets += len(objs)
                last = objs
            else:
                objs = [{**o, "carried": True} for o in last]
            f.write(json.dumps({"frame": fidx, "t": by_frame[fidx]["t"], "detected": do_det, "objects": objs},
                               separators=(",", ":")) + "\n")
        batch.clear()

    with open(tmp, "w", encoding="utf-8") as f:
        for fidx, t, img in VideoReader(cfg.video_path, start=f0, max_frames=n):
            if fidx not in by_frame:
                raise AlignmentError(f"video frame {fidx} has no PS1 record")
            if abs(by_frame[fidx]["t"] - t) > tol:
                raise AlignmentError(f"frame {fidx}: video t={t:.4f} vs PS1 t={by_frame[fidx]['t']}")
            batch.append((fidx, t, (fidx - f0) % cfg.det_every == 0, img))
            done += 1
            if len(batch) >= cfg.det_batch:
                flush(f)
            if done % cfg.progress_every == 0:
                el = time.perf_counter() - t_start
                log(f"[ps2] objects {done}/{n} frames | {done / el:.2f} fps | ETA {(n - done) / max(done / el, 1e-6) / 60:.1f} min")
        if batch:
            flush(f)
    if done != n:
        os.remove(tmp)
        raise AlignmentError(f"video ended after {done} frames but PS1 has {n} records")
    os.replace(tmp, path)
    wall = time.perf_counter() - t_start
    meta = {"fingerprint": fp, "profile": vocab.profile, "frames": n, "detected_frames": n_det_frames, "detections": n_dets,
            "object_ids": tracker.next_id - 1, "detector_s": round(det_s, 2), "wall_s": round(wall, 2),
            "detector_fps": round(n_det_frames / det_s, 2) if det_s else None,
            "pass_fps": round(n / wall, 2) if wall else None, "video": {"width": W, "height": H, "fps": info.fps},
            "detector": detector.describe() if hasattr(detector, "describe") else str(detector)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"[ps2] object detection done: {n} frames in {wall / 60:.1f} min, {n_dets} detections, "
        f"{meta['object_ids']} object IDs")
    return {"path": path, "meta": meta, "cached": False}


# ----------------------------------------------------------------------------- step 3-4
def run_contacts_only(cfg: PS2Config, hand_records: List[dict], obj_records: List[dict], clip: str,
                      out_dir: str, det_meta: Optional[dict] = None, log=print) -> dict:
    t0 = time.perf_counter()
    if cfg.max_frames:
        hand_records = hand_records[: cfg.max_frames]
    fps = (det_meta or {}).get("video", {}).get("fps") or 30.0
    check_object_records(hand_records, obj_records, cfg.t_tolerance_frames / fps)
    vocab = ObjectVocabulary.load(cfg.prompts_path, cfg.profile, clip)
    res = analyze_contacts(hand_records, obj_records, cfg, vocab.co_contact)
    rows = event_rows(res["events"], clip)
    _write_jsonl(os.path.join(out_dir, "contacts.jsonl"), res["frames"])
    csv_path = os.path.join(out_dir, "contacts.csv")
    with open(csv_path + ".partial", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(csv_path + ".partial", csv_path)
    contact_s = time.perf_counter() - t0
    metrics = compute_metrics(hand_records, obj_records, res, rows, clip, det_meta, contact_s, cfg)
    with open(os.path.join(out_dir, "ps2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=_json_default)
    log(f"[ps2] contacts: {len(rows)} intervals ({metrics['portable_contacts']} portable, "
        f"{metrics['fixed_contacts']} fixed, {metrics['unknown_contacts']} unknown), "
        f"{res['rejected_candidates']} short candidate runs rejected")
    return {"frames": res["frames"], "events": res["events"], "rows": rows, "metrics": metrics, "csv": csv_path}


def compute_metrics(hand_records, obj_records, res, rows, clip, det_meta, contact_s, cfg) -> dict:
    by_name, by_type, ids = Counter(), Counter(), set()
    carried = 0
    for o in obj_records:
        carried += not o.get("detected", True)
        if not o.get("detected", True):
            continue
        for d in o["objects"]:
            by_name[d["object_name"]] += 1; by_type[d["object_type"]] += 1; ids.add(d["object_id"])
    dur, count = defaultdict(float), Counter()
    for r in rows:
        dur[r["object_name"]] += r["duration_s"]; count[r["object_name"]] += 1
    invalid = sum(1 for r in hand_records for h in r["hands"] if not h.get("kp2d") or len(h["kp2d"]) < 21)
    in_contact = sum(1 for fr in res["frames"] if any(h.get("contacts") for h in fr["hands"]))
    det_s = (det_meta or {}).get("wall_s", 0.0) or 0.0
    return {
        "disclaimer": DISCLAIMER, "clip": clip, "total_frames": len(hand_records),
        "object_detector": (det_meta or {}).get("detector"),
        "object_detections": int(sum(by_name.values())), "unique_object_ids": len(ids),
        "detections_by_name": dict(by_name.most_common()), "detections_by_type": dict(by_type),
        "frames_with_any_object": sum(1 for o in obj_records if o["objects"]),
        "contact_intervals": len(rows),
        "portable_contacts": sum(r["object_type"] == "portable" for r in rows),
        "fixed_contacts": sum(r["object_type"] == "fixed" for r in rows),
        "unknown_contacts": sum(r["object_type"] == "unknown" for r in rows),
        "grab_release_intervals": sum(r["start_event"] == "grab" and r["end_event"] == "release" for r in rows),
        "contact_duration_s_by_object": {k: round(v, 2) for k, v in sorted(dur.items(), key=lambda x: -x[1])},
        "most_contacted_objects": count.most_common(10),
        "frames_with_hand_in_contact": in_contact,
        "rejected_short_candidate_runs": res["rejected_candidates"],
        "intervals_with_interpolated_support": sum("interpolated_hand" in r["uncertainty"] for r in rows),
        "skipped_or_invalid": {"hands_without_21_kp2d": invalid, "frames_with_carried_objects": carried},
        "timing": {"object_detection_wall_s": det_s, "object_detector_fps": (det_meta or {}).get("detector_fps"),
                   "object_pass_fps": (det_meta or {}).get("pass_fps"),
                   "detections_cached": bool((det_meta or {}).get("_cached")),
                   "contact_analysis_s": round(contact_s, 3)},
        "config": cfg.to_dict(),
    }


# ----------------------------------------------------------------------------- step 5
def render_ps2(cfg: PS2Config, hand_records: List[dict], obj_records: List[dict], contact_frames: List[dict],
               out_dir: str, crf: int = 23, log=print) -> str:
    from handtrace.wilor.video_loader import VideoReader, VideoWriter, probe
    from .visualization import draw_ps2
    info = probe(cfg.video_path)
    writer = VideoWriter(os.path.join(out_dir, "annotated_ps2.mp4"), info.fps, (info.width, info.height), crf)
    f0, n = hand_records[0]["frame"], len(hand_records)
    for k, (fidx, _, img) in enumerate(VideoReader(cfg.video_path, start=f0, max_frames=n)):
        writer.write(draw_ps2(img, hand_records[k], obj_records[k], contact_frames[k]))
    path = writer.close()
    log(f"[ps2] annotated video -> {path}")
    return path


def release_detector(detector) -> None:
    """Free GPU memory held by an object detector."""
    if detector is not None and hasattr(detector, "close"):
        detector.close()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_ps2(cfg: PS2Config, detector=None, device: Optional[str] = None, log=print) -> Dict:
    """One clip end to end. `detector` may be passed in (built once for many clips); otherwise it is
    built only if detections are not already cached."""
    from handtrace.wilor.video_loader import probe
    from .object_detector import build_object_detector
    t0 = time.perf_counter()
    clip = os.path.splitext(os.path.basename(cfg.video_path))[0]
    vocab = ObjectVocabulary.load(cfg.prompts_path, cfg.profile, clip)
    log(f"[ps2] {clip}: object profile '{vocab.profile or 'flat'}' ({len(vocab.prompts)} objects)")
    out_dir = default_output_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    info = probe(cfg.video_path)
    hands = _read_jsonl(cfg.hands_jsonl)
    check_hand_records(hands, info.fps, cfg.t_tolerance_frames)
    if cfg.max_frames:
        hands = hands[: cfg.max_frames]
    if hands[-1]["frame"] >= info.frame_count > 0:
        raise AlignmentError(f"PS1 has frame {hands[-1]['frame']} but the video reports {info.frame_count} frames")
    with open(os.path.join(out_dir, "ps2_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    built = False
    meta_path = os.path.join(out_dir, "object_detections_meta.json")
    cached_ok = (not cfg.overwrite_detections and os.path.exists(meta_path)
                 and json.load(open(meta_path)).get("fingerprint") == cfg.detection_fingerprint(vocab.profile)
                 and json.load(open(meta_path)).get("frames") == len(hands))
    if detector is None and not cached_ok:
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = build_object_detector(cfg, vocab, device)
        built = True
    elif detector is not None and hasattr(detector, "set_vocabulary"):
        detector.set_vocabulary(vocab)                  # shared detector, this clip's profile
    det = run_object_detection(cfg, vocab, hands, detector, out_dir, log)
    if built:
        release_detector(detector)
    objs = _read_jsonl(det["path"])
    meta = {**det["meta"], "_cached": det["cached"]}
    res = run_contacts_only(cfg, hands, objs, clip, out_dir, meta, log)
    video = render_ps2(cfg, hands, objs, res["frames"], out_dir, log=log) if cfg.write_video else None
    total = time.perf_counter() - t0
    res["metrics"]["timing"]["total_ps2_s"] = round(total, 2)
    with open(os.path.join(out_dir, "ps2_metrics.json"), "w") as f:
        json.dump(res["metrics"], f, indent=1, default=_json_default)
    return {"out_dir": out_dir, "object_detections": det["path"], "contacts_jsonl": os.path.join(out_dir, "contacts.jsonl"),
            "contacts_csv": res["csv"], "metrics": res["metrics"], "video": video, "events": res["rows"]}


def rerun_contacts(cfg: PS2Config, log=print, **overrides) -> Dict:
    """Contact analysis again on cached detections with new contact settings (no GPU)."""
    from handtrace.wilor.video_loader import probe
    cfg = replace(cfg, **overrides)
    out_dir = default_output_dir(cfg)
    info = probe(cfg.video_path)
    hands = _read_jsonl(cfg.hands_jsonl)
    check_hand_records(hands, info.fps, cfg.t_tolerance_frames)
    if cfg.max_frames:
        hands = hands[: cfg.max_frames]
    meta = json.load(open(os.path.join(out_dir, "object_detections_meta.json")))
    objs = _read_jsonl(os.path.join(out_dir, "object_detections.jsonl"))
    clip = os.path.splitext(os.path.basename(cfg.video_path))[0]
    return run_contacts_only(cfg, hands, objs, clip, out_dir, {**meta, "_cached": True}, log)
