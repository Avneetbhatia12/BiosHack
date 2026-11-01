"""End-to-end: video -> official WiLoR detector -> tracker -> WiLoR reconstruction -> smoothing
-> gap filling -> hands.jsonl + annotated MP4 + metrics.json.

Per chunk of `chunk_size` frames:
  1. YOLO detector on all processed frames of the chunk (one batched call)
  2. tracker.step() frame by frame (bbox / centre / side cues only)  -> IDs + resolved side
  3. ONE WiLoR batch for every hand in the chunk, cropped with the resolved side
  4. per frame, in time order: One Euro smoothing per track ID, then a delayed writer that
     linearly interpolates short dropouts (<= max_interp_gap frames) before writing
Exactly one JSONL record is written for every frame of the clip (empty "hands" if none).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List

import cv2
import numpy as np

from .config import PipelineConfig
from .evaluation import compute_metrics
from .hand_tracker import HandTracker
from .smoothing import HandSmoother
from .video_loader import VideoReader, VideoWriter, probe
from .visualization import draw_record

COORDINATE_DOC = {
    "kp2d": "pixels in the ORIGINAL video frame, origin top-left, x right, y down. Obtained by "
            "projecting WiLoR's 21 3D joints with the official full-image camera "
            "(f = FOCAL_LENGTH/IMAGE_SIZE * max(W,H) = 5000/256*max(W,H), principal point = image "
            "centre). Points may lie outside the frame for partially visible hands.",
    "kp3d": {"wrist_relative": "metres, MANO scale, wrist (joint 0) at origin, axes aligned with the "
                               "camera: x right, y down, z away from camera.",
             "camera": "metres, WiLoR camera frame (x right, y down, z forward). Absolute depth "
                       "depends on WiLoR's fixed focal-length assumption and is NOT metric depth for "
                       "this fisheye camera; relative hand shape is reliable."},
    "kp3d_root_cam": "wrist position in the WiLoR camera frame (metres). camera-frame joints = "
                     "kp3d + kp3d_root_cam when kp3d is wrist_relative.",
    "bbox": "official WiLoR detector box, [x1, y1, x2, y2] original-frame pixels.",
    "score": "official detector confidence. Interpolated hands carry min(score before, after).",
    "side": "'left'/'right' from the detector class, optionally stabilised per track "
            "(side_mode=track_vote) and optionally globally swapped (handedness_swap).",
    "id": "persistent track ID from handtrace/wilor/hand_tracker.py.",
    "interpolated": "true when the hand was filled in by linear interpolation across a short "
                    "detection dropout or a skipped frame (not a model output).",
}
JOINT_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_MCP", "INDEX_PIP",
    "INDEX_DIP", "INDEX_TIP", "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP", "RING_MCP",
    "RING_PIP", "RING_DIP", "RING_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]


def _r(a, nd):
    return np.round(np.asarray(a, np.float64), nd).tolist()


class GapFillBuffer:
    """Holds the last `delay` frames so that when a track re-appears after a short dropout the
    missing frames can be filled by linear interpolation before they are written."""

    def __init__(self, delay: int, max_gap: int, enabled: bool, sink: Callable[[Dict], None]):
        self.delay, self.max_gap, self.enabled, self.sink = delay, max_gap, enabled, sink
        self.buf: Deque[Dict] = deque()
        self.last_seen: Dict[int, tuple] = {}
        self.filled = 0

    def push(self, rec: Dict):
        if self.enabled:
            for h in rec["hands"]:
                prev = self.last_seen.get(h["id"])
                if prev is not None:
                    f0, h0 = prev
                    gap = rec["frame"] - f0 - 1
                    if 1 <= gap <= self.max_gap and h0["side"] == h["side"]:
                        self._fill(f0, h0, rec["frame"], h)
        for h in rec["hands"]:
            self.last_seen[h["id"]] = (rec["frame"], h)
        self.buf.append(rec)
        while len(self.buf) > self.delay:
            self.sink(self.buf.popleft())

    def _fill(self, f0, h0, f1, h1):
        for r in self.buf:
            f = r["frame"]
            if f0 < f < f1 and not any(x["id"] == h1["id"] for x in r["hands"]):
                a = (f - f0) / (f1 - f0)
                lerp = lambda k: _r((1 - a) * np.asarray(h0[k]) + a * np.asarray(h1[k]), 5)
                hand = {"id": h1["id"], "side": h1["side"], "bbox": lerp("bbox"),
                        "score": round(min(h0["score"], h1["score"]), 4),
                        "kp2d": lerp("kp2d"), "kp3d": lerp("kp3d"), "interpolated": True}
                if "kp3d_root_cam" in h1 and "kp3d_root_cam" in h0:
                    hand["kp3d_root_cam"] = lerp("kp3d_root_cam")
                r["hands"].append(hand)
                r["hands"].sort(key=lambda x: x["id"])
                self.filled += 1

    def flush(self):
        while self.buf:
            self.sink(self.buf.popleft())


def _device():
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_video(cfg: PipelineConfig, detector=None, reconstructor=None, log=print) -> Dict:
    cfg = cfg.resolved()
    info = probe(cfg.video_path)
    W, H = info.width, info.height
    stem = os.path.splitext(os.path.basename(cfg.video_path))[0]
    out_dir = os.path.join(cfg.output_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))

    device = _device()
    if detector is None:
        from .wilor_detector import build_detector
        detector = build_detector(cfg, device)
    if reconstructor is None:
        from .wilor_inference import WiLoRReconstructor
        reconstructor = WiLoRReconstructor(cfg, device)

    tracker = HandTracker(cfg, (W, H))
    smoother = HandSmoother(cfg)
    jsonl_path = os.path.join(out_dir, "hands.jsonl")
    jf = open(jsonl_path, "w")
    writer = VideoWriter(os.path.join(out_dir, "annotated_video.mp4"), info.fps, (W, H),
                         cfg.video_crf) if cfg.write_video else None
    records_light: List[Dict] = []
    vert_store = {"frame": [], "id": [], "is_right": [], "vertices": [], "cam_t": []}
    mesh_dir = os.path.join(out_dir, "meshes")
    timing = {"detector_s": 0.0, "wilor_s": 0.0, "processed_frames": 0}
    n_written = [0]

    def sink(rec: Dict):
        img = rec.pop("_img", None)
        mesh = rec.pop("_mesh", None)
        rec.pop("_processed", None)
        jf.write(json.dumps(rec, separators=(",", ":")) + "\n")
        n_written[0] += 1
        if n_written[0] % cfg.flush_every == 0:
            jf.flush()
        records_light.append(rec)
        if writer is not None and img is not None:
            if cfg.render_mesh and mesh:
                try:
                    img = reconstructor.render_mesh_overlay(img, *mesh)
                except Exception as e:  # noqa: BLE001
                    log(f"[viz] mesh overlay disabled: {e}")
                    cfg.render_mesh = False
            writer.write(draw_record(img, rec))

    step = cfg.frame_step
    buffer = GapFillBuffer(delay=max(cfg.max_interp_gap, step) + 2,
                           max_gap=max(cfg.max_interp_gap, step - 1),
                           enabled=cfg.interpolate_gaps or step > 1, sink=sink)

    def process_chunk(chunk):
        proc = [(k, c) for k, c in enumerate(chunk) if (c[0] - cfg.start_frame) % step == 0]
        frames = []
        for _, (_, _, img) in proc:
            frames.append(img if cfg.proc_scale == 1.0 else
                          cv2.resize(img, None, fx=cfg.proc_scale, fy=cfg.proc_scale,
                                     interpolation=cv2.INTER_AREA))
        dets = detector(frames) if frames else []
        timing["detector_s"] += getattr(detector, "last_time_s", 0.0)
        timing["processed_frames"] += len(frames)
        from .wilor_inference import HandRequest
        per_frame_assign, requests = [], []
        for slot, ((_, (fidx, _, _)), d) in enumerate(zip(proc, dets)):
            assigns = tracker.step(d, fidx)
            per_frame_assign.append(assigns)
            for a in assigns:
                requests.append(HandRequest(slot, d[a.det_index, :4].copy(), a.is_right))
        need_verts = cfg.save_vertices or cfg.render_mesh or cfg.export_obj_every > 0
        res = reconstructor.reconstruct(frames, requests, return_vertices=need_verts)
        timing["wilor_s"] += getattr(reconstructor, "last_time_s", 0.0)

        ri, slot_of = 0, {k: s for s, (k, _) in enumerate(proc)}
        inv = 1.0 / cfg.proc_scale
        for k, (fidx, t, img) in enumerate(chunk):
            rec = {"frame": int(fidx), "t": round(float(t), 4), "hands": [], "_img": img,
                   "_processed": k in slot_of}
            mesh = ([], [], [], None)
            if k in slot_of:
                s = slot_of[k]
                d = dets[s]
                for a in per_frame_assign[s]:
                    r = res[ri]; ri += 1
                    joints, cam_t = r["joints"], r["cam_t"]
                    kp3d = joints - joints[0] if cfg.kp3d_frame == "wrist_relative" else joints + cam_t
                    kp2d = r["kp2d"] * inv
                    kp2d, kp3d = smoother(a.track_id, float(t), kp2d, kp3d)
                    side = "right" if a.is_right else "left"
                    hand = {"id": int(a.track_id), "side": side,
                            "bbox": _r(d[a.det_index, :4] * inv, 2),
                            "score": round(float(d[a.det_index, 4]), 4),
                            "kp2d": _r(kp2d, 2), "kp3d": _r(kp3d, 5), "interpolated": False}
                    if cfg.kp3d_frame == "wrist_relative":
                        hand["kp3d_root_cam"] = _r(joints[0] + cam_t, 5)
                    rec["hands"].append(hand)
                    v = r.get("vertices")
                    if v is not None:
                        if cfg.save_vertices:
                            vert_store["frame"].append(fidx); vert_store["id"].append(a.track_id)
                            vert_store["is_right"].append(a.is_right)
                            vert_store["vertices"].append(v.astype(np.float16))
                            vert_store["cam_t"].append(cam_t.astype(np.float32))
                        if cfg.export_obj_every and fidx % cfg.export_obj_every == 0:
                            os.makedirs(mesh_dir, exist_ok=True)
                            reconstructor.export_obj(os.path.join(mesh_dir, f"f{fidx:06d}_id{a.track_id}_{side}.obj"),
                                                     v, cam_t, a.is_right)
                        if cfg.render_mesh and cfg.proc_scale == 1.0:
                            mesh[0].append(v); mesh[1].append(cam_t); mesh[2].append(a.is_right)
                            mesh = (mesh[0], mesh[1], mesh[2], r["focal"])
                rec["hands"].sort(key=lambda h: h["id"])
            if mesh[0]:
                rec["_mesh"] = mesh
            buffer.push(rec)
        smoother.drop_missing(tracker.live_ids())   # free filters of deleted tracks

    t_start = time.perf_counter()
    reader = VideoReader(cfg.video_path, start=cfg.start_frame, max_frames=cfg.max_frames)
    total = min(info.frame_count - cfg.start_frame, cfg.max_frames or 10 ** 9)
    chunk, last_report = [], 0
    log(f"[run] {stem}: {W}x{H} @ {info.fps:.2f} fps, {total} frames -> {out_dir}")
    for item in reader:
        chunk.append(item)
        if len(chunk) >= cfg.chunk_size:
            process_chunk(chunk); chunk = []
        done = item[0] - cfg.start_frame + 1
        if done - last_report >= cfg.progress_every:
            last_report = done
            el = time.perf_counter() - t_start
            log(f"[run] {done}/{total} frames | {done / el:.2f} fps | "
                f"ids so far {tracker.next_id - 1} | ETA {(total - done) / max(done / el, 1e-6) / 60:.1f} min")
    if chunk:
        process_chunk(chunk)
    buffer.flush()
    jf.close()
    timing["wall_s"] = time.perf_counter() - t_start
    video_path = writer.close() if writer else None

    if cfg.save_vertices and vert_store["frame"]:
        np.savez_compressed(os.path.join(out_dir, "vertices.npz"),
                            frame=np.array(vert_store["frame"]), id=np.array(vert_store["id"]),
                            is_right=np.array(vert_store["is_right"]),
                            vertices=np.stack(vert_store["vertices"]), cam_t=np.stack(vert_store["cam_t"]),
                            faces=np.asarray(getattr(reconstructor, "faces", np.zeros((0, 3)))))

    frames_out = [r["frame"] for r in records_light]
    # one record per decoded frame, consecutive, no duplicates
    assert frames_out == list(range(cfg.start_frame, cfg.start_frame + len(frames_out))), "frame gap in JSONL"
    if len(frames_out) != total:
        log(f"[run] note: container reported {total} frames, decoded {len(frames_out)}")

    metrics = compute_metrics(records_light, timing, tracker.events, (W, H), info.fps)
    metrics["interpolation_fills"] = buffer.filled
    metrics["video"] = dataclasses.asdict(info)
    metrics["config"] = {k: getattr(cfg, k) for k in ("fast", "fp16", "det_imgsz", "det_conf", "frame_step",
                                                       "smoothing", "side_mode", "handedness_swap")}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    meta = {"joint_names": JOINT_NAMES, "coordinates": COORDINATE_DOC, "kp3d_frame": cfg.kp3d_frame,
            "video": dataclasses.asdict(info),
            "detector": detector.describe() if hasattr(detector, "describe") else None,
            "checkpoint_load_report": getattr(reconstructor, "load_report", None)}
    with open(os.path.join(out_dir, "hands_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(os.path.join(out_dir, "tracker_events.json"), "w") as f:
        json.dump(tracker.events, f, default=str)
    log(f"[run] done in {timing['wall_s'] / 60:.1f} min | detection rate "
        f"{metrics['frame_detection_rate']:.3f} | ids {metrics['tracking_ids_generated']}")
    return {"out_dir": out_dir, "jsonl": jsonl_path, "video": video_path, "metrics": metrics}


# --------------------------------------------------------------------------------------------
BENCH_MODES = {"fp32": {"fast": False, "fp16": False},      # reference
               "fp16": {"fast": False, "fp16": True},       # half precision, every ViT block kept
               "fast": {"fast": True, "fp16": False}}       # official --fast: FP16 + skipped ViT blocks


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_mode(report: Dict, max_mean_px: float = 1.0, max_p95_px: float = 3.0) -> str:
    """Fastest mode whose 2D keypoints stay within the thresholds of the FP32 reference.

    Defaults are strict on purpose: hand-tracking accuracy outweighs speed in the judging. Falls back to
    "fp32" when no faster mode qualifies (or when no hand was detected in the benchmark frames)."""
    best, best_s = "fp32", report.get("fp32", {}).get("wilor_s", float("inf"))
    for m in ("fp16", "fast"):
        r = report.get(m)
        if not r or "kp2d_mean_px_diff" not in r:
            continue
        if (r["kp2d_mean_px_diff"] <= max_mean_px and r["kp2d_p95_px_diff"] <= max_p95_px
                and r["wilor_s"] < best_s):
            best, best_s = m, r["wilor_s"]
    return best


def benchmark_modes(cfg: PipelineConfig, n_frames: int = 150, modes=("fp32", "fp16", "fast"),
                    max_mean_px: float = 1.0, max_p95_px: float = 3.0, log=print) -> Dict:
    """Same detections, same crops, different precision. For every mode: WiLoR time and, against the
    FP32 reference, the keypoint deviation it introduces (2D in pixels, 3D wrist-relative in mm).
    `report["recommended"]` is the fastest mode within the thresholds (see pick_mode)."""
    from .wilor_detector import build_detector
    from .wilor_inference import HandRequest, WiLoRReconstructor
    cfg = cfg.resolved()
    device = _device()
    modes = ["fp32"] + [m for m in modes if m != "fp32"]
    det = build_detector(cfg, device)
    frames = [img for _, _, img in VideoReader(cfg.video_path, cfg.start_frame, n_frames)]
    dets = []
    for s in range(0, len(frames), cfg.chunk_size):
        dets += det(frames[s:s + cfg.chunk_size])
    del det
    reqs = [HandRequest(i, d[j, :4], bool(d[j, 5] > 0.5)) for i, d in enumerate(dets) for j in range(len(d))]
    out: Dict = {"frames": len(frames), "hands": len(reqs)}
    preds: Dict = {}
    for name in modes:
        c = dataclasses.replace(cfg, **BENCH_MODES[name])
        rec = WiLoRReconstructor(c, device, verify_checkpoint=False)
        rec.reconstruct(frames[:4], [r for r in reqs if r.frame_slot < 4])      # warm-up
        t0 = time.perf_counter()
        res = []
        for s in range(0, len(frames), cfg.chunk_size):
            sub = [HandRequest(r.frame_slot - s, r.bbox, r.is_right)
                   for r in reqs if s <= r.frame_slot < s + cfg.chunk_size]
            res += rec.reconstruct(frames[s:s + cfg.chunk_size], sub)
        dt = time.perf_counter() - t0
        preds[name] = res
        out[name] = {"wilor_s": round(dt, 3), "ms_per_hand": round(1000 * dt / max(1, len(reqs)), 2),
                     "frames_per_s": round(len(frames) / dt, 2)}
        del rec
        _free_gpu()
    if reqs:
        ref = preds["fp32"]
        for name in modes[1:]:
            d2 = [np.linalg.norm(a["kp2d"] - b["kp2d"], axis=1).mean() for a, b in zip(ref, preds[name])]
            d3 = [1000 * np.linalg.norm((a["joints"] - a["joints"][0]) - (b["joints"] - b["joints"][0]),
                                        axis=1).mean() for a, b in zip(ref, preds[name])]
            out[name].update(kp2d_mean_px_diff=round(float(np.mean(d2)), 3),
                             kp2d_p95_px_diff=round(float(np.percentile(d2, 95)), 3),
                             kp3d_rel_mean_mm_diff=round(float(np.mean(d3)), 3),
                             speedup=round(out["fp32"]["wilor_s"] / max(out[name]["wilor_s"], 1e-9), 2))
    out["recommended"] = pick_mode(out, max_mean_px, max_p95_px)
    out["thresholds"] = {"max_mean_px": max_mean_px, "max_p95_px": max_p95_px}
    log(json.dumps(out, indent=2))
    return out


def _cli():
    p = argparse.ArgumentParser(description="WiLoR PS1 video -> hands.jsonl")
    for f in dataclasses.fields(PipelineConfig):
        if f.name == "extra":
            continue
        default = f.default
        if isinstance(default, bool):
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        elif default is None:
            p.add_argument(f"--{f.name}", type=int, default=None)
        else:
            p.add_argument(f"--{f.name}", type=type(default), default=default)
    a = p.parse_args()
    cfg = PipelineConfig(**vars(a))
    if not cfg.video_path:
        p.error("--video_path is required")
    run_video(cfg)


if __name__ == "__main__":
    _cli()
