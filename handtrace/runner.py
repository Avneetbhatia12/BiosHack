"""Run PS1 + PS2 over many clips: resumable, multi-GPU, models loaded once per worker.

A *worker* is one process bound to one GPU. Workers share the clip list and claim clips by creating a lock
file atomically, so any number of workers balances itself, a finished clip is never repeated, and an
interrupted run continues where it stopped. Two stages, run in this order:

  ps1  WiLoR tracking -> post-processing -> re-reconstruction of side-corrected hands -> hands_final.jsonl
  ps2  object detection + contacts from the saved hands_final.jsonl (WiLoR is never loaded)

Videos are NOT rendered here (an annotated copy of every clip costs minutes of CPU per clip); render only
the demo clips with handtrace.render. A clip that raises is logged and skipped so that one bad file cannot
end a multi-hour run; the process then exits with status 1.

    python -m handtrace.runner --stage ps1 --settings settings.json --worker-id 0
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")
MODES = ("fp32", "fp16", "fast")
STAGE_FORCE = {"ps1": {"ps1", "post"}, "ps2": {"ps2"}}


# ----------------------------------------------------------------------------- settings
@dataclass
class Settings:
    clips: List[str] = field(default_factory=list)
    out_root: str = "outputs"
    wilor_root: str = "WiLoR"
    prompts_path: str = ""                    # "" = configs/ps2_objects.json next to the package
    priority: List[str] = field(default_factory=list)   # name substrings processed first (the demo clips)
    # ---- PS1
    mode: str = "fp32"                        # fp32 | fp16 | fast   (see wilor.pipeline.benchmark_modes)
    det_imgsz: int = 640
    handedness_swap: bool = False
    chunk_size: int = 16                      # frames per batch; 16 frames ~ 32 hands = one WiLoR batch
    wilor_batch_size: int = 32
    frame_step: int = 1
    compile_backbone: bool = False
    post: dict = field(default_factory=dict)              # PostConfig overrides
    rereconstruct: bool = True
    ps1_extra: dict = field(default_factory=dict)         # any other PipelineConfig field
    # ---- PS2
    ps2: dict = field(default_factory=dict)               # PS2Config overrides
    # ---- run control
    max_frames: Optional[int] = None          # smoke tests
    deadline: Optional[float] = None          # epoch seconds: never start a clip that cannot finish before it
    est_clip_s: dict = field(default_factory=dict)        # {"ps1": s, "ps2": s}: per-clip prior until a worker has its own
    force: List[str] = field(default_factory=list)        # "ps1" (redo raw), "post" (redo post-proc), "ps2" (redo detections)
    force_since: Optional[float] = None       # epoch s: `force` redoes only outputs OLDER than this, so an interrupted
                                              # forced run resumes; None = redo everything on every launch
    model_factory: str = ""                   # "module:function" replacing the real models (tests / plug-ins)

    def save(self, path: str) -> None:
        _atomic_write(path, json.dumps(dataclasses.asdict(self), indent=1))

    @classmethod
    def load(cls, path: str) -> "Settings":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- clips and markers
def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def discover_clips(roots: Sequence[str]) -> List[str]:
    """Every video under `roots`. When one file name occurs several times (a dataset often holds the same
    clip in more than one folder, e.g. full resolution and a 480p copy) the largest file is kept."""
    best: Dict[str, str] = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if os.path.isfile(p) and p.lower().endswith(VIDEO_EXT):
                name = os.path.basename(p)
                if name not in best or os.path.getsize(p) > os.path.getsize(best[name]):
                    best[name] = p
    return [best[k] for k in sorted(best)]


def order_clips(clips: Sequence[str], priority: Sequence[str] = ()) -> List[str]:
    """Clips whose name contains a `priority` substring first (in that order), then the rest by name."""
    rest = sorted(clips, key=lambda c: os.path.basename(c))
    first: List[str] = []
    for key in priority:
        for c in rest:
            if key.lower() in os.path.basename(c).lower() and c not in first:
                first.append(c)
    return first + [c for c in rest if c not in first]


def clip_dir(out_root: str, stem: str) -> str:
    return os.path.join(out_root, stem)


def raw_done(out_root: str, stem: str) -> bool:
    d = clip_dir(out_root, stem)       # run_video writes metrics.json, then hands_meta.json: both mean it finished
    return all(os.path.exists(os.path.join(d, f)) for f in ("hands.jsonl", "metrics.json", "hands_meta.json"))


def ps1_done(out_root: str, stem: str) -> bool:
    d = clip_dir(out_root, stem)       # the report is written last, so its presence means the clip is complete
    return os.path.exists(os.path.join(d, "hands_final.jsonl")) and os.path.exists(
        os.path.join(d, "postprocess_report.json"))


def ps2_done(out_root: str, stem: str) -> bool:
    return os.path.exists(os.path.join(clip_dir(out_root, stem), "ps2_final", "ps2_metrics.json"))


def _stale(path: str, s: Settings, flag: str) -> bool:
    """True when `flag` was requested in s.force and `path` is older than the request (or there is no request time)."""
    if flag not in s.force or not os.path.exists(path):
        return False
    return s.force_since is None or os.path.getmtime(path) < s.force_since


def needs_work(stage: str, s: Settings, stem: str) -> bool:
    d = clip_dir(s.out_root, stem)
    if stage == "ps1":
        marker = os.path.join(d, "postprocess_report.json")
        return (not ps1_done(s.out_root, stem)) or any(_stale(marker, s, f) for f in ("ps1", "post"))
    marker = os.path.join(d, "ps2_final", "ps2_metrics.json")
    return (not ps2_done(s.out_root, stem)) or _stale(marker, s, "ps2")


def status(s: Settings) -> Dict[str, dict]:
    out = {}
    for stage in ("ps1", "ps2"):
        pending = [stem_of(c) for c in s.clips if needs_work(stage, s, stem_of(c))]
        out[stage] = {"done": len(s.clips) - len(pending), "total": len(s.clips), "pending": pending}
    return out


# ----------------------------------------------------------------------------- locks
def _lock_dir(out_root: str, stage: str) -> str:
    return os.path.join(out_root, "_locks", stage)


def clear_locks(out_root: str, stage: str) -> None:
    d = _lock_dir(out_root, stage)
    if os.path.isdir(d):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))


def claim(out_root: str, stage: str, stem: str) -> bool:
    """True for exactly one caller per (stage, clip): O_EXCL file creation is atomic."""
    d = _lock_dir(out_root, stage)
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, stem + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


# ----------------------------------------------------------------------------- PS1
def ps1_config(s: Settings, clip: str):
    from .wilor.config import PipelineConfig
    if s.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {s.mode!r}")
    kw = dict(video_path=clip, output_dir=s.out_root, wilor_root=s.wilor_root, fast=s.mode == "fast",
              fp16=s.mode == "fp16", det_imgsz=s.det_imgsz, handedness_swap=s.handedness_swap,
              chunk_size=s.chunk_size, wilor_batch_size=s.wilor_batch_size, frame_step=s.frame_step,
              compile_backbone=s.compile_backbone, max_frames=s.max_frames, write_video=False,
              save_vertices=False, export_obj_every=0)
    kw.update(s.ps1_extra)
    return PipelineConfig(**kw)


def postprocess_clip(out_dir: str, **post_kw):
    """hands.jsonl -> (final records, report). Pure Python, seconds per clip."""
    from .postprocess import PostConfig, postprocess
    from .wilor.evaluation import load_jsonl
    with open(os.path.join(out_dir, "hands_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)["video"]
    cfg = PostConfig(frame_wh=(meta["width"], meta["height"]), **post_kw)
    return postprocess(load_jsonl(os.path.join(out_dir, "hands.jsonl")), cfg)


def save_final(out_dir: str, final: List[dict], report: dict) -> None:
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in final)
    _atomic_write(os.path.join(out_dir, "hands_final.jsonl"), body)
    _atomic_write(os.path.join(out_dir, "postprocess_report.json"), json.dumps(report, indent=1))   # marker


def process_ps1_clip(clip: str, s: Settings, get_models: Callable[[], tuple], log=print) -> dict:
    """Raw WiLoR run (skipped when its metrics.json exists) -> post-processing -> re-reconstruct hands whose
    side changed -> hands_final.jsonl. The models are only built when something needs the GPU."""
    from .wilor.pipeline import run_video
    from .wilor.refine import rereconstruct_corrected
    stem = stem_of(clip)
    out_dir = clip_dir(s.out_root, stem)
    cfg = ps1_config(s, clip)
    t0 = time.perf_counter()
    if raw_done(s.out_root, stem) and not _stale(os.path.join(out_dir, "metrics.json"), s, "ps1"):
        with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        log(f"[ps1] {stem}: reusing raw hands.jsonl")
    else:
        detector, recon = get_models()
        metrics = run_video(cfg, detector=detector, reconstructor=recon, log=log)["metrics"]
    final, rep = postprocess_clip(out_dir, **s.post)
    need = s.rereconstruct and any(h.get("side_corrected") and not h.get("interpolated")
                                   for r in final for h in r["hands"])
    rep["rereconstructed"] = 0
    if need:
        rep["rereconstructed"] = rereconstruct_corrected(clip, final, get_models()[1],
                                                         kp3d_frame=cfg.kp3d_frame, chunk_size=cfg.chunk_size, log=log)
    rep["mode"] = s.mode
    save_final(out_dir, final, rep)
    wall = time.perf_counter() - t0
    log(f"[ps1] {stem}: done in {wall / 60:.1f} min | frames with a hand {rep['frames_with_ge1_hand']:.3f} | "
        f"ids {metrics['tracking_ids_generated']} -> {rep['ids_out']}")
    return {"clip": stem, "wall_s": round(wall, 1), "end_to_end_fps": metrics.get("end_to_end_fps")}


# ----------------------------------------------------------------------------- PS2
def ps2_config(s: Settings, clip: str, hands_jsonl: str = "", output_dir: str = ""):
    from .ps2.config import PS2Config
    kw = dict(video_path=clip, hands_jsonl=hands_jsonl, output_dir=output_dir, write_video=False,
              max_frames=s.max_frames)
    if s.prompts_path:
        kw["prompts_path"] = s.prompts_path
    kw.update(s.ps2)
    return PS2Config(**kw)


def process_ps2_clip(clip: str, s: Settings, detector, log=print) -> dict:
    from .ps2.pipeline import run_ps2
    stem = stem_of(clip)
    d = clip_dir(s.out_root, stem)
    hands = os.path.join(d, "hands_final.jsonl")
    if not os.path.exists(hands):
        raise FileNotFoundError(f"{hands} is missing: run the ps1 stage for {stem} first")
    cfg = ps2_config(s, clip, hands, os.path.join(d, "ps2_final"))
    if _stale(os.path.join(d, "ps2_final", "object_detections_meta.json"), s, "ps2"):
        cfg = dataclasses.replace(cfg, overwrite_detections=True)
    t0 = time.perf_counter()
    res = run_ps2(cfg, detector=detector, log=log)
    m = res["metrics"]
    log(f"[ps2] {stem}: done in {(time.perf_counter() - t0) / 60:.1f} min | {m['contact_intervals']} contact intervals "
        f"({m['grab_release_intervals']} grab->release)")
    return {"clip": stem, "wall_s": round(time.perf_counter() - t0, 1), "contact_intervals": m["contact_intervals"]}


def retune_contacts(s: Settings, log=print, **overrides) -> Dict[str, dict]:
    """New contact thresholds on the cached object detections of every finished PS2 clip: CPU only, seconds."""
    from .ps2.pipeline import rerun_contacts
    out = {}
    for clip in s.clips:
        stem = stem_of(clip)
        if ps2_done(s.out_root, stem):
            d = clip_dir(s.out_root, stem)
            cfg = ps2_config(s, clip, os.path.join(d, "hands_final.jsonl"), os.path.join(d, "ps2_final"))
            out[stem] = rerun_contacts(cfg, log=log, **overrides)["metrics"]
    return out


# ----------------------------------------------------------------------------- models
def load_factory(spec: str) -> Callable:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def real_models(stage: str, s: Settings, log=print) -> tuple:
    """(detector, reconstructor) for ps1; (object detector,) for ps2. Imported lazily: torch only here."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if stage == "ps1":
        from .wilor.wilor_detector import build_detector
        from .wilor.wilor_inference import WiLoRReconstructor
        base = ps1_config(s, s.clips[0]).resolved()
        detector = build_detector(base, device)
        log(f"[ps1] detector classes: {detector.describe()}")
        recon = WiLoRReconstructor(base, device)
        log(f"[ps1] WiLoR loaded ({s.mode}) on {device}; checkpoint check: {recon.load_report}")
        return detector, recon
    from .ps2.config import ObjectVocabulary
    from .ps2.object_detector import build_object_detector
    cfg = ps2_config(s, s.clips[0])
    vocab = ObjectVocabulary.load(cfg.prompts_path, cfg.profile, stem_of(s.clips[0]))
    try:
        det = build_object_detector(cfg, vocab, device)
    except Exception as e:  # noqa: BLE001
        fb = "owlv2" if cfg.backend != "owlv2" else "yolo"
        log(f"[ps2] backend {cfg.backend!r} failed to load ({e!r}); falling back to {fb!r}")
        det = build_object_detector(dataclasses.replace(cfg, backend=fb, object_imgsz=960 if fb == "owlv2" else 640),
                                    vocab, device)
    log(f"[ps2] object detector: {json.dumps(det.describe())[:600]}")
    return (det,)


def _memo(fn: Callable[[], tuple]) -> Callable[[], tuple]:
    box: list = []

    def get():
        if not box:
            box.append(fn())
        return box[0]
    return get


# ----------------------------------------------------------------------------- worker
def run_stage(stage: str, s: Settings, worker_id: int = 0, log=None) -> dict:
    """Process clips for `stage` until none is left (or the time budget would be exceeded)."""
    log = log or (lambda *a: print(*a, flush=True))
    tag = f"[w{worker_id}]"
    say = lambda *a: log(tag, *a)  # noqa: E731
    factory = load_factory(s.model_factory) if s.model_factory else (lambda st, se: real_models(st, se, say))
    get_models = _memo(lambda: factory(stage, s))
    done, failed, skipped, wall = [], [], [], []
    for clip in order_clips(s.clips, s.priority):
        stem = stem_of(clip)
        if not needs_work(stage, s, stem):
            continue
        if stage == "ps2" and not ps1_done(s.out_root, stem):
            skipped.append(stem)                                 # waiting for PS1: not an error
            continue
        est = statistics.median(wall) if wall else s.est_clip_s.get(stage, 0.0)
        if s.deadline and time.time() + est > s.deadline:
            say(f"time budget: stopping before {stem} (about {est / 60:.0f} min needed)")
            break
        if not claim(s.out_root, stage, stem):
            continue
        try:
            if stage == "ps1":
                r = process_ps1_clip(clip, s, get_models, say)
            else:
                r = process_ps2_clip(clip, s, get_models()[0], say)
            wall.append(r["wall_s"])
            done.append(stem)
        except Exception:  # noqa: BLE001 - one bad clip must not end the run
            tb = traceback.format_exc()
            say(f"FAILED {stem}:\n{tb}")
            _atomic_write(os.path.join(s.out_root, "_failures", f"{stage}_{stem}.txt"), tb)
            failed.append(stem)
    say(f"finished: {len(done)} done, {len(failed)} failed" +
        (f", {len(skipped)} skipped (PS1 not finished): {skipped}" if skipped else ""))
    return {"stage": stage, "worker": worker_id, "done": done, "failed": failed, "skipped": skipped}


# ----------------------------------------------------------------------------- launcher
def gpu_count() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return sum(1 for l in r.stdout.splitlines() if l.startswith("GPU "))
    except Exception:  # noqa: BLE001
        return 0


def gpu_names() -> List[str]:
    """["Tesla T4, 15360 MiB", ...]; empty when there is no usable nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []
    except Exception:  # noqa: BLE001
        return []


def _tail(path: str, n: int = 1) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines()[-n:]]
        return " | ".join(lines) if n == 1 else "\n".join(lines)
    except OSError:
        return ""


def launch(stage: str, s: Settings, settings_path: str, code_root: str, workers: Optional[int] = None,
           poll_s: float = 30.0, show: Optional[Callable[[str], None]] = None) -> dict:
    """Start one worker process per GPU, wait for all, print progress. Ctrl-C / interrupt stops the workers;
    finished clips are kept and the next launch continues. Returns per-worker exit codes and log paths."""
    show = show or print
    s.save(settings_path)
    clear_locks(s.out_root, stage)
    for f in glob.glob(os.path.join(s.out_root, "_failures", f"{stage}_*.txt")):
        os.remove(f)                                   # only failures of THIS launch are reported
    n_gpu = gpu_count()
    n = max(1, workers if workers is not None else n_gpu)
    log_dir = os.path.join(s.out_root, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    procs, logs = [], []
    for k in range(n):
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": code_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpu)
        path = os.path.join(log_dir, f"{stage}_w{k}.log")
        logs.append(path)
        with open(path, "w") as lf:
            procs.append(subprocess.Popen([sys.executable, "-m", "handtrace.runner", "--stage", stage,
                                           "--settings", settings_path, "--worker-id", str(k)],
                                          cwd=code_root, env=env, stdout=lf, stderr=subprocess.STDOUT))
    show(f"[{stage}] {n} worker(s) on {n_gpu or 'no'} GPU(s); logs in {log_dir}")
    try:
        last = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(min(1.0, poll_s))                      # notice the end quickly, but only print every poll_s
            if time.time() - last >= poll_s:
                last = time.time()
                st = status(s)[stage]
                show(f"[{stage}] {st['done']}/{st['total']} clips done | " +
                     " || ".join(f"w{k}: {_tail(l)[-110:]}" for k, l in enumerate(logs)))
    except BaseException:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    codes = [p.returncode for p in procs]
    st = status(s)[stage]
    failures = {}
    for f in sorted(glob.glob(os.path.join(s.out_root, "_failures", f"{stage}_*.txt"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            failures[os.path.basename(f)[len(stage) + 1:-4]] = "".join(fh.readlines()[-25:])
    if any(codes) and not failures:                    # a worker died before it could record a clip failure
        failures["worker"] = "\n".join(_tail(l, 25) for l in logs)
    for name, tb in failures.items():
        show(f"[{stage}] ERROR in {name}:\n{tb}")
    show(f"[{stage}] finished: {st['done']}/{st['total']} clips done, exit codes {codes}")
    if st["pending"]:
        show(f"[{stage}] still pending: {st['pending']} (failures: {os.path.join(s.out_root, '_failures')})")
    return {"returncodes": codes, "logs": logs, "status": st, "failures": failures}


# ----------------------------------------------------------------------------- reporting
def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def clip_row(out_root: str, stem: str) -> dict:
    d = clip_dir(out_root, stem)
    raw, post = _read_json(os.path.join(d, "metrics.json")), _read_json(os.path.join(d, "postprocess_report.json"))
    ps2 = _read_json(os.path.join(d, "ps2_final", "ps2_metrics.json"))
    cfg = raw.get("config", {})
    return {
        "clip": stem, "mode": post.get("mode") or ("fast" if cfg.get("fast") else "fp16" if cfg.get("fp16") else "fp32"),
        "frames": raw.get("total_frames"), "raw_frame_detection_rate": raw.get("frame_detection_rate"),
        "raw_two_hand_rate": raw.get("two_hand_frame_rate"), "raw_ids": raw.get("tracking_ids_generated"),
        "raw_frames_gt2_hands": raw.get("frames_with_gt2_hands_observed"),
        "ps1_end_to_end_fps": raw.get("end_to_end_fps"), "ps1_minutes": round(raw.get("total_processing_time_s", 0) / 60, 1) or None,
        "handedness_spatial_consistency": raw.get("handedness_spatial_consistency"),
        "final_frames_with_hand": post.get("frames_with_ge1_hand"), "final_frames_with_2_hands": post.get("frames_with_2_hands"),
        "final_ids": post.get("ids_out"), "background_tracks_removed": (post.get("wearer") or {}).get("background_tracks"),
        "relinks": post.get("relinks"), "side_changes": post.get("side_changes"), "rereconstructed": post.get("rereconstructed"),
        "ps2_contact_intervals": ps2.get("contact_intervals"), "ps2_portable": ps2.get("portable_contacts"),
        "ps2_fixed": ps2.get("fixed_contacts"), "ps2_grab_release": ps2.get("grab_release_intervals"),
        "ps2_frames_in_contact": ps2.get("frames_with_hand_in_contact"), "ps2_object_ids": ps2.get("unique_object_ids"),
        "ps2_detector_fps": (ps2.get("timing") or {}).get("object_detector_fps"),
    }


def write_summary(out_root: str, clips: Sequence[str]) -> List[dict]:
    rows = [clip_row(out_root, stem_of(c)) for c in clips]
    if rows:
        path = os.path.join(out_root, "summary_all_clips.csv")
        with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(path + ".tmp", path)
    return rows


def collect_contacts(out_root: str, clips: Sequence[str]) -> int:
    """All clips' contacts.csv in one file (clip, t_start, t_end, hand, state, object_name, ...). Returns rows."""
    rows, fields = [], None
    for c in clips:
        p = os.path.join(clip_dir(out_root, stem_of(c)), "ps2_final", "contacts.csv")
        if os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                fields = fields or rd.fieldnames
                rows += list(rd)
    if fields:
        path = os.path.join(out_root, "contacts_all.csv")
        with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(path + ".tmp", path)
    return len(rows)


def package_results(out_root: str, clips: Sequence[str], dest: str, include_detections: bool = False) -> str:
    """Zip the result files (no videos, no frames): per clip PS1 / PS2 / PS3 / PS5 results, the manual audit and the
    review player; plus the cross-clip summaries and the dashboard."""
    names = ["hands_final.jsonl", "postprocess_report.json", "metrics.json", "hands_meta.json",
             os.path.join("ps2_final", "contacts.csv"), os.path.join("ps2_final", "contacts.jsonl"),
             os.path.join("ps2_final", "ps2_metrics.json"),
             os.path.join("ps3", "actions.json"), os.path.join("ps3", "actions.csv"),
             os.path.join("ps3", "actions_final.json"), os.path.join("ps3", "actions_final.csv"),
             os.path.join("ps5", "quality.json"), os.path.join("ps5", "safety_events.csv"),
             os.path.join("review", "manual_audit.csv"), os.path.join("review", "{stem}_review.html")]
    if include_detections:
        names.append(os.path.join("ps2_final", "object_detections.jsonl"))
    with zipfile.ZipFile(dest + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for top in ("summary_all_clips.csv", "contacts_all.csv", "environment.json", "actions_all.csv",
                    "ps5_events_all.csv", "ps5_quality_all.csv", "ps5_safety_all.csv", "ps5_dashboard.html",
                    "ps5_dashboard_clips.csv", "ps5_time_per_step.csv", "ps5_tool_usage.csv", "accuracy_report.json"):
            p = os.path.join(out_root, top)
            if os.path.exists(p):
                z.write(p, top)
        for c in clips:
            stem = stem_of(c)
            for n in names:
                n = n.format(stem=stem)
                p = os.path.join(clip_dir(out_root, stem), n)
                if os.path.exists(p):
                    z.write(p, os.path.join(stem, n))
    os.replace(dest + ".tmp", dest)
    return dest


def project_hours(fps: float, n_clips: int, frames_per_clip: int = 5400, workers: int = 1) -> float:
    """Wall-clock hours to process n_clips at `fps` per worker with `workers` workers in parallel."""
    return n_clips * frames_per_clip / max(fps, 1e-9) / max(workers, 1) / 3600.0


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PS1/PS2 worker: process clips until none is left")
    ap.add_argument("--stage", choices=["ps1", "ps2"], required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--worker-id", type=int, default=0)
    a = ap.parse_args(argv)
    res = run_stage(a.stage, Settings.load(a.settings), a.worker_id)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
