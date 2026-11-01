"""PS3 pipeline: per clip -> windows -> Gemini -> validate -> merge -> actions.json / actions.csv.

Resumable at two levels: a finished clip (ps3/actions.json exists) is skipped, and inside an unfinished clip every
finished window is reused from ps3/raw/. Clips are processed one at a time.

Outputs per clip (OUT_ROOT/<stem>/ps3/):
    actions.json / actions.csv          the model's timeline after validation + merge (actions.json = "done" marker)
    actions_final.json / .csv           the same, with edges snapped to PS2 grab/release moments (refine.py);
                                        identical to actions.* when PS2 has not run. This is the file to use.
    raw/window_*.json                   every model reply (never pay twice), dropped_actions.json, state.json
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

from .config import PS3Config
from .merge import merge_actions
from .prompt import build_prompt, prompt_key
from .validate import parse_model_json, validate_actions
from .windows import make_windows

CSV_COLUMNS = ["clip", "start_s", "end_s", "verb", "object", "hand"]


# ------------------------------------------------------------------------------------------- small helpers
def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def video_duration_s(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if n <= 0:
        raise IOError(f"cannot read the frame count of {path}")
    return float(n / fps)


def clip_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def small_copy(clip_path: str, cfg: PS3Config, work_dir: str, log: Callable = print) -> tuple:
    """(path to upload, is_temporary). Same clip at lower resolution and the SAME timeline, so every timestamp the
    model returns is valid for the original. Prefers a ready-made copy (e.g. clips_480p/<same name>)."""
    if not cfg.upload_small_copy:
        return clip_path, False
    name = os.path.basename(clip_path)
    import glob
    for root in cfg.small_copy_roots:
        for p in glob.glob(os.path.join(root, "**", name), recursive=True):
            if os.path.abspath(p) != os.path.abspath(clip_path) and os.path.getsize(p) < os.path.getsize(clip_path):
                return p, False
    ff = shutil.which("ffmpeg")
    if not ff:
        return clip_path, False
    try:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:                                               # noqa: BLE001
        h = 0
    if h and h <= cfg.transcode_height:
        return clip_path, False
    out = os.path.join(work_dir, f"upload_{cfg.transcode_height}p.mp4")
    if not os.path.exists(out):
        base_cmd = [ff, "-y", "-loglevel", "error", "-i", clip_path, "-vf", f"scale=-2:{cfg.transcode_height}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-an"]
        for timing in (["-fps_mode", "passthrough"], ["-vsync", "0"]):   # keep every frame time (ffmpeg >= 5.1 / older)
            if subprocess.run(base_cmd + timing + [out + ".part.mp4"]).returncode == 0:
                break
        else:
            log("  ffmpeg could not make a small copy; uploading the original")
            return clip_path, False
        os.replace(out + ".part.mp4", out)
    return out, True


def ps3_dir(out_root: str, stem: str) -> str:
    return os.path.join(out_root, stem, "ps3")


def ps3_done(out_root: str, stem: str) -> bool:
    return os.path.exists(os.path.join(ps3_dir(out_root, stem), "actions.json"))


def ps2_object_hints(out_root: str, stem: str, top_n: int = 25) -> List[str]:
    """Object names PS2 saw in contact (most frequent first). Context for the model only, never ground truth."""
    path = os.path.join(out_root, stem, "ps2_final", "contacts.csv")
    if not os.path.exists(path):
        return []
    names: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("object_name") or "").strip().lower()
            if name and name not in {"unknown", "none"}:
                names[name] += 1
    return [n for n, _ in names.most_common(top_n)]


def write_actions_csv(path: str, clip: str, actions: List[dict]) -> None:
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for a in actions:
        w.writerow([clip, f"{a['start_s']:.2f}", f"{a['end_s']:.2f}", a["verb"], a["object"], a["hand"]])
    _atomic_write(path, buf.getvalue())


# ------------------------------------------------------------------------------------------- one window
def _analyse_window(client, get_remote, idx: int, win: tuple, duration: float, hints: List[str], cfg: PS3Config,
                    raw_dir: str, log: Callable, model_id: str = "") -> dict:
    """Return {"actions", "dropped", "info", "cached"} for one window, or {"error"} if it could not be parsed."""
    ws, we = win
    prompt = build_prompt(ws, we, duration, hints)
    model_id = model_id or getattr(client, "model", cfg.model)
    key = prompt_key(model_id, cfg.fps, prompt, ws, we)
    raw_path = os.path.join(raw_dir, f"window_{idx:03d}_{ws:.0f}-{we:.0f}.json")

    cached = _read_json(raw_path)
    if cached and cached.get("key") == key and cached.get("text"):
        text, was_cached = cached["text"], True
    else:
        text, was_cached = None, False

    last_err = "no reply"
    for attempt in range(1 + cfg.parse_retries):
        if text is None:
            text = client.generate(get_remote(), ws, we, prompt)
            _atomic_write(raw_path, json.dumps({"key": key, "model": model_id, "fps": cfg.fps, "window": [ws, we],
                                                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text}, indent=1))
        try:
            raw_actions = parse_model_json(text)
        except ValueError as exc:
            last_err = str(exc)
            log(f"  window {idx}: unusable reply ({last_err})" + ("; asking again" if attempt < cfg.parse_retries else ""))
            text = None
            if was_cached:                                           # a bad cached reply is not reused
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
                was_cached = False
            continue
        clean, dropped, info = validate_actions(raw_actions, ws, we, duration, cfg, window_index=idx)
        return {"actions": clean, "dropped": dropped, "info": info, "cached": was_cached, "window": [ws, we]}
    return {"error": last_err, "window": [ws, we]}


# ------------------------------------------------------------------------------------------- one clip
def process_clip(clip_path: str, out_root: str, cfg: PS3Config, client=None, log: Callable = print) -> dict:
    stem = clip_stem(clip_path)
    base = ps3_dir(out_root, stem)
    subset = cfg.only_windows is not None
    final_dir = os.path.join(base, "partial") if subset else base
    raw_dir = os.path.join(base, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if not subset and ps3_done(out_root, stem) and not cfg.force:
        log(f"[{stem}] already done: skipped (set force=True to redo)")
        finalize_clip(out_root, stem, cfg, log)
        return {"clip": stem, "status": "skipped", "actions": _read_json(os.path.join(base, "actions.json"))["actions"]}

    duration = video_duration_s(clip_path)
    windows = make_windows(duration, cfg.window_s, cfg.overlap_s)
    chosen = [(i, w) for i, w in enumerate(windows) if (not subset or i in cfg.only_windows)]
    if cfg.only_windows and not chosen:
        raise ValueError(f"only_windows={cfg.only_windows} but the clip has windows 0..{len(windows) - 1}")
    log(f"[{stem}] {duration:.1f}s -> {len(windows)} window(s), running {len(chosen)}"
        + (" (partial)" if subset else ""))

    if client is None:
        from .gemini_client import GeminiVideoClient
        client = GeminiVideoClient(cfg, log=log)
    model_id = getattr(client, "model", cfg.model)
    log(f"  model {model_id} | fps {cfg.fps} | thinking {cfg.thinking_level} | media resolution {cfg.media_resolution}")

    hints = ps2_object_hints(out_root, stem) if cfg.use_ps2_hints else []
    state_path = os.path.join(base, "state.json")
    state = _read_json(state_path) or {}
    holder = {"remote": client.get_active(state["remote_name"]) if state.get("remote_name") else None}

    tmp_copy = {"path": None}

    def get_remote():
        """Upload once, on first use (a clip whose windows are all cached never uploads)."""
        if holder["remote"] is None:
            up, is_tmp = small_copy(clip_path, cfg, base, log)
            tmp_copy["path"] = up if is_tmp else None
            log(f"  uploading {os.path.basename(up)} ({os.path.getsize(up) / 1e6:.1f} MB) ...")
            holder["remote"] = client.upload(up)
            _atomic_write(state_path, json.dumps({"remote_name": holder["remote"].name, "uploaded_at": time.time()}))
            log("  upload ACTIVE")
        return holder["remote"]

    results: Dict[int, dict] = {}
    for i, w in chosen:
        res = _analyse_window(client, get_remote, i, w, duration, hints, cfg, raw_dir, log, model_id)
        results[i] = res
        if "error" in res:
            log(f"  window {i} [{w[0]:.0f}-{w[1]:.0f}s]: FAILED ({res['error']})")
        else:
            inf = res["info"]
            log(f"  window {i} [{w[0]:.0f}-{w[1]:.0f}s]{' (cached)' if res['cached'] else ''}: {inf['kept']}/{inf['raw_count']} "
                f"actions kept, time base {inf['time_base']}")

    failed = [i for i, r in results.items() if "error" in r]
    all_actions = [a for r in results.values() if "error" not in r for a in r["actions"]]
    merged = merge_actions(all_actions, cfg)
    status = "complete" if not failed else "incomplete"

    doc = {"clip": stem, "duration_s": round(duration, 2), "model": model_id, "status": status,
           "windows": [{"index": i, "start_s": w[0], "end_s": w[1], **(
               {"error": results[i]["error"]} if "error" in results[i] else
               {"time_base": results[i]["info"]["time_base"], "raw": results[i]["info"]["raw_count"],
                "kept": results[i]["info"]["kept"], "dropped": results[i]["info"]["dropped"]})} for i, w in chosen],
           "ps2_object_hints": hints, "actions": merged}
    os.makedirs(final_dir, exist_ok=True)
    _atomic_write(os.path.join(final_dir, "dropped_actions.json"), json.dumps(
        {str(i): r.get("dropped", []) for i, r in results.items()}, indent=1))
    write_actions_csv(os.path.join(final_dir, "actions.csv"), stem, merged)
    if failed:
        _atomic_write(os.path.join(final_dir, "actions_incomplete.json"), json.dumps(doc, indent=1))
        log(f"[{stem}] {len(failed)} window(s) failed {failed}: rerun to retry only those (finished windows are cached)")
    else:
        _atomic_write(os.path.join(final_dir, "actions.json"), json.dumps(doc, indent=1))    # written last = done
        remote = holder["remote"]
        if not subset and remote is not None and cfg.delete_remote:
            if client.delete(remote.name):
                _atomic_write(state_path, json.dumps({"remote_name": None, "deleted_at": time.time()}))
                log("  uploaded file deleted from Gemini storage")
        if not subset and tmp_copy["path"] and os.path.exists(tmp_copy["path"]):
            os.remove(tmp_copy["path"])                              # data rule: no stray copies of the footage
        if not subset:
            finalize_clip(out_root, stem, cfg, log)
    log(f"[{stem}] {len(merged)} action(s) -> {final_dir}")
    return {"clip": stem, "status": status, "actions": merged, "failed_windows": failed, "dir": final_dir}


def finalize_clip(out_root: str, stem: str, cfg: PS3Config, log: Callable = print) -> Optional[dict]:
    """actions.json (+ PS2 contacts.csv when present) -> actions_final.json / actions_final.csv. CPU only, cheap,
    safe to rerun: always starts from the untouched model timeline."""
    from .refine import load_contacts, refine_actions
    base = ps3_dir(out_root, stem)
    doc = _read_json(os.path.join(base, "actions.json"))
    if not doc:
        return None
    contacts = load_contacts(os.path.join(out_root, stem, "ps2_final", "contacts.csv")) if cfg.refine_with_ps2 else []
    final, rep = refine_actions(doc["actions"], contacts, cfg) if contacts else ([dict(a) for a in doc["actions"]],
                                                                                 {"ps2_contacts": 0})
    out = {**doc, "actions": final, "refinement": rep}
    _atomic_write(os.path.join(base, "actions_final.json"), json.dumps(out, indent=1))
    write_actions_csv(os.path.join(base, "actions_final.csv"), stem, final)
    if contacts:
        log(f"  [{stem}] PS2 refinement: {rep.get('edges_snapped', 0)} edge(s) snapped, "
            f"{rep.get('hands_filled', 0)} hand(s) and {rep.get('objects_filled', 0)} object(s) filled from PS2")
    return out


# ------------------------------------------------------------------------------------------- many clips
def collect_actions(out_root: str, clips: Sequence[str]) -> int:
    """All finished clips' actions_final.csv (else actions.csv) -> OUT_ROOT/actions_all.csv. Returns the row count."""
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    n = 0
    for c in clips:
        stem = clip_stem(c)
        if not ps3_done(out_root, stem):
            continue
        d = ps3_dir(out_root, stem)
        src = os.path.join(d, "actions_final.csv")
        src = src if os.path.exists(src) else os.path.join(d, "actions.csv")
        with open(src, newline="", encoding="utf-8") as f:
            for row in list(csv.reader(f))[1:]:
                w.writerow(row)
                n += 1
    _atomic_write(os.path.join(out_root, "actions_all.csv"), buf.getvalue())
    return n


def run_all(clips: Sequence[str], out_root: str, cfg: PS3Config, log: Callable = print) -> List[dict]:
    """One clip at a time, in the order given (put demo clips first). Finished clips are skipped."""
    client = None
    results = []
    for c in clips:
        try:
            if client is None and not (ps3_done(out_root, clip_stem(c)) and not cfg.force):
                from .gemini_client import GeminiVideoClient
                client = GeminiVideoClient(cfg, log=log)
            results.append(process_clip(c, out_root, cfg, client=client, log=log))
        except Exception as exc:                                    # noqa: BLE001: one bad clip must not stop the rest
            log(f"[{clip_stem(c)}] FAILED: {exc}")
            results.append({"clip": clip_stem(c), "status": "failed", "error": str(exc)})
    log(f"{collect_actions(out_root, clips)} action rows in actions_all.csv")
    return results


def cleanup_remote(cfg: PS3Config, log: Callable = print) -> int:
    """Event-end data-use cleanup: delete EVERY file stored under this API key in Gemini file storage."""
    from .gemini_client import GeminiVideoClient
    n = GeminiVideoClient(cfg, log=log).delete_all_remote()
    log(f"deleted {n} file(s) from Gemini file storage")
    return n
