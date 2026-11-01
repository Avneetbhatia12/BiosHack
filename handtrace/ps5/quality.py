"""PS5 automatic quality filter: which clips (and which stretches of a clip) are too blurry, too dark, or have no
hands, so they can be dropped before labelling, training or review.

Per sampled frame (default 2 per second, on a 480p copy in memory):
    sharpness   variance of the Laplacian (low = blurry; motion blur from head movement is the usual cause)
    brightness  mean grey level 0..255
    hands       number of PS1 hands at that frame (0 when PS1 has not run)
Thresholds are absolute defaults that can be tuned; `relative=True` also flags frames whose sharpness is below a
fraction of the clip's own median (robust to cameras that are softer overall).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results as R


@dataclass
class QualityConfig:
    samples_per_s: float = 2.0
    analysis_height: int = 480
    blur_abs: float = 40.0              # Laplacian variance below this = blurry (480p grey)
    blur_rel: float = 0.35              # ... or below this share of the clip's median sharpness
    relative: bool = True
    dark: float = 45.0                  # mean grey below this = too dark
    bright: float = 235.0               # above this = washed out
    min_bad_s: float = 1.0              # a bad stretch shorter than this is not reported
    max_bad_share: float = 0.5          # a clip with more bad time than this share is "drop"
    min_hand_share: float = 0.3         # a clip with hands in fewer sampled frames than this is "drop"


def frame_measures(img: np.ndarray, height: int = 480) -> Dict[str, float]:
    import cv2
    if img.shape[0] > height:
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()), "brightness": float(g.mean())}


def analyse_clip(clip_path: str, out_root: str, cfg: Optional[QualityConfig] = None, write: bool = True,
                 log=print) -> Dict:
    import cv2
    cfg = cfg or QualityConfig()
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    hands = {r["frame"]: len(r["hands"]) for r in R.load_hands(out_root, stem)}
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / cfg.samples_per_s)))
    rows = []
    for f in range(0, n, step):
        if step > 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        else:
            while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < f:
                cap.grab()
        ok, img = cap.read()
        if not ok:
            break
        m = frame_measures(img, cfg.analysis_height)
        rows.append({"frame": f, "t": round(f / fps, 2), **m, "hands": hands.get(f, 0) if hands else None})
    cap.release()
    if not rows:
        raise IOError(f"could not read frames from {clip_path}")
    sharp = np.array([r["sharpness"] for r in rows])
    med = float(np.median(sharp))
    for r in rows:
        reasons = []
        if r["sharpness"] < cfg.blur_abs or (cfg.relative and r["sharpness"] < cfg.blur_rel * med):
            reasons.append("blurry")
        if r["brightness"] < cfg.dark:
            reasons.append("dark")
        if r["brightness"] > cfg.bright:
            reasons.append("washed out")
        if r["hands"] == 0:
            reasons.append("no hands")
        r["reasons"] = reasons
        r["bad_image"] = bool({"blurry", "dark", "washed out"} & set(reasons))
    dt = 1.0 / cfg.samples_per_s
    segs, cur = [], None
    for r in rows:
        bad = r["bad_image"] or (r["hands"] == 0)
        if bad and cur is None:
            cur = {"t_start": r["t"], "t_end": r["t"] + dt, "reasons": set(r["reasons"])}
        elif bad:
            cur["t_end"] = r["t"] + dt; cur["reasons"] |= set(r["reasons"])
        elif cur is not None:
            segs.append(cur); cur = None
    if cur is not None:
        segs.append(cur)
    segs = [{"t_start": round(s["t_start"], 2), "t_end": round(s["t_end"], 2), "reason": ", ".join(sorted(s["reasons"]))}
            for s in segs if s["t_end"] - s["t_start"] >= cfg.min_bad_s]
    dur = n / fps if fps else 0.0
    bad_s = sum(s["t_end"] - s["t_start"] for s in segs)
    have_hands = [r["hands"] for r in rows if r["hands"] is not None]
    hand_share = (sum(h > 0 for h in have_hands) / len(have_hands)) if have_hands else None
    image_bad_share = float(np.mean([r["bad_image"] for r in rows]))
    verdict, why = "keep", []
    if dur and bad_s / dur > cfg.max_bad_share:
        verdict = "drop"; why.append(f"{bad_s / dur:.0%} of the clip is blurry / dark / without hands")
    if hand_share is not None and hand_share < cfg.min_hand_share:
        verdict = "drop"; why.append(f"hands in only {hand_share:.0%} of sampled frames")
    if verdict == "keep" and (image_bad_share > 0.2 or (hand_share is not None and hand_share < 0.6)):
        verdict = "review"; why.append("partly blurry / dark or hands often out of view")
    out = {"clip": stem, "duration_s": round(dur, 2), "sampled_frames": len(rows), "median_sharpness": round(med, 1),
           "median_brightness": round(float(np.median([r["brightness"] for r in rows])), 1),
           "blurry_share": round(float(np.mean(["blurry" in r["reasons"] for r in rows])), 3),
           "dark_share": round(float(np.mean(["dark" in r["reasons"] for r in rows])), 3),
           "hand_share": round(hand_share, 3) if hand_share is not None else None,
           "bad_time_s": round(bad_s, 1), "verdict": verdict, "why": why, "bad_segments": segs,
           "config": asdict(cfg), "series": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()
                                              if k != "reasons"} for r in rows]}
    if write:
        p = R.paths(out_root, stem)
        os.makedirs(p["ps5"], exist_ok=True)
        import json
        with open(p["quality"] + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(p["quality"] + ".tmp", p["quality"])
    log(f"[quality] {stem}: {verdict} | sharpness {med:.0f} | hands in {hand_share if hand_share is not None else 'n/a'} "
        f"| {len(segs)} bad stretch(es), {bad_s:.0f} s")
    return out


def analyse_all(clips: Sequence[str], out_root: str, cfg: Optional[QualityConfig] = None, force: bool = False,
                log=print) -> List[Dict]:
    rows = []
    for c in clips:
        stem = os.path.splitext(os.path.basename(c))[0]
        q = None if force else R.read_json(R.paths(out_root, stem)["quality"])
        if not q:
            try:
                q = analyse_clip(c, out_root, cfg, log=log)
            except Exception as e:                               # noqa: BLE001
                log(f"[quality] {stem}: FAILED {e!r}")
                continue
        rows.append({k: v for k, v in q.items() if k not in ("series", "config", "bad_segments")}
                    | {"why": "; ".join(q.get("why", [])), "bad_stretches": len(q.get("bad_segments", []))})
    if rows:
        R.write_csv(os.path.join(out_root, "ps5_quality_all.csv"), rows)
    return rows
