"""One command for the whole pipeline (the README's "how to run"):

    python -m handtrace.cli all --clips data/clips --out outputs --wilor WiLoR --demo 000060-010
    python -m handtrace.cli ps1 --config configs/example_config.json      # one stage, settings from a JSON file
    python -m handtrace.cli --clips data/clips --stages ps1,ps2             # explicit stage list (same as before)

Command-line flags override values from --config. Keys in the JSON file are the flag names with underscores
(e.g. "obj_backend", "det_imgsz"); see configs/example_config.json.

Stages run in order and each one skips clips it has already finished, so the command can simply be re-run:
    ps1  WiLoR hand tracking + post-processing          (GPU; one worker per GPU)
    ps2  open-vocabulary objects + fingertip contacts   (GPU)
    ps3  Gemini task-step timeline + PS2 refinement      (needs GEMINI_API_KEY in the environment; CPU)
    ps5  quality filter, safety stretches, event index, dashboard, review players for the demo clips (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


STAGES = ("ps1", "ps2", "ps3", "ps5")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HandTrace: PS1 + PS2 + PS3 + PS5 on a folder of clips")
    ap.add_argument("stage", nargs="?", choices=STAGES + ("all",), help="stage to run (default: --stages)")
    ap.add_argument("--config", help="JSON file with default values for any option below")
    ap.add_argument("--clips", nargs="+", help="folders (searched recursively) or video files")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--wilor", default="WiLoR", help="WiLoR checkout with pretrained_models/ and mano_data/")
    ap.add_argument("--stages", default="ps1,ps2,ps3,ps5")
    ap.add_argument("--demo", nargs="*", default=[], help="name parts of demo clips (processed first)")
    ap.add_argument("--mode", default="fp32", choices=["fp32", "fp16", "fast"])
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--handedness-swap", action="store_true")
    ap.add_argument("--obj-backend", default="owlv2", choices=["owlv2", "sam3", "yolo"])
    ap.add_argument("--ps3-model", default="auto")
    ap.add_argument("--ps3-fps", type=float, default=2.0)
    ap.add_argument("--small-copies", nargs="*", default=[], help="folders with 480p copies for the PS3 upload")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test: only the first N frames")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    """Parse flags; values from --config are defaults that explicit flags override."""
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        with open(pre.config, encoding="utf-8") as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        known = {a.dest for a in ap._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            ap.error(f"unknown keys in {pre.config}: {unknown}")
        if isinstance(cfg.get("stages"), list):
            cfg["stages"] = ",".join(cfg["stages"])
        ap.set_defaults(**cfg)
    a = ap.parse_args(argv)
    if a.stage:
        a.stages = ",".join(STAGES) if a.stage == "all" else a.stage
    if not a.clips:
        ap.error("--clips is required (on the command line or in --config)")
    bad = [s for s in a.stages.split(",") if s.strip() and s.strip() not in STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)

    from . import runner
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    files = [c for c in a.clips if os.path.isfile(c)]
    clips = runner.order_clips(files + runner.discover_clips([c for c in a.clips if os.path.isdir(c)]), a.demo)
    if not clips:
        print("no videos found under", a.clips); return 2
    os.makedirs(a.out, exist_ok=True)
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    print(f"{len(clips)} clip(s) -> {a.out} | stages {stages}")

    s = runner.Settings(clips=clips, out_root=a.out, wilor_root=a.wilor, priority=a.demo, mode=a.mode,
                        det_imgsz=a.det_imgsz, handedness_swap=a.handedness_swap, frame_step=a.frame_step,
                        ps2={"backend": a.obj_backend}, max_frames=a.max_frames,
                        prompts_path=os.path.join(code_root, "configs", "ps2_objects.json"))
    settings_path = os.path.join(a.out, "settings.json")
    rc = 0
    for st in ("ps1", "ps2"):
        if st in stages:
            r = runner.launch(st, s, settings_path, code_root, workers=a.workers, poll_s=60)
            rc |= int(any(r["returncodes"]))
    if "ps1" in stages or "ps2" in stages:
        runner.write_summary(a.out, clips)
        runner.collect_contacts(a.out, clips)
    if "ps3" in stages:
        from . import ps3
        cfg = ps3.PS3Config(model=a.ps3_model, fps=a.ps3_fps, small_copy_roots=a.small_copies)
        res = ps3.run_all(clips, a.out, cfg)
        rc |= int(any(r.get("status") == "failed" for r in res))
    if "ps5" in stages:
        from .ps5 import dashboard, index, quality, safety
        from .review_html import export_review_html
        stems = [runner.stem_of(c) for c in clips]
        quality.analyse_all(clips, a.out)
        safety.analyse_all(clips, a.out)
        index.build_index(a.out, stems)
        print("dashboard:", dashboard.write_html(a.out, stems))
        for c in [c for c in clips if any(k.lower() in os.path.basename(c).lower() for k in a.demo)] or clips[:1]:
            print("review player:", export_review_html(c, a.out))
    print(f"done in {(time.time() - t0) / 60:.1f} min, exit code {rc}")
    with open(os.path.join(a.out, "cli_run.json"), "w") as f:
        json.dump({"argv": sys.argv, "stages": stages, "clips": len(clips), "minutes": round((time.time() - t0) / 60, 1),
                   "exit_code": rc}, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
