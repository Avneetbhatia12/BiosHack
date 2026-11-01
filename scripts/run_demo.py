"""Render the demo videos for the demo clips: all layers (PS1 + PS2 + PS3 + PS5) and PS1 raw vs post-processed.

    python scripts/run_demo.py --clips data/clips --out outputs --demo 000003-004 --start 35 --seconds 15

The rendered MP4s contain company footage: keep them in outputs/ (gitignored) and upload them ONLY to the event's
private Drive folder. The hackathon allows 2 minutes of demo video in total.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="+", required=True, help="folders or video files")
    ap.add_argument("--out", default="outputs", help="pipeline output root (results are read from here)")
    ap.add_argument("--demo", nargs="*", default=[], help="name parts of the demo clips (default: first clip)")
    ap.add_argument("--start", type=float, default=0.0, help="segment start in seconds")
    ap.add_argument("--seconds", type=float, default=15.0, help="segment length per video")
    ap.add_argument("--dest", default=None, help="folder for the MP4s (default: <out>/demo)")
    a = ap.parse_args(argv)

    from handtrace import runner
    from handtrace.review import ClipView, render_review_segment

    files = [c for c in a.clips if os.path.isfile(c)]
    clips = runner.order_clips(files + runner.discover_clips([c for c in a.clips if os.path.isdir(c)]), a.demo)
    targets = [c for c in clips if any(k.lower() in os.path.basename(c).lower() for k in a.demo)] or clips[:1]
    if not targets:
        print("no clips found"); return 2
    dest = a.dest or os.path.join(a.out, "demo")
    os.makedirs(dest, exist_ok=True)
    for clip in targets:
        stem = runner.stem_of(clip)
        view = ClipView(clip, a.out)
        render_review_segment(view, os.path.join(dest, f"{stem}_all_layers.mp4"), a.start, a.start + a.seconds, width=1280)
        render_review_segment(view, os.path.join(dest, f"{stem}_ps1_raw_vs_final.mp4"), a.start, a.start + a.seconds,
                              layers={"ps1", "hud"}, width=1920, compare_raw=True)
        view.close()
    total = len(targets) * a.seconds * 2
    print(f"{len(targets) * 2} videos in {dest} ({total:.0f} s total{'; over the 120 s limit' if total > 120 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
