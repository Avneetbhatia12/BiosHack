"""Zip every clip's PS1 / PS2 / PS3 / PS5 results, the audit, review players, dashboard and cross-clip CSVs.

    python scripts/package_results.py --clips data/clips --out outputs --dest outputs/results.zip

The zip holds numbers only (no video, no frames). It is still derived from private footage: share it only with
the team and the organisers, never publicly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="+", required=True, help="folders or video files (for the clip list)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--dest", default=None, help="zip path (default: <out>/results.zip)")
    ap.add_argument("--include-detections", action="store_true", help="also include per-frame object detections")
    a = ap.parse_args(argv)
    from handtrace import runner
    files = [c for c in a.clips if os.path.isfile(c)]
    clips = runner.order_clips(files + runner.discover_clips([c for c in a.clips if os.path.isdir(c)]))
    print(runner.package_results(a.out, clips, a.dest or os.path.join(a.out, "results.zip"), a.include_detections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
