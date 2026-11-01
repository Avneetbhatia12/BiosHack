"""Temporal windows over the original video timeline."""
from __future__ import annotations

from typing import List, Tuple


def make_windows(duration_s: float, window_s: float, overlap_s: float) -> List[Tuple[float, float]]:
    """Overlapping (start, end) windows in ORIGINAL video seconds. The last window ends exactly at the video end;
    a tiny tail is absorbed into the previous window instead of becoming its own window."""
    if duration_s <= 0:
        return []
    if duration_s <= window_s * 1.25:
        return [(0.0, round(duration_s, 3))]
    step = max(window_s - overlap_s, 1.0)
    out, start = [], 0.0
    while True:
        end = min(start + window_s, duration_s)
        out.append((round(start, 3), round(end, 3)))
        if end >= duration_s - 1e-6:
            break
        start += step
        if duration_s - start < overlap_s + 5.0:        # remaining tail is mostly overlap: stretch the last window
            out[-1] = (out[-1][0], round(duration_s, 3))
            break
    return out
