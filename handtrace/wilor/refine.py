"""Re-run WiLoR on hands whose side was changed by post-processing.

WiLoR mirrors the crop of a hand it is told is "left", so a hand reconstructed with the wrong side
gets a mirrored fit (thumb and pinky can swap). After `handtrace.postprocess` corrects a side,
this re-reconstructs only those hands with the corrected side. Smoothing is not re-applied to them.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .video_loader import VideoReader


def rereconstruct_corrected(video_path: str, records: List[Dict], reconstructor, kp3d_frame: str =
                            "wrist_relative", chunk_size: int = 8, log=print) -> int:
    """Update kp2d / kp3d in place for every real (not interpolated) hand with side_corrected.

    `reconstructor.reconstruct(frames, [HandRequest]) -> [{"joints", "cam_t", "kp2d"}]`, as in
    wilor_inference.WiLoRReconstructor. Returns the number of hands re-reconstructed."""
    from .wilor_inference import HandRequest

    todo = {r["frame"]: [h for h in r["hands"] if h.get("side_corrected") and not h.get("interpolated")]
            for r in records}
    todo = {f: hs for f, hs in todo.items() if hs}
    if not todo:
        return 0
    last = max(todo)
    done, batch = 0, []

    def flush():
        nonlocal done
        frames = [img for _, img, _ in batch]
        reqs, owners = [], []
        for slot, (_, _, hands) in enumerate(batch):
            for h in hands:
                reqs.append(HandRequest(slot, np.asarray(h["bbox"], np.float32), h["side"] == "right"))
                owners.append(h)
        for h, r in zip(owners, reconstructor.reconstruct(frames, reqs)):
            joints, cam_t = np.asarray(r["joints"]), np.asarray(r["cam_t"])
            kp3d = joints - joints[0] if kp3d_frame == "wrist_relative" else joints + cam_t
            h["kp2d"] = np.round(np.asarray(r["kp2d"], np.float64), 2).tolist()
            h["kp3d"] = np.round(kp3d.astype(np.float64), 5).tolist()
            if kp3d_frame == "wrist_relative":
                h["kp3d_root_cam"] = np.round((joints[0] + cam_t).astype(np.float64), 5).tolist()
            h["rereconstructed"] = True
            done += 1
        batch.clear()

    for fidx, _, img in VideoReader(video_path, 0, last + 1):
        if fidx in todo:
            batch.append((fidx, img, todo[fidx]))
            if len(batch) >= chunk_size:
                flush()
    if batch:
        flush()
    log(f"[refine] re-reconstructed {done} side-corrected hands in {len(todo)} frames")
    return done
