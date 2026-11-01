"""Video I/O: threaded OpenCV reader (decode overlaps GPU work) and an MP4 writer that keeps the
source FPS / resolution and re-encodes to H.264 with ffmpeg when available."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = VideoInfo(path, float(fps), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


class VideoReader:
    """Yields (frame_index, timestamp_s, bgr_image) for EVERY frame in order.

    Timestamps are frame_index / fps (constant-frame-rate clips, as in this dataset); frames are
    never dropped, so the JSONL can contain exactly one record per frame."""

    def __init__(self, path: str, start: int = 0, max_frames: Optional[int] = None,
                 prefetch: int = 32):
        self.info = probe(path)
        self.start, self.max_frames, self.prefetch = start, max_frames, prefetch

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def worker():
            cap = cv2.VideoCapture(self.info.path)
            if self.start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            i, n = self.start, 0
            try:
                while not stop.is_set():
                    if self.max_frames is not None and n >= self.max_frames:
                        break
                    ok, img = cap.read()
                    if not ok:
                        break
                    q.put((i, i / self.info.fps, img))
                    i += 1; n += 1
            finally:
                cap.release()
                q.put(None)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            stop.set()
            while th.is_alive():
                try:
                    q.get_nowait()
                except queue.Empty:
                    th.join(timeout=0.05)


class VideoWriter:
    def __init__(self, path: str, fps: float, size_wh: Tuple[int, int], crf: int = 23):
        self.final_path, self.fps, self.size, self.crf = path, fps, size_wh, crf
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tmp_path = path[:-4] + "_raw.mp4" if path.endswith(".mp4") else path + "_raw.mp4"
        self.w = cv2.VideoWriter(self.tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size_wh)
        if not self.w.isOpened():
            raise IOError("cv2.VideoWriter failed to open")
        self.n = 0

    def write(self, img: np.ndarray):
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        self.w.write(img)
        self.n += 1

    def close(self) -> str:
        self.w.release()
        ff = shutil.which("ffmpeg")
        if ff:
            cmd = [ff, "-y", "-loglevel", "error", "-i", self.tmp_path, "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                   "-r", f"{self.fps:.6f}", "-movflags", "+faststart", self.final_path]
            if subprocess.run(cmd).returncode == 0:
                os.remove(self.tmp_path)
                return self.final_path
        os.replace(self.tmp_path, self.final_path)   # mp4v fallback (plays in VLC)
        return self.final_path
