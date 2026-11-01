"""One Euro filter (Casiez et al., CHI 2012), vectorised over all keypoint coordinates.

Each track ID owns its own filters for kp2d (21x2) and optionally kp3d (21x3). Filters are
created when a track is first seen and dropped when the tracker deletes the track, so a new
ID always starts unfiltered. A long dropout (> reset_after_s) also resets the filter so the
hand does not 'slide' from a stale position.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # Per-coordinate adaptive cutoff: fast-moving joints get less smoothing (less lag).
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class HandSmoother:
    """Per-track filter bank."""

    def __init__(self, cfg, reset_after_s: float = 0.5):
        self.cfg = cfg
        self.reset_after_s = reset_after_s
        self.f2d: Dict[int, OneEuroFilter] = {}
        self.f3d: Dict[int, OneEuroFilter] = {}

    def _get(self, bank, tid, mc, beta):
        if tid not in bank:
            bank[tid] = OneEuroFilter(mc, beta, self.cfg.one_euro_dcutoff)
        return bank[tid]

    def __call__(self, tid: int, t: float, kp2d: np.ndarray, kp3d: Optional[np.ndarray]):
        c = self.cfg
        if not c.smoothing:
            return kp2d, kp3d
        f = self._get(self.f2d, tid, c.one_euro_min_cutoff_2d, c.one_euro_beta_2d)
        if f.t_prev is not None and t - f.t_prev > self.reset_after_s:
            f.reset()
            self.f3d.pop(tid, None)
        k2 = f(kp2d, t).astype(np.float32)
        k3 = kp3d
        if kp3d is not None and c.smooth_3d:
            g = self._get(self.f3d, tid, c.one_euro_min_cutoff_3d, c.one_euro_beta_3d)
            k3 = g(kp3d, t).astype(np.float32)
        return k2, k3

    def drop_missing(self, live_ids):
        live = set(live_ids)
        for bank in (self.f2d, self.f3d):
            for tid in [k for k in bank if k not in live]:
                del bank[tid]
