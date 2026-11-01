"""Model-independent multi-hand tracker.

Inputs per processed frame: N x 6 detections [x1, y1, x2, y2, score, is_right].
No dependency on WiLoR / torch.

Association (ByteTrack-style two stages, Hungarian assignment):
  cost = w_iou * (1 - IoU(predicted_track_box, det))
       + w_center * min(1, centre_dist / (center_gate * diag))
       + side_penalty * [track side != det side]        (soft handedness cue)
       + 0.1 * min(1, |log(area ratio)|)                 (size consistency)
  * the track box is predicted with a damped constant-velocity model (previous-frame motion)
  * stage 1: all live tracks (incl. temporarily lost ones = ID recovery) vs high-score detections
  * stage 2: still-unmatched tracks vs low-score detections (keeps IDs through blur, where the
    detector score drops)
  * unmatched high-score detections start new tracks with fresh IDs
  * a track is deleted after `max_missed` processed frames without a match
Handedness (side_mode="track_vote"): each track keeps an EMA vote of detector labels; an isolated
detector label that contradicts a confident track is overridden (fixes 1-2 frame L/R flicker),
while a sustained change is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

INF = 1e6


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return [(i, j) for i, j in zip(r, c) if cost[i, j] < INF]
    pairs, used_r, used_c = [], set(), set()          # greedy fallback
    for idx in np.argsort(cost, axis=None):
        i, j = divmod(int(idx), cost.shape[1])
        if cost[i, j] >= INF:
            break
        if i not in used_r and j not in used_c:
            pairs.append((i, j)); used_r.add(i); used_c.add(j)
    return pairs


@dataclass
class Track:
    id: int
    bbox: np.ndarray
    side_vote: float
    score: float
    last_frame: int
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: int = 1
    missed: int = 0
    side_flips: int = 0

    @property
    def center(self) -> np.ndarray:
        return (self.bbox[:2] + self.bbox[2:4]) / 2

    def predicted_bbox(self) -> np.ndarray:
        steps = self.missed + 1
        damp = 0.8 ** self.missed
        shift = self.vel * min(steps, 5) * damp
        return self.bbox + np.array([shift[0], shift[1], shift[0], shift[1]], np.float32)

    @property
    def side_is_right(self) -> bool:
        return self.side_vote >= 0


@dataclass
class Assignment:
    det_index: int
    track_id: int
    is_right: bool          # resolved side (what WiLoR will be run with)
    det_is_right: bool      # raw detector label (after optional global swap)
    new_track: bool
    recovered_after: int    # >0 if this matched a track that had been missing that many frames
    side_corrected: bool


class HandTracker:
    def __init__(self, cfg, frame_wh: Tuple[int, int]):
        self.cfg = cfg
        self.diag = float(np.hypot(*frame_wh))
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.events = {"births": [], "deaths": [], "recoveries": [], "side_corrections": 0}

    # ------------------------------------------------------------------
    def _cost(self, tracks: List[Track], dets: np.ndarray, low_stage: bool) -> np.ndarray:
        c = self.cfg
        if not tracks or len(dets) == 0:
            return np.zeros((len(tracks), len(dets)))
        pb = np.stack([t.predicted_bbox() for t in tracks])
        iou = _iou(pb, dets[:, :4])
        pc = (pb[:, :2] + pb[:, 2:4]) / 2
        dc = (dets[:, :2] + dets[:, 2:4]) / 2
        dist = np.linalg.norm(pc[:, None] - dc[None], axis=2) / self.diag
        miss = np.array([t.missed for t in tracks], np.float32)[:, None]
        gate = c.center_gate * (1 + 0.15 * np.minimum(miss, 10))
        if low_stage:
            gate = np.minimum(gate, c.reid_center_gate)
        ta = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
        da = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        size = np.minimum(1, np.abs(np.log(np.maximum(da[None], 1) / np.maximum(ta[:, None], 1))))
        side_t = np.array([t.side_is_right for t in tracks])[:, None]
        side_d = (dets[:, 5] > 0.5)[None, :]
        cost = (c.match_iou_weight * (1 - iou) + c.match_center_weight * np.minimum(1, dist / gate)
                + c.match_side_penalty * (side_t != side_d) + 0.1 * size)
        bad = (dist > gate) & (iou <= 0.0)
        if low_stage:
            bad |= iou < 0.1
        cost[bad | (cost > c.match_max_cost)] = INF
        return cost

    def _resolve_side(self, t: Track, det_right: bool, score: float) -> Tuple[bool, bool]:
        c = self.cfg
        sign = 1.0 if det_right else -1.0
        if c.side_mode == "detector":
            t.side_vote = sign * score
            return det_right, False
        corrected = False
        side = det_right
        if (t.side_vote >= 0) != det_right and abs(t.side_vote) >= c.side_flip_margin:
            side, corrected = t.side_vote >= 0, True
        t.side_vote = c.side_vote_decay * t.side_vote + (1 - c.side_vote_decay) * sign * score
        if not corrected and (t.side_vote >= 0) != det_right:
            # accepted a change of label (sustained evidence): snap the vote to the new side
            t.side_vote = sign * max(abs(t.side_vote), 0.5 * score)
            t.side_flips += 1
        return side, corrected

    def _update(self, t: Track, det: np.ndarray, frame_idx: int):
        new_c = (det[:2] + det[2:4]) / 2
        steps = max(1, frame_idx - t.last_frame) if t.last_frame >= 0 else 1
        if t.hits > 0:
            v = (new_c - t.center) / steps
            t.vel = 0.6 * t.vel + 0.4 * v.astype(np.float32)
        t.bbox = det[:4].astype(np.float32).copy()
        t.score = float(det[4]); t.hits += 1; t.missed = 0; t.last_frame = frame_idx

    # ------------------------------------------------------------------
    def step(self, dets: np.ndarray, frame_idx: int) -> List[Assignment]:
        c = self.cfg
        dets = np.asarray(dets, np.float32).reshape(-1, 6)
        if c.handedness_swap and len(dets):
            dets = dets.copy(); dets[:, 5] = 1.0 - dets[:, 5]
        high = np.where(dets[:, 4] >= c.det_high_conf)[0]
        low = np.where(dets[:, 4] < c.det_high_conf)[0]
        tracks = list(self.tracks.values())
        matched_t, out = set(), []

        def commit(t: Track, di: int, new: bool):
            gap = t.missed if not new else 0
            det_right = bool(dets[di, 5] > 0.5)
            side, corr = self._resolve_side(t, det_right, float(dets[di, 4]))
            if not new:
                self._update(t, dets[di], frame_idx)
            if gap > 0:
                self.events["recoveries"].append((frame_idx, t.id, gap))
            if corr:
                self.events["side_corrections"] += 1
            out.append(Assignment(int(di), t.id, side, det_right, new, gap, corr))
            matched_t.add(t.id)

        # stage 1: high-score detections
        cost = self._cost(tracks, dets[high], low_stage=False)
        for ti, dj in _assign(cost):
            commit(tracks[ti], int(high[dj]), False)
        un_high = [int(high[j]) for j in range(len(high))
                   if int(high[j]) not in {a.det_index for a in out}]
        # stage 2: low-score detections vs unmatched tracks
        rem = [t for t in tracks if t.id not in matched_t]
        cost = self._cost(rem, dets[low], low_stage=True)
        for ti, dj in _assign(cost):
            commit(rem[ti], int(low[dj]), False)
        # births
        for di in un_high:
            d = dets[di]
            t = Track(id=self.next_id, bbox=d[:4].copy(), side_vote=(1.0 if d[5] > 0.5 else -1.0) * float(d[4]),
                      score=float(d[4]), last_frame=frame_idx)
            self.next_id += 1
            self.tracks[t.id] = t
            self.events["births"].append((frame_idx, t.id, bool(d[5] > 0.5), t.center.tolist()))
            commit(t, di, True)
        # age / delete
        for t in list(self.tracks.values()):
            if t.id not in matched_t:
                t.missed += 1
                if t.missed > c.max_missed:
                    self.events["deaths"].append((frame_idx, t.id, t.side_is_right, t.center.tolist(),
                                                  t.last_frame))
                    del self.tracks[t.id]
        out.sort(key=lambda a: a.track_id)
        return out

    def live_ids(self) -> List[int]:
        return list(self.tracks)
