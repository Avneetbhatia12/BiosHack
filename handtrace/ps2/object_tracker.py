"""Lightweight object tracker: stable object IDs ("O1", "O2", ...; hand IDs are integers, so the two
namespaces cannot collide).

Per frame, detections are matched to live tracks by cost = (1 - IoU) + centre distance / gate.
A pair is allowed only if IoU >= iou_gate or centre distance <= centre_gate * frame diagonal, and
(by default) the object TYPE agrees (a portable box never continues a fixed track). Names inside a
type may differ (OWLv2 can call one tool "wrench" then "spanner"); the track keeps a
confidence-weighted name vote and reports the majority name. Boxes are re-detected every frame,
so fixed objects follow the camera motion instead of keeping one static box.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .object_detector import box_iou

INF = 1e9


@dataclass
class ObjTrack:
    id: str
    bbox: np.ndarray
    object_type: str
    last_frame: int
    names: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    hits: int = 0
    missed: int = 0

    @property
    def name(self) -> str:
        return max(self.names, key=self.names.get)


class ObjectTracker:
    def __init__(self, frame_wh: tuple, iou_gate: float = 0.2, center_gate: float = 0.15,
                 max_missed: int = 10, type_mismatch: str = "forbid"):
        self.diag = float(np.hypot(*frame_wh))
        self.iou_gate, self.center_gate, self.max_missed = iou_gate, center_gate, max_missed
        self.type_mismatch = type_mismatch
        self.tracks: Dict[str, ObjTrack] = {}
        self.next_id = 1

    def _cost(self, tracks: List[ObjTrack], dets: List[dict]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.stack([t.bbox for t in tracks])
        db = np.array([d["bbox"] for d in dets], float)
        iou = box_iou(tb, db)
        tc = (tb[:, :2] + tb[:, 2:]) / 2
        dc = (db[:, :2] + db[:, 2:]) / 2
        dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / self.diag
        cost = (1 - iou) + dist / max(self.center_gate, 1e-9)
        ok = (iou >= self.iou_gate) | (dist <= self.center_gate)
        if self.type_mismatch == "forbid":
            ok &= np.array([[t.object_type == d["object_type"] for d in dets] for t in tracks])
        same_name = np.array([[d["object_name"] in t.names for d in dets] for t in tracks])
        cost = cost - 0.1 * same_name
        cost[~ok] = INF
        return cost

    def update(self, dets: List[dict], frame: int) -> List[dict]:
        """Returns copies of `dets` with object_id, object_name (track majority) and det_name."""
        tracks = list(self.tracks.values())
        cost = self._cost(tracks, dets)
        pairs = []
        if cost.size:
            for idx in np.argsort(cost, axis=None):
                i, j = divmod(int(idx), cost.shape[1])
                if cost[i, j] >= INF:
                    break
                if all(i != a and j != b for a, b in pairs):
                    pairs.append((i, j))
        assigned = {j: tracks[i] for i, j in pairs}
        out = []
        for j, d in enumerate(dets):
            t = assigned.get(j)
            if t is None:
                t = ObjTrack(f"O{self.next_id}", np.asarray(d["bbox"], float), d["object_type"], frame)
                self.next_id += 1
                self.tracks[t.id] = t
            t.bbox = np.asarray(d["bbox"], float)
            t.names[d["object_name"]] += float(d["confidence"])
            t.hits += 1
            t.missed = 0
            t.last_frame = frame
            out.append({**d, "object_id": t.id, "det_name": d["object_name"], "object_name": t.name})
        seen = {o["object_id"] for o in out}
        for t in list(self.tracks.values()):
            if t.id not in seen:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[t.id]
        return out
