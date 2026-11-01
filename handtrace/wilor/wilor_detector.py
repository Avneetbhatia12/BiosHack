"""Official WiLoR hand detector (ultralytics YOLO, `pretrained_models/detector.pt`).

demo.py usage being reproduced:
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    bbox      = det.boxes.data[:4]            # xyxy, original image pixels
    is_right  = det.boxes.cls                 # class 0 -> left, 1 -> right

Additions (all configurable, none replace the official model):
  * batched inference over a chunk of frames
  * class-agnostic NMS: the detector occasionally fires both a 'left' and a 'right' box on the
    same hand; we keep the higher-scoring one (prevents a ghost second hand)
  * top-K by score
  * a `build_detector` factory so an external detector with the same output contract
    (N x [x1, y1, x2, y2, score, is_right]) can be plugged in as a fallback.
"""
from __future__ import annotations

import time
from typing import List, Protocol

import numpy as np


class HandDetector(Protocol):
    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]: ...


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def agnostic_nms(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """dets: N x 6 [x1,y1,x2,y2,score,is_right]. Greedy, class-agnostic."""
    if len(dets) <= 1 or iou_thr <= 0:
        return dets
    order = np.argsort(-dets[:, 4])
    dets = dets[order]
    iou = box_iou_matrix(dets[:, :4], dets[:, :4])
    keep, suppressed = [], np.zeros(len(dets), bool)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_thr
    return dets[keep]


class WiLoRHandDetector:
    def __init__(self, weights: str, device: str = "cuda", conf: float = 0.2, iou: float = 0.5,
                 imgsz: int = 640, half: bool = False, agnostic_iou: float = 0.7,
                 max_hands: int = 4):
        from ultralytics import YOLO
        from .setup_utils import trusted_torch_load
        with trusted_torch_load():
            self.model = YOLO(weights)
        self.model.to(device)
        self.device, self.conf, self.iou, self.imgsz = device, conf, iou, imgsz
        self.half = bool(half and str(device).startswith("cuda"))
        self.agnostic_iou, self.max_hands = agnostic_iou, max_hands
        self.right_classes = self._resolve_right_classes(getattr(self.model, "names", {}) or {})
        self.last_time_s = 0.0

    @staticmethod
    def _resolve_right_classes(names: dict) -> set:
        """Map class ids to handedness. Uses the model's own class names when they say
        left/right; otherwise falls back to demo.py's convention (cls == 1 -> right)."""
        right = {int(k) for k, v in names.items() if "right" in str(v).lower()}
        left = {int(k) for k, v in names.items() if "left" in str(v).lower()}
        if right and left:
            return right
        return {1}

    def describe(self) -> dict:
        return {"names": dict(getattr(self.model, "names", {})),
                "right_class_ids": sorted(self.right_classes)}

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        t0 = time.perf_counter()
        results = self.model(frames, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                             half=self.half, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            xyxy = b.xyxy.detach().float().cpu().numpy()
            score = b.conf.detach().float().cpu().numpy()
            cls = b.cls.detach().cpu().numpy().astype(int)
            is_right = np.array([1.0 if c in self.right_classes else 0.0 for c in cls], np.float32)
            dets = np.concatenate([xyxy, score[:, None], is_right[:, None]], 1).astype(np.float32)
            dets = agnostic_nms(dets, self.agnostic_iou)
            dets = dets[np.argsort(-dets[:, 4])][: self.max_hands]
            out.append(dets)
        self.last_time_s = time.perf_counter() - t0
        return out


def build_detector(cfg, device: str) -> HandDetector:
    """Factory. `cfg.detector_weights` may point to any ultralytics hand model whose classes
    are named left/right; the default is the official WiLoR detector."""
    return WiLoRHandDetector(cfg.detector_weights, device=device, conf=cfg.det_conf,
                             iou=cfg.det_iou, imgsz=cfg.det_imgsz, half=cfg.det_half,
                             agnostic_iou=cfg.agnostic_nms_iou, max_hands=cfg.max_hands_per_frame)
