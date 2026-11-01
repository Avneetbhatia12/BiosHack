"""Object detectors behind one interface: frames (BGR, original size) -> per-frame detections.

Each detection: {"bbox": [x1, y1, x2, y2] original pixels, "object_name", "object_type",
"confidence"}. frame / timestamp are attached by the PS2 pipeline.

Backends
  owlv2  Open-vocabulary OWLv2 (HF transformers), queried with the profile's prompts in
         configs/ps2_objects.json. Pure PyTorch, so it runs on Kaggle's preinstalled torch.
  sam3   SAM 3 (Meta, HF transformers): text-prompted detection WITH instance masks. Gated weights
         (HF_TOKEN), ~0.9B parameters. Masks are stored as compact box-relative bitmaps (masks.py)
         and used by the contact rule. Implemented from the transformers docs; not yet run.
  yolo   Ultralytics YOLO. With COCO weights only classes in `yolo_coco_map` are reported (COCO has
         no screwdriver / wrench / pliers / machine classes). With a model fine-tuned on our own
         labels, class names equal to the vocabulary's canonical names
         are reported with their vocabulary type.
Only detector output becomes a detection; nothing is invented when a backend finds nothing.
"""
from __future__ import annotations

import time
from typing import List, Protocol

import numpy as np

from .config import ObjectVocabulary, PS2Config


class ObjectDetector(Protocol):
    name: str
    last_time_s: float

    def __call__(self, frames: List[np.ndarray]) -> List[List[dict]]: ...


# ----------------------------------------------------------------------------- pure helpers
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS; returns kept indices, best score first."""
    order = list(np.argsort(-np.asarray(scores)))
    iou = box_iou(boxes, boxes)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if iou[i, j] <= iou_thr]
    return keep


def filter_detections(dets: List[dict], vocab: ObjectVocabulary, cfg: PS2Config,
                      frame_wh: tuple) -> List[dict]:
    """Per-class confidence, size limits, per-class NMS, then cross-class NMS for synonyms
    (e.g. 'table' and 'workbench' on the same surface), then top-K."""
    W, H = frame_wh
    area_f = float(W * H)
    out = []
    for d in dets:
        x1, y1, x2, y2 = (float(v) for v in d["bbox"])
        x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
        y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
        a = (x2 - x1) * (y2 - y1) / area_f
        thr = vocab.per_class_conf.get(d["object_name"], cfg.object_conf)
        if d["confidence"] >= thr and cfg.min_object_area_frac <= a <= cfg.max_object_area_frac:
            out.append({**d, "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(float(d["confidence"]), 4)})
    kept = []
    for name in {d["object_name"] for d in out}:
        group = [d for d in out if d["object_name"] == name]
        kept += [group[i] for i in nms(np.array([g["bbox"] for g in group]),
                                       np.array([g["confidence"] for g in group]), cfg.object_nms_iou)]
    if kept:
        idx = nms(np.array([d["bbox"] for d in kept]), np.array([d["confidence"] for d in kept]), 0.85)
        kept = [kept[i] for i in idx]
    kept.sort(key=lambda d: -d["confidence"])
    return kept[: cfg.max_objects_per_frame]


def decode_owl(logits: np.ndarray, pred_boxes: np.ndarray, queries: List[str], vocab: ObjectVocabulary,
               padded_side_px: float, min_score: float) -> List[dict]:
    """One image. logits (P, Q), pred_boxes (P, 4) normalized cx, cy, w, h relative to the square the
    image was padded to (OWLv2 pads bottom/right). `padded_side_px` = that square's side expressed in
    ORIGINAL frame pixels = max(original W, H)."""
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    labels = probs.argmax(-1)
    scores = probs.max(-1)
    keep = np.where(scores >= min_score)[0]
    out = []
    for p in keep:
        cx, cy, w, h = (float(v) for v in pred_boxes[p] * padded_side_px)    # plain floats: JSON-safe
        name = queries[int(labels[p])]
        out.append({"bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "object_name": name,
                    "object_type": vocab.object_type(name), "confidence": round(float(scores[p]), 4)})
    return out


def _resize_longest(img: np.ndarray, side: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    s = side / max(h, w)
    if abs(s - 1) < 1e-3:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- backends
class Owlv2Detector:
    name = "owlv2"

    def __init__(self, cfg: PS2Config, vocab: ObjectVocabulary, device: str = "cuda"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch, self.cfg, self.vocab, self.device = torch, cfg, vocab, device
        self.processor = Owlv2Processor.from_pretrained(cfg.owl_model)
        self.model = Owlv2ForObjectDetection.from_pretrained(cfg.owl_model).to(device).eval()
        self.half = bool(cfg.half and str(device).startswith("cuda"))
        if self.half:
            self.model = self.model.half()
        self.last_time_s = 0.0
        self.set_vocabulary(vocab)

    def set_vocabulary(self, vocab: ObjectVocabulary) -> None:
        """Switch prompts (e.g. per-clip workspace profile) without reloading the model."""
        self.vocab = vocab
        self.queries = list(vocab.prompts)
        self.text = [vocab.query_text(q) for q in self.queries]
        self.min_score = min([self.cfg.object_conf] + list(vocab.per_class_conf.values()))

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.cfg.owl_model, "profile": self.vocab.profile,
                "queries": dict(zip(self.queries, self.text)), "half": self.half,
                "input_longest_side": self.cfg.object_imgsz}

    def __call__(self, frames: List[np.ndarray]) -> List[List[dict]]:
        import cv2
        torch = self.torch
        t0 = time.perf_counter()
        small = [cv2.cvtColor(_resize_longest(f, self.cfg.object_imgsz), cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self.processor(text=[self.text] * len(small), images=small, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inputs)
        logits = out.logits.float().cpu().numpy()
        boxes = out.pred_boxes.float().cpu().numpy()
        res = []
        for i, f in enumerate(frames):
            H, W = f.shape[:2]
            dets = decode_owl(logits[i], boxes[i], self.queries, self.vocab, float(max(W, H)), self.min_score)
            res.append(filter_detections(dets, self.vocab, self.cfg, (W, H)))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.last_time_s = time.perf_counter() - t0
        return res

    def close(self):
        del self.model
        if str(self.device).startswith("cuda"):
            self.torch.cuda.empty_cache()


class Sam3Detector:
    """SAM 3 text-prompted instance segmentation. Vision features are computed once per frame and
    reused for every prompt; prompts are tokenized once per vocabulary."""
    name = "sam3"

    def __init__(self, cfg: PS2Config, vocab: ObjectVocabulary, device: str = "cuda"):
        import os
        import torch
        from transformers import Sam3Model, Sam3Processor
        token = os.environ.get("HF_TOKEN")              # gated model; never hard-code the token
        self.torch, self.cfg, self.device = torch, cfg, device
        self.processor = Sam3Processor.from_pretrained(cfg.sam3_model, token=token)
        self.half = bool(cfg.half and str(device).startswith("cuda"))
        dtype = torch.float16 if self.half else torch.float32
        self.model = Sam3Model.from_pretrained(cfg.sam3_model, token=token, torch_dtype=dtype).to(device).eval()
        self.last_time_s = 0.0
        self.set_vocabulary(vocab)

    def set_vocabulary(self, vocab: ObjectVocabulary) -> None:
        self.vocab = vocab
        self.queries = list(vocab.prompts)
        self.text = [vocab.query.get(q, q.replace("_", " ")) for q in self.queries]   # short noun phrases
        self.min_score = min([self.cfg.object_conf] + list(vocab.per_class_conf.values()))
        self._text_inputs = [self.processor(text=t, return_tensors="pt") for t in self.text]

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.cfg.sam3_model, "profile": self.vocab.profile,
                "queries": dict(zip(self.queries, self.text)), "half": self.half, "masks": True}

    def __call__(self, frames: List[np.ndarray]) -> List[List[dict]]:
        import cv2
        from .masks import mask_to_grid
        torch = self.torch
        t0 = time.perf_counter()
        res = []
        for f in frames:
            H, W = f.shape[:2]
            small = cv2.cvtColor(_resize_longest(f, self.cfg.object_imgsz), cv2.COLOR_BGR2RGB)
            scale = max(W, H) / max(small.shape[:2])
            img_in = self.processor(images=small, return_tensors="pt").to(self.device)
            pv = img_in["pixel_values"].half() if self.half else img_in["pixel_values"]
            dets = []
            with torch.inference_mode():
                vision = self.model.get_vision_features(pixel_values=pv)
                for name, ti in zip(self.queries, self._text_inputs):
                    out = self.model(vision_embeds=vision, **{k: v.to(self.device) for k, v in ti.items()})
                    r = self.processor.post_process_instance_segmentation(
                        out, threshold=self.min_score, mask_threshold=self.cfg.sam3_mask_threshold,
                        target_sizes=img_in.get("original_sizes").tolist())[0]
                    for box, score, mask in zip(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(),
                                                r["masks"].cpu().numpy()):
                        bb = [float(v) * scale for v in box]
                        m = mask.astype(np.uint8)
                        full = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if scale != 1 else m) > 0
                        dets.append({"bbox": bb, "object_name": name, "object_type": self.vocab.object_type(name),
                                     "confidence": round(float(score), 4),
                                     "mask_grid": mask_to_grid(full, bb, self.cfg.mask_grid)})
            res.append(filter_detections(dets, self.vocab, self.cfg, (W, H)))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.last_time_s = time.perf_counter() - t0
        return res

    def close(self):
        del self.model
        if str(self.device).startswith("cuda"):
            self.torch.cuda.empty_cache()


class YoloDetector:
    """Ultralytics YOLO. COCO weights: only yolo_coco_map classes. Fine-tuned weights: classes whose
    names are canonical vocabulary names (type from the vocabulary)."""
    name = "yolo"

    def __init__(self, cfg: PS2Config, vocab: ObjectVocabulary, device: str = "cuda"):
        from ultralytics import YOLO
        self.cfg, self.vocab = cfg, vocab
        self.model = YOLO(cfg.yolo_model)
        self.model.to(device)
        self.names = dict(self.model.names)
        self.half = bool(cfg.half and str(device).startswith("cuda"))
        self.last_time_s = 0.0

    def set_vocabulary(self, vocab: ObjectVocabulary) -> None:
        self.vocab = vocab

    def _type(self, name: str):
        return self.vocab.type_of.get(name) or self.vocab.coco_map.get(name)

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.cfg.yolo_model,
                "reportable_classes": sorted(n for n in self.names.values() if self._type(n)),
                "ignored_classes": sorted(n for n in self.names.values() if not self._type(n))}

    def __call__(self, frames: List[np.ndarray]) -> List[List[dict]]:
        t0 = time.perf_counter()
        results = self.model(frames, conf=min([self.cfg.object_conf] + list(self.vocab.per_class_conf.values())),
                             iou=self.cfg.object_nms_iou, imgsz=self.cfg.object_imgsz, half=self.half, verbose=False)
        out = []
        for f, r in zip(frames, results):
            H, W = f.shape[:2]
            dets = []
            if r.boxes is not None and len(r.boxes):
                for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                  r.boxes.cls.cpu().numpy()):
                    name = self.names[int(c)]
                    if self._type(name):
                        dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "object_name": name,
                                     "object_type": self._type(name), "confidence": round(float(s), 4)})
            out.append(filter_detections(dets, self.vocab, self.cfg, (W, H)))
        self.last_time_s = time.perf_counter() - t0
        return out

    def close(self):
        del self.model


def build_object_detector(cfg: PS2Config, vocab: ObjectVocabulary, device: str) -> ObjectDetector:
    if cfg.backend == "owlv2":
        return Owlv2Detector(cfg, vocab, device)
    if cfg.backend == "sam3":
        return Sam3Detector(cfg, vocab, device)
    if cfg.backend == "yolo":
        return YoloDetector(cfg, vocab, device)
    raise ValueError(f"unknown object detector backend {cfg.backend!r} (owlv2 | sam3 | yolo)")
