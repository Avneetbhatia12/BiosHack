"""PS2 configuration: object prompts (JSON file) and every tunable of the PS2 pass."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "configs" / "ps2_objects.json"
OBJECT_TYPES = ("portable", "fixed", "unknown")


@dataclass
class ObjectVocabulary:
    """Canonical object names, the text each one is queried with, and name -> portable / fixed / unknown.

    `prompts` holds canonical names (e.g. "metal_part", reported as object_name); `query` maps a name
    to the detector text (e.g. "small metal part"). `co_contact` lists name pairs one hand may touch at
    the same time even with shared fingertips (a cloth held while wiping a slab)."""
    prompts: list
    type_of: dict
    prompt_template: str = "{}"
    per_class_conf: dict = field(default_factory=dict)
    coco_map: dict = field(default_factory=dict)
    query: dict = field(default_factory=dict)
    co_contact: list = field(default_factory=list)
    profile: str = ""

    def object_type(self, name: str) -> str:
        return self.type_of.get(name, self.coco_map.get(name, "unknown"))

    def query_text(self, name: str) -> str:
        return self.prompt_template.format(self.query.get(name, name.replace("_", " ")))

    @staticmethod
    def resolve_profile(d: dict, profile: Optional[str], clip: Optional[str]) -> Optional[str]:
        """'auto' (or None) picks the first profile_rules entry whose `match` is in the clip name."""
        if "profiles" not in d:
            return None
        if profile and profile != "auto":
            if profile not in d["profiles"]:
                raise ValueError(f"unknown profile {profile!r}; have {sorted(d['profiles'])}")
            return profile
        for rule in d.get("profile_rules", []):
            if clip and rule["match"].lower() in clip.lower():
                return rule["profile"]
        return d.get("default_profile", next(iter(d["profiles"])))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROMPTS, profile: Optional[str] = None,
             clip: Optional[str] = None) -> "ObjectVocabulary":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        name = cls.resolve_profile(d, profile, clip)
        src = d["profiles"][name] if name else d
        type_of, prompts, query = {}, [], {}
        for t in OBJECT_TYPES:
            entries = src.get(t, [])
            items = entries.items() if isinstance(entries, dict) else ((e, e) for e in entries)
            for obj, text in items:
                if obj.startswith("_"):
                    continue
                if obj in type_of:
                    raise ValueError(f"object '{obj}' is listed under both {type_of[obj]} and {t}")
                type_of[obj], query[obj] = t, text
                prompts.append(obj)
        if not prompts:
            raise ValueError(f"{path}: no prompts" + (f" in profile {name}" if name else ""))
        coco = {k: v for k, v in d.get("yolo_coco_map", {}).items() if not k.startswith("_")}
        conf = {k: float(v) for k, v in {**d.get("per_class_conf", {}), **src.get("per_class_conf", {})}.items()
                if not k.startswith("_")}              # "_comment" entries are documentation, not thresholds
        pairs = [tuple(p) for p in src.get("co_contact_pairs", d.get("co_contact_pairs", []))]
        return cls(prompts, type_of, d.get("prompt_template", "{}"), conf, coco, query, pairs, name or "")


@dataclass
class PS2Config:
    # inputs / outputs
    video_path: str = ""
    hands_jsonl: str = ""                    # PS1 hands.jsonl (or hands_final.jsonl)
    output_dir: str = ""                     # default: <dir of hands_jsonl>/ps2
    prompts_path: str = str(DEFAULT_PROMPTS)
    profile: str = "auto"                    # workspace prompt profile; auto = chosen from the clip name
    # object detector
    backend: str = "owlv2"                   # owlv2 | sam3 (masks, gated weights) | yolo (COCO or fine-tuned)
    sam3_model: str = "facebook/sam3"        # gated on Hugging Face: needs HF_TOKEN in the environment
    sam3_mask_threshold: float = 0.5
    mask_grid: int = 32                      # object masks are stored as a grid x grid bitmap inside the box
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    yolo_model: str = "yolov8s.pt"           # COCO; see yolo_coco_map limitations
    object_conf: float = 0.15                # global threshold (per_class_conf overrides)
    object_nms_iou: float = 0.5
    object_imgsz: int = 960                  # owlv2: longest side fed to the model; yolo: imgsz
    max_objects_per_frame: int = 30
    min_object_area_frac: float = 0.0005     # drop tiny boxes (fraction of frame area)
    max_object_area_frac: float = 0.9        # drop boxes covering almost the whole frame
    det_batch: int = 4
    det_every: int = 1                       # >1 = detect every Nth frame, carry boxes in between
    half: bool = True                        # fp16 on CUDA
    # object tracker
    obj_iou_gate: float = 0.2
    obj_center_gate: float = 0.15            # centre distance / frame diagonal
    obj_max_missed: int = 10                 # frames an object ID survives without detection
    obj_type_mismatch: str = "forbid"        # forbid | allow  (different object type)
    # contact
    fingertip_threshold: int = 2
    pad_frac: float = 0.15
    min_contact_frames: int = 3
    min_contact_s: Optional[float] = None    # if set, also require this duration
    release_grace_frames: int = 3
    allow_multiple: bool = False
    multi_min_fingertips: int = 2            # extra contact needs this many fingertips not used by the first
    sticky_margin: int = 1                   # keep last frame's object if within this many fingertips of the best
    type_priority: Optional[tuple] = ("portable", "unknown", "fixed")   # None = fingertip count only
    use_masks: bool = True                   # if an object has a mask (sam3), a fingertip must be near the mask
    co_contact: bool = True                  # allow the vocabulary's co_contact_pairs (held cloth + touched slab)
    interpolated_can_start: bool = False
    min_contact_object_conf: float = 0.0
    hand_ids: Optional[list] = None          # allowlist; None = every PS1 hand track
    # alignment / run
    max_frames: Optional[int] = None
    t_tolerance_frames: float = 0.5          # |t - frame/fps| allowed, in frames
    write_video: bool = True
    overwrite_detections: bool = False
    progress_every: int = 300

    def detection_fingerprint(self, resolved_profile: str = "") -> str:
        """Settings that change object_detections.jsonl (contact settings are excluded)."""
        keys = ("backend", "profile", "owl_model", "yolo_model", "sam3_model", "sam3_mask_threshold", "mask_grid",
                "object_conf", "object_nms_iou", "object_imgsz",
                "max_objects_per_frame", "min_object_area_frac", "max_object_area_frac", "det_every",
                "half", "obj_iou_gate", "obj_center_gate", "obj_max_missed", "obj_type_mismatch",
                "max_frames")
        d = {k: getattr(self, k) for k in keys}
        d["prompts"] = Path(self.prompts_path).read_text(encoding="utf-8")
        d["resolved_profile"] = resolved_profile
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)
