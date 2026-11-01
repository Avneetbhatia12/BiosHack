"""Central configuration for the PS1 WiLoR hand-tracking pipeline.

Every tunable lives here so the notebook / CLI never hard-codes paths or thresholds.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    # ---------------- paths ----------------
    video_path: str = ""                     # input clip (never hard-coded; set by notebook/CLI)
    output_dir: str = "outputs"              # per-clip sub-folder is created inside
    wilor_root: str = "WiLoR"                # official repository checkout
    detector_weights: str = ""               # default: <wilor_root>/pretrained_models/detector.pt
    wilor_checkpoint: str = ""               # default: <wilor_root>/pretrained_models/wilor_final.ckpt
    wilor_model_cfg: str = ""                # default: <wilor_root>/pretrained_models/model_config.yaml

    # ---------------- frames ----------------
    frame_step: int = 1                      # 1 = every frame (default). >1 = skip, gaps interpolated
    max_frames: Optional[int] = None         # process only the first N frames (quick tests)
    start_frame: int = 0
    proc_scale: float = 1.0                  # downscale frames before inference; outputs mapped back
    chunk_size: int = 8                      # frames per detector/WiLoR batch

    # ---------------- detector (official WiLoR YOLO) ----------------
    det_conf: float = 0.20                   # low threshold feeds the 2nd (low-score) tracking stage
    det_high_conf: float = 0.40              # detections >= this may start new tracks
    det_iou: float = 0.50                    # YOLO per-class NMS IoU
    det_imgsz: int = 640                     # YOLO inference size (960/1280 helps small/distant hands)
    agnostic_nms_iou: float = 0.70           # merge left/right duplicate boxes on the same hand; <=0 disables
    max_hands_per_frame: int = 4             # keep top-K by score after NMS (worker + occasional co-worker)
    det_half: bool = False                   # FP16 detector (benchmark before enabling)

    # ---------------- WiLoR reconstruction ----------------
    fast: bool = False                       # official --fast: FP16 + ViT block skipping (changes keypoints)
    fp16: bool = False                       # FP16 only, NO block skipping (benchmark_modes decides if it is safe)
    compile_backbone: bool = False           # torch.compile (needs Triton; T4 OK, P100 not supported)
    wilor_batch_size: int = 32
    rescale_factor: float = 2.0              # demo.py default bbox padding
    exact_official_crop: bool = False        # True = use official ViTDetDataset (full-frame blur, slower)
    kp3d_frame: str = "wrist_relative"       # "wrist_relative" | "camera"
    save_vertices: bool = False              # write mesh vertices per hand to vertices.npz (big)
    export_obj_every: int = 0                # export .obj meshes every N frames (0 = off)

    # ---------------- handedness ----------------
    handedness_swap: bool = False            # global swap AFTER verifying convention on real frames
    side_mode: str = "track_vote"            # "detector" | "track_vote" (temporal majority per track)
    side_vote_decay: float = 0.90            # EMA decay of the per-track side vote
    side_flip_margin: float = 0.60           # |vote| needed before a track overrides a detector label

    # ---------------- tracking ----------------
    match_iou_weight: float = 0.6
    match_center_weight: float = 0.4
    match_side_penalty: float = 0.25         # soft cost added when sides disagree
    match_max_cost: float = 0.85             # gate
    center_gate: float = 0.35                # max centre distance / frame diagonal for a match
    max_missed: int = 20                     # processed frames a track survives without detection
    reid_center_gate: float = 0.20           # recovery radius (fraction of diag) for lost tracks

    # ---------------- gap filling ----------------
    interpolate_gaps: bool = True
    max_interp_gap: int = 8                  # frames; linear interpolation of short dropouts

    # ---------------- smoothing ----------------
    smoothing: bool = True
    smooth_3d: bool = True
    one_euro_min_cutoff_2d: float = 1.5      # Hz; lower = smoother when still
    one_euro_beta_2d: float = 0.02           # higher = less lag when moving fast (px/s units)
    one_euro_min_cutoff_3d: float = 1.5
    one_euro_beta_3d: float = 5.0            # metres/s units -> larger beta
    one_euro_dcutoff: float = 1.0

    # ---------------- outputs ----------------
    write_video: bool = True
    render_mesh: bool = False                # pyrender overlay (slow, needs EGL)
    video_crf: int = 23
    progress_every: int = 150                # frames
    flush_every: int = 300                   # frames

    extra: dict = field(default_factory=dict)

    def resolved(self) -> "PipelineConfig":
        import os
        c = PipelineConfig(**asdict(self))
        pm = os.path.join(c.wilor_root, "pretrained_models")
        c.detector_weights = c.detector_weights or os.path.join(pm, "detector.pt")
        c.wilor_checkpoint = c.wilor_checkpoint or os.path.join(pm, "wilor_final.ckpt")
        c.wilor_model_cfg = c.wilor_model_cfg or os.path.join(pm, "model_config.yaml")
        assert c.kp3d_frame in ("wrist_relative", "camera")
        assert c.side_mode in ("detector", "track_vote")
        assert c.frame_step >= 1
        return c

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls(**json.load(f))
