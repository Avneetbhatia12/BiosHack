"""WiLoR 3D reconstruction stage, built on the official repository code.

Reused official components (not re-implemented):
    wilor.models.load_wilor                      model + config loading
    WiLoR.forward                                -> pred_cam, pred_keypoints_3d, pred_vertices
    wilor.datasets.utils.expand_to_aspect_ratio  crop size
    wilor.datasets.utils.generate_image_patch_cv2 crop / flip / warp
    wilor.datasets.vitdet_dataset.ViTDetDataset  (exact_official_crop=True path)
    wilor.utils.renderer.cam_crop_to_full        crop camera -> full-image camera
    wilor.utils.renderer.Renderer                optional mesh overlay / .obj export

Post-processing mirrors demo.py exactly:
    pred_cam[:,1] *= (2*right-1)                 undo the horizontal flip used for left hands
    joints[:,0]   *= (2*right-1)
    f_full = FOCAL_LENGTH / IMAGE_SIZE * max(W, H)
    cam_t  = cam_crop_to_full(pred_cam, box_center, box_size, img_size, f_full)
    uv     = f_full * (X + cam_t)[:2] / (X + cam_t)[2] + (W/2, H/2)
demo.py applies this projection to the 778 mesh vertices; we apply the same camera to the
21 joints, which is what PS1 needs.

Joint order: wilor/models/mano_wrapper.py remaps MANO joints with
    mano_to_openpose = [0, 13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]
so `pred_keypoints_3d[:, :21]` is OpenPose-hand order, which is the 21-point order the hackathon
format uses (0 wrist, 1-4 thumb CMC..TIP, 5-8 index MCP..TIP, 9-12 middle, 13-16 ring, 17-20 pinky).
Fingertips 4/8/12/16/20 are MANO surface vertices (smplx vertex_ids['mano']); the other 16 are MANO
skeleton joints (rotation centres), so they sit slightly inside the skin rather than on the surface.
No re-ordering is applied.
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
# Index i of our output = index MANO_TO_OPENPOSE[i] of raw MANO(+tips) joints (documentation only;
# the official wrapper already applies it).
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


@dataclass
class HandRequest:
    frame_slot: int          # index into the list of frames passed to reconstruct()
    bbox: np.ndarray         # (4,) xyxy in the frame's pixel coords
    is_right: bool


def _import_official(wilor_root: str):
    root = os.path.abspath(wilor_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # pyrender is imported by wilor.utils
    if "pyrender" in sys.modules:
        return
    try:
        import pyrender  # noqa: F401
    except Exception as e:  # noqa: BLE001
        # No usable OpenGL backend. wilor.utils imports pyrender at module level but only calls it
        # for rendering; install a placeholder so detection/reconstruction still import. Any
        # rendering call (mesh overlay, official demo renderer) then raises a clear error.
        import types
        print(f"[wilor] pyrender unavailable ({type(e).__name__}: {e}); mesh rendering disabled")
        stub = types.ModuleType("pyrender")

        def _missing(name):
            if name.startswith("__"):
                raise AttributeError(name)

            class _Unavailable:  # usable in annotations (e.g. List[pyrender.Node]); fails on use
                def __init__(self, *a, **k):
                    raise RuntimeError(f"pyrender.{name} unavailable: no OpenGL/EGL backend")
            _Unavailable.__name__ = name
            return _Unavailable

        stub.__getattr__ = _missing
        sys.modules["pyrender"] = stub


class WiLoRReconstructor:
    def __init__(self, cfg, device: str = "cuda", verify_checkpoint: bool = True):
        import torch
        from .setup_utils import pushd, trusted_torch_load

        _import_official(cfg.wilor_root)
        self.cfg, self.device = cfg, torch.device(device)
        ckpt = os.path.abspath(cfg.wilor_checkpoint)
        mcfg = os.path.abspath(cfg.wilor_model_cfg)
        with pushd(cfg.wilor_root), trusted_torch_load():
            from wilor.models import load_wilor
            try:
                model, model_cfg = load_wilor(checkpoint_path=ckpt, cfg_path=mcfg)
                self.load_path = "official load_wilor"
            except Exception as e:  # noqa: BLE001
                # WiLoR.__init__ builds training-time pyrender renderers, which need an OpenGL
                # (EGL) context. If the machine has none, repeat load_wilor's exact config steps
                # with the class's own init_renderer=False switch (renderers unused at inference).
                print(f"[wilor] load_wilor failed ({type(e).__name__}: {e}); retrying with "
                      f"init_renderer=False")
                model, model_cfg = self._load_without_renderer(ckpt, mcfg)
                self.load_path = "load_wilor steps + init_renderer=False"
            self.load_report = self._verify_checkpoint(model, ckpt) if verify_checkpoint else None

        from wilor.datasets.utils import expand_to_aspect_ratio, generate_image_patch_cv2
        from wilor.utils.renderer import cam_crop_to_full
        self._expand = expand_to_aspect_ratio
        self._gen_patch = generate_image_patch_cv2
        self._cam_crop_to_full = cam_crop_to_full

        self.fp16 = bool(cfg.fast or cfg.fp16)
        if self.fp16:
            torch.set_float32_matmul_precision("high")
            model = model.half()                       # FP16 weights; inputs are cast in reconstruct()
        if cfg.fast:
            # Official --fast recipe (demo.py) = FP16 + skipping 5 ViT blocks. skip_blocks is set on
            # the underlying ViT *before* any torch.compile wrapping so it is guaranteed to apply.
            # `fp16=True` alone keeps every block (no skipping).
            model.backbone.skip_blocks = True
        if cfg.compile_backbone:
            try:
                model.backbone = torch.compile(model.backbone)
            except Exception as e:  # noqa: BLE001 - e.g. P100 (sm60) unsupported by Triton
                print(f"[wilor] torch.compile unavailable, continuing eager: {e}")
        self.model = model.to(self.device).eval()
        self.model_cfg = model_cfg
        self.faces = np.asarray(model.mano.faces)
        self.image_size = int(model_cfg.MODEL.IMAGE_SIZE)
        self.focal = float(model_cfg.EXTRA.FOCAL_LENGTH)
        self.bbox_shape = model_cfg.MODEL.get("BBOX_SHAPE", None)
        self.mean = 255.0 * np.asarray(model_cfg.MODEL.IMAGE_MEAN, np.float32)
        self.std = 255.0 * np.asarray(model_cfg.MODEL.IMAGE_STD, np.float32)
        self.pool = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4))))
        self._renderer = None
        self.last_time_s = 0.0

    @staticmethod
    def _load_without_renderer(checkpoint_path: str, cfg_path: str):
        """Line-for-line the config handling of wilor.models.load_wilor, plus init_renderer=False."""
        from wilor.configs import get_config
        from wilor.models import WiLoR
        model_cfg = get_config(cfg_path, update_cachedir=True)
        if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in model_cfg.MODEL):
            model_cfg.defrost()
            assert model_cfg.MODEL.IMAGE_SIZE == 256
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost(); model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS"); model_cfg.freeze()
        if "DATA_DIR" in model_cfg.MANO:
            model_cfg.defrost()
            model_cfg.MANO.DATA_DIR = "./mano_data/"
            model_cfg.MANO.MODEL_PATH = "./mano_data/"
            model_cfg.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
            model_cfg.freeze()
        model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg,
                                           init_renderer=False)
        return model, model_cfg

    # ------------------------------------------------------------------ verification
    @staticmethod
    def _verify_checkpoint(model, ckpt_path: str) -> dict:
        """load_wilor uses strict=False; confirm every learned weight really came from the file."""
        import torch
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        msd = model.state_dict()
        missing = [k for k in msd if k not in sd]
        unexpected = [k for k in sd if k not in msd]
        crit = [k for k in msd if k.startswith(("backbone.", "refine_net."))]
        crit_missing = [k for k in crit if k not in sd]
        mismatch = [k for k in crit if k in sd and tuple(sd[k].shape) != tuple(msd[k].shape)]
        sample = [k for k in crit if k in sd][:: max(1, len(crit) // 20)]
        equal = all(torch.equal(sd[k].to(msd[k].dtype), msd[k].cpu()) for k in sample)
        rep = {"model_tensors": len(msd), "ckpt_tensors": len(sd), "missing_in_ckpt": len(missing),
               "unexpected_in_ckpt": len(unexpected), "critical_tensors": len(crit),
               "critical_missing": crit_missing[:10], "shape_mismatch": mismatch[:10],
               "sampled_weights_equal": bool(equal), "missing_examples": missing[:10]}
        if crit_missing or mismatch or not equal:
            raise RuntimeError(f"WiLoR checkpoint did not load correctly: {rep}")
        del sd
        return rep

    # ------------------------------------------------------------------ preprocessing
    def _patch(self, img: np.ndarray, bbox: np.ndarray, is_right: bool):
        """Same maths as ViTDetDataset.__getitem__, but the anti-alias Gaussian is applied to a
        local crop instead of the whole frame. The crop includes the full warp footprint plus
        the Gaussian support (truncate=4 sigma), so the patch is identical up to float rounding
        (verified in the notebook against the official dataset class)."""
        from skimage.filters import gaussian
        H, W = img.shape[:2]
        b = bbox.astype(np.float32)
        center = (b[2:4] + b[0:2]) / 2.0
        scale = self.cfg.rescale_factor * (b[2:4] - b[0:2]) / 200.0
        bbox_size = float(self._expand(scale * 200, target_aspect_ratio=self.bbox_shape).max())
        ps = self.image_size
        down = (bbox_size / ps) / 2.0
        sigma = (down - 1) / 2 if down > 1.1 else 0.0
        if self.cfg.exact_official_crop or sigma <= 0:
            # No blur needed -> warping the full frame is already cheap; keeping the original
            # origin also keeps OpenCV's fixed-point uint8 interpolation bit-identical.
            x0, y0, crop = 0, 0, img
        else:
            margin = bbox_size / 2.0 + 4 + (int(4.0 * sigma + 0.5) + 2 if sigma > 0 else 0)
            x0 = max(0, int(math.floor(center[0] - margin)))
            y0 = max(0, int(math.floor(center[1] - margin)))
            x1 = min(W, int(math.ceil(center[0] + margin)) + 1)
            y1 = min(H, int(math.ceil(center[1] + margin)) + 1)
            crop = img[y0:y1, x0:x1]
        if sigma > 0:
            if self.cfg.exact_official_crop:
                crop = gaussian(crop, sigma=sigma, channel_axis=2, preserve_range=True)
            else:
                # Same kernel as skimage/scipy (sampled Gaussian, radius int(4*sigma+0.5),
                # 'nearest' == BORDER_REPLICATE); ~10x faster. Max deviation ~6e-5 grey levels.
                # float64 like skimage: OpenCV's float32 warp uses quantised weights (~0.5 grey).
                k = cv2.getGaussianKernel(2 * int(4.0 * sigma + 0.5) + 1, sigma, cv2.CV_64F)
                crop = cv2.sepFilter2D(np.ascontiguousarray(crop).astype(np.float64), -1, k, k,
                                       borderType=cv2.BORDER_REPLICATE)
        patch, _ = self._gen_patch(crop, center[0] - x0, center[1] - y0, bbox_size, bbox_size,
                                   ps, ps, not is_right, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
        patch = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)   # BGR->RGB, CHW
        patch = (patch - self.mean[:, None, None]) / self.std[:, None, None]
        return patch, center, bbox_size, np.array([W, H], np.float32)

    def official_patch(self, img, bbox, is_right):
        """Reference path through the untouched official ViTDetDataset (for verification)."""
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        ds = ViTDetDataset(self.model_cfg, img, bbox[None].astype(np.float32),
                           np.array([1.0 if is_right else 0.0]), rescale_factor=self.cfg.rescale_factor)
        return ds[0]

    # ------------------------------------------------------------------ inference
    def reconstruct(self, frames: List[np.ndarray], requests: List[HandRequest],
                    return_vertices: bool = False) -> List[Dict]:
        import torch
        if not requests:
            self.last_time_s = 0.0
            return []
        t0 = time.perf_counter()
        prep = list(self.pool.map(lambda r: self._patch(frames[r.frame_slot], r.bbox, r.is_right),
                                  requests))
        results: List[Dict] = []
        bs = self.cfg.wilor_batch_size
        dtype = torch.float16 if self.fp16 else torch.float32
        for s in range(0, len(requests), bs):
            chunk, reqs = prep[s:s + bs], requests[s:s + bs]
            img = torch.from_numpy(np.stack([c[0] for c in chunk])).to(self.device, dtype)
            right = torch.tensor([1.0 if r.is_right else 0.0 for r in reqs], device=self.device)
            center = torch.from_numpy(np.stack([c[1] for c in chunk])).to(self.device).float()
            bsize = torch.tensor([c[2] for c in chunk], device=self.device).float()
            isize = torch.from_numpy(np.stack([c[3] for c in chunk])).to(self.device).float()
            batch = {"img": img, "right": right, "box_center": center, "box_size": bsize,
                     "img_size": isize}
            with torch.inference_mode():
                out = self.model(batch)
            mult = 2 * right - 1
            pred_cam = out["pred_cam"].float().clone()
            pred_cam[:, 1] = mult * pred_cam[:, 1]
            focal = self.focal / self.image_size * isize.max(dim=1).values          # (B,)
            cam_t = self._cam_crop_to_full(pred_cam, center, bsize, isize, focal)   # (B,3)
            joints = out["pred_keypoints_3d"].float()
            assert joints.shape[1] >= 21, f"unexpected joint count {joints.shape}"
            joints = joints[:, :21].clone()
            joints[..., 0] = mult[:, None] * joints[..., 0]
            pc = joints + cam_t[:, None, :]
            uv = focal[:, None, None] * pc[..., :2] / pc[..., 2:3] + isize[:, None, :] / 2.0
            verts = None
            if return_vertices:
                verts = out["pred_vertices"].float().clone()
                verts[..., 0] = mult[:, None] * verts[..., 0]
                verts = verts.cpu().numpy()
            joints_np, cam_np, uv_np = joints.cpu().numpy(), cam_t.cpu().numpy(), uv.cpu().numpy()
            focal_np = focal.cpu().numpy()
            for i in range(len(reqs)):
                results.append({"joints": joints_np[i], "cam_t": cam_np[i], "kp2d": uv_np[i],
                                "focal": float(focal_np[i]),
                                "vertices": None if verts is None else verts[i]})
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_time_s = time.perf_counter() - t0
        return results

    # ------------------------------------------------------------------ meshes
    @property
    def renderer(self):
        if self._renderer is None:
            from wilor.utils.renderer import Renderer
            self._renderer = Renderer(self.model_cfg, faces=self.faces)
        return self._renderer

    def export_obj(self, path: str, vertices: np.ndarray, cam_t: np.ndarray, is_right: bool):
        tm = self.renderer.vertices_to_trimesh(vertices, cam_t.copy(), (0.25, 0.27, 0.66),
                                               is_right=int(is_right))
        tm.export(path)

    def render_mesh_overlay(self, img_bgr: np.ndarray, verts: List[np.ndarray],
                            cam_ts: List[np.ndarray], rights: List[bool], focal: float) -> np.ndarray:
        """Official pyrender overlay (demo.py). Requires a working EGL/OSMesa context."""
        if not verts:
            return img_bgr
        H, W = img_bgr.shape[:2]
        rgba = self.renderer.render_rgba_multiple(
            verts, cam_t=cam_ts, render_res=[W, H], is_right=[int(r) for r in rights],
            mesh_base_color=(0.25098039, 0.274117647, 0.65882353), scene_bg_color=(1, 1, 1),
            focal_length=focal)
        a = rgba[:, :, 3:]
        out = img_bgr[:, :, ::-1].astype(np.float32) / 255.0 * (1 - a) + rgba[:, :, :3] * a
        return np.ascontiguousarray((out[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8))
