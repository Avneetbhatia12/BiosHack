"""Environment / asset helpers for running the official WiLoR code on Kaggle.

* environment_report()      - Python / torch / CUDA / GPU summary
* trusted_torch_load()      - torch>=2.6 defaults torch.load(weights_only=True), which rejects the
                              official checkpoints (they pickle a yacs CfgNode and ultralytics
                              model classes). Only used for the two official files.
* prepare_mano_right()      - copies the user's licensed MANO_RIGHT.pkl into WiLoR/mano_data and
                              strips chumpy objects so `chumpy` (broken on numpy>=1.24 / py>=3.11)
                              is not needed. Array values are unchanged.
* verify_assets()           - existence + size sanity checks.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import pickle
import platform
import shutil
import sys
from typing import Optional

WEIGHT_URLS = {
    "detector.pt": "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
    "wilor_final.ckpt": "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
}
# Rough lower bounds; a truncated / HTML error download is caught by these.
MIN_SIZES_MB = {"detector.pt": 5, "wilor_final.ckpt": 500, "MANO_RIGHT.pkl": 1}


def environment_report() -> dict:
    rep = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        rep.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                   cuda_available=torch.cuda.is_available(),
                   cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            rep.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1),
                       compute_capability=f"{p.major}.{p.minor}")
    except ImportError:
        rep["torch"] = None
    for mod in ("torchvision", "numpy", "cv2", "ultralytics", "pytorch_lightning", "smplx", "timm",
                "skimage", "pyrender", "trimesh", "scipy"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            rep[mod] = f"MISSING ({type(e).__name__})"
    return rep


@contextlib.contextmanager
def trusted_torch_load():
    """Force weights_only=False inside this block (official, trusted checkpoints only)."""
    import torch
    orig = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = orig


@contextlib.contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def file_md5(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# MANO: remove chumpy dependency
# --------------------------------------------------------------------------------------
class _ChumpyStub:
    """Stand-in for chumpy classes during unpickling; captures the pickled state."""

    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _NoChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return type(name, (_ChumpyStub,), {})
        return super().find_class(module, name)


def _to_numpy(v):
    import numpy as np
    if isinstance(v, _ChumpyStub):
        x = v.__dict__.get("x", None)
        if isinstance(x, np.ndarray):
            return np.array(x)
        arrays = [a for a in v.__dict__.values() if isinstance(a, np.ndarray)]
        if len(arrays) == 1:
            return np.array(arrays[0])
        raise ValueError(
            f"Cannot convert chumpy object {type(v).__name__} without chumpy "
            f"(keys={list(v.__dict__)}). Install chumpy and retry with use_real_chumpy=True.")
    return v


def load_mano_pkl_without_chumpy(path: str) -> dict:
    with open(path, "rb") as f:
        data = _NoChumpyUnpickler(f, encoding="latin1").load()
    return {k: _to_numpy(v) for k, v in data.items()}


def prepare_mano_right(src: Optional[str], wilor_root: str, search_root: str = "/kaggle/input") -> str:
    """Place a chumpy-free MANO_RIGHT.pkl in <wilor_root>/mano_data. Returns its path.

    `src` is the user's licensed file (download from https://mano.is.tue.mpg.de after accepting
    the MANO licence). If None, the first MANO_RIGHT.pkl found under `search_root` is used.
    """
    import numpy as np
    dst = os.path.join(wilor_root, "mano_data", "MANO_RIGHT.pkl")
    if src is None:
        hits = sorted(glob.glob(os.path.join(search_root, "**", "MANO_RIGHT.pkl"), recursive=True))
        if not hits:
            if os.path.exists(dst):
                src = dst
            else:
                raise FileNotFoundError(
                    "MANO_RIGHT.pkl not found. Register at https://mano.is.tue.mpg.de, download "
                    "mano_v1_2.zip, and attach MANO_RIGHT.pkl as a PRIVATE Kaggle dataset.")
        else:
            src = hits[0]
    data = load_mano_pkl_without_chumpy(src)
    for key in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f", "kintree_table"):
        if key not in data:
            raise KeyError(f"{src} does not look like a MANO model (missing '{key}')")
    assert np.asarray(data["v_template"]).shape == (778, 3), "expected 778 MANO vertices"
    # Without chumpy, a chumpy array can come back flat (seen on Kaggle: shapedirs (23340,) instead of (778, 3, 10)),
    # which makes WiLoR fail later. Restore the MANO shapes here.
    for key, tail in (("shapedirs", None), ("posedirs", 135)):
        a = np.asarray(data[key])
        if a.ndim != 3 and a.size % (778 * 3) == 0:
            data[key] = a.reshape(778, 3, tail or -1)
    assert np.asarray(data["shapedirs"]).shape[:2] == (778, 3), f"bad shapedirs {np.asarray(data['shapedirs']).shape}"
    assert np.asarray(data["posedirs"]).shape == (778, 3, 135), f"bad posedirs {np.asarray(data['posedirs']).shape}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=2)
    os.replace(tmp, dst)
    return dst


def verify_assets(wilor_root: str) -> dict:
    out = {}
    pm = os.path.join(wilor_root, "pretrained_models")
    for name in ("detector.pt", "wilor_final.ckpt"):
        p = os.path.join(pm, name)
        ok = os.path.exists(p) and os.path.getsize(p) / 1e6 >= MIN_SIZES_MB[name]
        out[name] = {"path": p, "ok": ok,
                     "size_mb": round(os.path.getsize(p) / 1e6, 1) if os.path.exists(p) else None}
    p = os.path.join(wilor_root, "mano_data", "MANO_RIGHT.pkl")
    out["MANO_RIGHT.pkl"] = {"path": p, "ok": os.path.exists(p)}
    p = os.path.join(wilor_root, "mano_data", "mano_mean_params.npz")
    out["mano_mean_params.npz"] = {"path": p, "ok": os.path.exists(p)}
    return out


def copy_weights_from(dir_with_weights: str, wilor_root: str) -> None:
    """Alternative to downloading: copy weights from an attached Kaggle dataset."""
    for name in WEIGHT_URLS:
        hits = glob.glob(os.path.join(dir_with_weights, "**", name), recursive=True)
        if hits:
            shutil.copy2(hits[0], os.path.join(wilor_root, "pretrained_models", name))


# --------------------------------------------------------------------------------------
# One-call asset preparation (formerly a notebook cell)
# --------------------------------------------------------------------------------------
WILOR_REPO = "https://github.com/rolpotamias/WiLoR.git"
WILOR_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"      # pinned: upstream changes cannot alter behaviour
CHUMPY_REPO = "https://github.com/mattloper/chumpy"
CHUMPY_COMMIT = "580566eafc9ac68b2614b64d6f7aaa84eebb70da"


def _run(cmd: list, cwd: Optional[str] = None) -> None:
    import subprocess
    subprocess.run(cmd, check=True, cwd=cwd)


def clone_wilor(wilor_root: str, commit: str = WILOR_COMMIT) -> None:
    """Clone the official WiLoR repository (with submodules) and check out the pinned commit."""
    if not os.path.isdir(os.path.join(wilor_root, ".git")):
        _run(["git", "clone", "-q", "--recursive", WILOR_REPO, wilor_root])
    _run(["git", "checkout", "-q", commit], cwd=wilor_root)
    _run(["git", "submodule", "update", "--init", "--recursive", "-q"], cwd=wilor_root)


def download_weights(wilor_root: str, weights_dir: Optional[str] = None) -> None:
    """detector.pt + wilor_final.ckpt into <wilor_root>/pretrained_models (from `weights_dir` when offline)."""
    import urllib.request
    pm = os.path.join(wilor_root, "pretrained_models")
    os.makedirs(pm, exist_ok=True)
    if weights_dir:
        copy_weights_from(weights_dir, wilor_root)
    for name, url in WEIGHT_URLS.items():
        dst = os.path.join(pm, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)


def ensure_mano(wilor_root: str, search_roots=("/kaggle/input",), work_dir: str = "/tmp") -> str:
    """MANO_RIGHT.pkl -> <wilor_root>/mano_data as plain numpy arrays.

    The real chumpy (pinned commit, cloned to `work_dir`) unpickles every object exactly; if it cannot be used,
    the chumpy-free reader is used instead (it reshapes flat arrays back to MANO shapes)."""
    import numpy as np
    dst = os.path.join(wilor_root, "mano_data", "MANO_RIGHT.pkl")
    hits = sorted(h for r in search_roots for h in glob.glob(os.path.join(r, "**", "MANO_RIGHT.pkl"), recursive=True)
                  if os.path.abspath(h) != os.path.abspath(dst))
    if not hits:
        if os.path.exists(dst):
            return dst
        raise FileNotFoundError("MANO_RIGHT.pkl not found: attach it as a PRIVATE dataset (mano.is.tue.mpg.de, "
                                "accept the licence) or place it under one of the search roots")
    try:
        d = os.path.join(work_dir, "chumpy_src")
        if not os.path.isdir(d):
            _run(["git", "clone", "-q", CHUMPY_REPO, d])
            _run(["git", "checkout", "-q", CHUMPY_COMMIT], cwd=d)
        if d not in sys.path:
            sys.path.insert(0, d)
        import chumpy as ch
        with open(hits[0], "rb") as f:
            data = pickle.load(f, encoding="latin1")
        data = {k: (np.array(v.r) if isinstance(v, ch.Ch) else v) for k, v in data.items()}
        assert data["shapedirs"].shape == (778, 3, 10) and data["posedirs"].shape == (778, 3, 135)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump(data, f, protocol=2)
    except Exception as e:  # noqa: BLE001
        print("chumpy route failed:", repr(e), "-> chumpy-free fallback")
        prepare_mano_right(hits[0], wilor_root)
    return dst


def prepare_wilor(wilor_root: str, mano_search_roots=("/kaggle/input",), weights_dir: Optional[str] = None,
                  commit: str = WILOR_COMMIT, work_dir: str = "/tmp") -> dict:
    """Clone WiLoR (pinned), fetch weights, prepare MANO, verify. Returns verify_assets(); raises if incomplete."""
    clone_wilor(wilor_root, commit)
    download_weights(wilor_root, weights_dir)
    ensure_mano(wilor_root, mano_search_roots, work_dir)
    status = verify_assets(wilor_root)
    missing = [k for k, v in status.items() if not v["ok"]]
    if missing:
        raise RuntimeError(f"missing or truncated WiLoR assets: {missing}")
    return status
