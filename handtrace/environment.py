"""Runtime environment helpers: shell commands, secrets, and dependency installation that keeps the
preinstalled PyTorch / CUDA stack (Kaggle, Colab) untouched.

    from handtrace import environment as env
    env.install_dependencies(ps1=True, ps2=True, ps3=True)     # every pip install runs under a constraints file
    env.load_secret("GEMINI_API_KEY")                          # Kaggle secret -> Colab userdata -> environment
"""
from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

#: packages whose preinstalled versions must never change (CUDA builds)
PROTECTED = ("torch", "torchvision", "torchaudio", "numpy", "opencv-python", "opencv-python-headless",
             "opencv-contrib-python", "scipy", "pillow")
CONSTRAINTS = "/tmp/handtrace_constraints.txt"


def sh(cmd: str, check: bool = True, quiet: bool = False, cwd: Optional[str] = None) -> str:
    """Run a shell command, stream its output, raise with the output tail on failure."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not quiet:
            print(line, end="")
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f"command failed ({rc}): {cmd}\n" + "".join(lines[-30:]))
    return "".join(lines)


def pkg_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def get_secret(name: str) -> Optional[str]:
    """Environment -> Kaggle secret -> Colab userdata. Returns None when absent. Never prints the value."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:  # noqa: BLE001
        return None


def load_secret(name: str) -> bool:
    """Copy a secret into os.environ[name] if found; returns whether it is set (the value is never shown)."""
    value = get_secret(name)
    if value:
        os.environ[name] = value
    return bool(os.environ.get(name))


def write_constraints(path: str = CONSTRAINTS) -> str:
    """Pin the currently installed protected packages so pip cannot replace them."""
    pins = [f"{p}=={pkg_version(p)}" for p in PROTECTED if pkg_version(p)]
    with open(path, "w") as f:
        f.write("\n".join(pins) + "\n")
    return path


def install_dependencies(ps1: bool = True, ps2: bool = True, ps3: bool = True, review: bool = True,
                         quiet: bool = True) -> dict:
    """Install what each stage needs without touching torch / CUDA / numpy / OpenCV. Returns versions."""
    c = write_constraints()
    pip = f"{sys.executable} -m pip install -q -c {c}"
    before = {m: pkg_version(m) for m in ("torch", "torchvision", "numpy", "scipy", "pillow")}
    if ps1 or ps2:
        sh("apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y libegl1 libgl1 >/dev/null 2>&1",
           check=False, quiet=True)                                 # optional: mesh renderer only
        sh(f"{pip} smplx==0.1.28 yacs timm einops pytorch-lightning scikit-image trimesh pyrender", quiet=quiet)
        sh(f"{sys.executable} -m pip install -q --no-deps ultralytics==8.1.34", quiet=quiet)   # WiLoR's detector runtime
        sh(f"{pip} thop py-cpuinfo", quiet=quiet)
        tv = tuple(int(x) for x in (pkg_version("transformers") or "0.0").split(".")[:2])
        if tv < (4, 40):                                            # OWLv2 needs a recent transformers
            sh(f"{pip} 'transformers>=4.40'", quiet=quiet)
        if not pkg_version("scipy"):
            sh(f"{pip} scipy", quiet=quiet)
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import ultralytics  # noqa: F401
        except Exception as e:  # noqa: BLE001  fallback if 8.1.34 cannot import on this image
            print("ultralytics 8.1.34 failed to import:", repr(e), "-> trying 8.3.x")
            sh(f"{sys.executable} -m pip install -q --no-deps 'ultralytics>=8.3,<8.4' ultralytics-thop", quiet=quiet)
    if ps3 and not pkg_version("google-genai"):
        sh(f"{pip} google-genai pydantic", quiet=quiet)
    if review and not pkg_version("ipywidgets"):
        sh(f"{pip} ipywidgets", check=False, quiet=quiet)
    after = {m: pkg_version(m) for m in before}
    return {"protected_unchanged": before == after, "versions": after, "transformers": pkg_version("transformers"),
            "google-genai": pkg_version("google-genai")}
