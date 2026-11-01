"""Guards for a public repo: nothing private or heavy may be committed (data-use rules)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"\.(mp4|avi|mov|mkv|webm|jpg|jpeg|png|bmp|gif|tif|tiff|webp|pt|pth|ckpt|task|onnx|tflite|"
                       r"safetensors|bin|h5|pkl|npz|obj|jsonl)$", re.I)
SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,}|\"key\"\s*:\s*\"[0-9a-f]{32}\")")


def _publishable_files():
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [ROOT / p for p in out if p and (ROOT / p).exists()]


def test_no_data_weights_or_outputs_would_be_committed():
    bad = [str(p.relative_to(ROOT)) for p in _publishable_files() if FORBIDDEN.search(p.name)]
    assert not bad, f"private/heavy files would be committed: {bad}"


def test_no_secrets_or_private_kaggle_paths_in_text_files():
    hits = []
    for p in _publishable_files():
        if p.suffix not in {".py", ".md", ".txt", ".json", ".ipynb", ".yaml", ".yml", ".cfg", ".toml"} or p == Path(__file__):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if SECRET.search(text) or "/kaggle/input/datasets/" in text:
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"secret-like strings or private dataset paths in: {hits}"


def test_notebooks_have_no_outputs():
    for nb_path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                assert not c.get("outputs") and c.get("execution_count") is None, f"{nb_path.name} cell {i} has output"
