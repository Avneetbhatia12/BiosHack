#!/usr/bin/env bash
# One-time setup on a Linux machine with a CUDA GPU (Kaggle users: the notebook does all of this for you).
#
#   bash scripts/setup.sh [WILOR_DIR] [MANO_SEARCH_DIR]
#
# 1. Install PyTorch for YOUR CUDA version first (https://pytorch.org/get-started/locally/); it is not pinned here.
# 2. MANO_RIGHT.pkl must be downloaded by you from https://mano.is.tue.mpg.de after accepting its licence;
#    put it anywhere under MANO_SEARCH_DIR (default: data/). It is never committed (see .gitignore).
set -euo pipefail

WILOR_DIR="${1:-WiLoR}"
MANO_DIR="${2:-data}"

python -c "import torch" 2>/dev/null || { echo "Install PyTorch for your CUDA version first."; exit 1; }
python -m pip install -r requirements.txt
python -m pip install --no-deps ultralytics==8.1.34          # WiLoR's detector runtime, without replacing torch
python -m pip install -e .

python - "$WILOR_DIR" "$MANO_DIR" <<'PY'
import json, sys
from handtrace.wilor.setup_utils import prepare_wilor
status = prepare_wilor(sys.argv[1], mano_search_roots=[sys.argv[2]])
print(json.dumps(status, indent=1))
PY

echo
echo "Ready. Optional: copy .env.example to .env and set GEMINI_API_KEY (PS3) / HF_TOKEN (SAM 3)."
echo "Run:  python -m handtrace.cli all --clips data/clips --out outputs --wilor $WILOR_DIR"
