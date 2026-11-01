"""Run the full pipeline (thin wrapper around `python -m handtrace.cli`).

    python scripts/run_pipeline.py all --clips data/clips --out outputs --wilor WiLoR
    python scripts/run_pipeline.py ps3 --config configs/example_config.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handtrace.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
