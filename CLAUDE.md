# CLAUDE.md — HandTrace

## Scope
- PS1 (hand tracking), PS2 (hand-object contact), PS3 (Gemini task-step timeline), PS5 (insight tools). Not PS4.
- Track the **camera wearer's** hands only; background workers' hands are false positives.
- Hand-tracking accuracy is judged in every problem statement (30/100, tie-breaker): prefer accuracy and robustness.

## Architecture
The `handtrace` package is the source of truth. `notebooks/HandTrace_Final_Pipeline.ipynb` only clones the repo,
sets runtime options and calls the package — never generate code from notebook cells again.

| Area | Modules |
|---|---|
| entry points | `cli.py` (`python -m handtrace.cli {ps1|ps2|ps3|ps5|all}`), `runner.py` (resumable multi-GPU workers) |
| PS1 | `wilor/` (detector, `hand_tracker.py`, `wilor_inference.py`, smoothing, pipeline, refine, `setup_utils.py`), `postprocess.py` |
| PS2 | `ps2/` (object_detector, object_tracker, contact_detector, state_machine, masks, pipeline, visualization) |
| PS3 | `ps3/` (windows, prompt, gemini_client, schema, validate, merge, refine, compare, pipeline) |
| PS5 | `ps5/` (index, search, quality, safety, dashboard, training) |
| shared | `results.py` (all output paths/loaders), `render.py`, `review*.py`, `environment.py`, `io.py` (labelling tools only) |

Output record (`hands_final.jsonl`): `frame`, `t`, `hands[id, side, bbox, score, kp2d, kp3d, interpolated, side_corrected]`.
Object IDs are strings `O<n>`, hand IDs integers. Add fields freely; rename/remove only with a documented change.

## Coding rules
- Python 3.10+, type hints on public functions, short docstrings, `pathlib` for new code.
- Heavy libraries (torch, ultralytics, transformers, WiLoR, google-genai, pydantic) are imported lazily inside functions.
- Thresholds live in config dataclasses / `configs/*.json` / CLI flags, not deep in code.
- Coordinates are pixels, `bbox = [x1, y1, x2, y2]`, keypoints in MediaPipe/OpenPose order (tips 4/8/12/16/20).
- New dependency → `requirements.txt`, `pyproject.toml` and the README stack table. Never replace Kaggle's torch.
- Never hard-code API keys; use `handtrace.environment.load_secret`.

## Data privacy (breaking these disqualifies the team)
- Never commit, upload or paste clips, frames, screenshots, crops, labels, audits, demo videos or weights.
- Data only in `data/`, outputs only in `outputs/`, labels only in `eval/labels/` — all gitignored.
- Tests use synthetic data only. Notebooks are committed with outputs cleared.
- `pytest` (incl. `tests/test_repo_hygiene.py`) must pass before any commit.

## Git
- `main` is the base branch; feature work on branches. Never commit or push without the team's explicit request.
- Before a commit, run `git status` and confirm no video, image, weight, jsonl, `.env` or `kaggle.json` is staged.
- Never force-push or change the git config.

## Testing
- `tests/test_ps1.py`, `test_ps2.py`, `test_ps3.py`, `test_ps5.py`, `test_pipeline.py`, `test_repo_hygiene.py`.
- Never describe synthetic-test results as real-footage results. Accuracy claims come only from the manual audit or
  `eval/` on labelled frames, with the count.
