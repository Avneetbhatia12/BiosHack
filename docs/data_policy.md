# Data policy

The hackathon footage belongs to a real company and shows real workplaces. Every participant signed a data-use
agreement; breaking it means disqualification.

## Never in this repository (or anywhere public)

- raw videos, downscaled copies, extracted frames, crops or screenshots
- annotated / demo / training videos, review sheets with frames, notebook outputs that show frames
- labels, audit CSVs, review players and per-frame result files derived from the footage
- Kaggle datasets, MANO files, model weights
- API keys or tokens (`GEMINI_API_KEY`, `HF_TOKEN`, `kaggle.json`)

`.gitignore` enforces this: `data/*`, `outputs/*`, `eval/labels/`, all video and image formats, `*.jsonl`, audit and
review files, weights (`*.pt`, `*.ckpt`, `*.pkl`, …), `/WiLoR/`, `.env*` (except `.env.example`) and `kaggle.json`.
`tests/test_repo_hygiene.py` fails if a forbidden file, a secret-like string, a private Kaggle dataset path or a
notebook output would be committed. Run `pytest` before every push.

## Working rules

- Keep footage only in private locations: a **private** Kaggle dataset, or `data/` on your own machine.
- Demo videos go **only** to the organisers' Drive folder (2 minutes in total).
- Clear all notebook outputs before saving or sharing a notebook outside Kaggle.
- Secrets come from Kaggle Secrets, Colab userdata or a local `.env`; code reads them with
  `handtrace.environment.load_secret` and never prints them.
- PS3 sends clips to the Gemini API, which the hackathon rules allow. The pipeline deletes each upload after the clip,
  and `ps3.cleanup_remote` deletes every file stored under the key.
- If a clip shows a face, a screen or company documents, keep it out of any public material.

## When the event ends

Run the notebook's cleanup cell (deletes working copies and Gemini storage), then delete the private Kaggle datasets
(clips, MANO, labels), the notebook's saved versions and outputs, and any local copies in `data/` and `outputs/`.
