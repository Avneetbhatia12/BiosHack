# Problem statements and where they live

The hackathon (Egocentric Vision Hackathon, BIOS × MytronLabs) offered five problem statements. HandTrace covers
**PS1, PS2, PS3 and PS5**; PS4 (3D hand motion) is not a deliverable, although PS1 stores WiLoR's 3D joints (`kp3d`).

## PS1 — Hand tracking in factory video
*For every frame: each of the worker's hands, left/right, 21 keypoints, the same ID across frames.*

| What | Where |
|---|---|
| Detection, tracking, 3D reconstruction, smoothing, gap fill | `handtrace/wilor/` |
| Wearer filter, side vote, relink, cap, gap fill, renumber | `handtrace/postprocess.py` |
| Re-reconstruction of side-corrected hands | `handtrace/wilor/refine.py` |
| Resumable multi-GPU run | `handtrace/runner.py` (`stage="ps1"`) |
| Output | `<clip>/hands_final.jsonl` |
| Bonus items | smoothing (One-Euro), re-entry (relink), left/right fixes (vote + refine), coverage and fps (`metrics.json`) |

## PS2 — What is the hand touching?
*Per hand and frame: nothing / portable / fixed; object box and name; grab/release timeline.*

| What | Where |
|---|---|
| Object detection (OWLv2 / SAM 3 / YOLO), workspace prompts | `handtrace/ps2/object_detector.py`, `configs/ps2_objects.json` |
| Object IDs | `handtrace/ps2/object_tracker.py` |
| Fingertip rule, grasp label, co-contact | `handtrace/ps2/contact_detector.py` |
| Temporal state machine | `handtrace/ps2/state_machine.py` |
| Output | `<clip>/ps2_final/contacts.csv` (`clip, t_start, t_end, hand, state, object_name`, …) |
| Bonus items | open-vocabulary naming, tool-use counts (dashboard), hand-near-machine safety flags (PS5 safety) |

## PS3 — Task-step timeline
*Actions with start, end, verb, object and hand.*

| What | Where |
|---|---|
| Windows, prompt, Gemini client, schema | `handtrace/ps3/windows.py`, `prompt.py`, `gemini_client.py`, `schema.py` |
| Validation, merging | `handtrace/ps3/validate.py`, `merge.py` |
| Edges refined with PS2 grab / release | `handtrace/ps3/refine.py` |
| Missing / out-of-order steps, two-worker comparison | `handtrace/ps3/compare.py` |
| Output | `<clip>/ps3/actions_final.csv` (`clip, start_s, end_s, verb, object, hand`, …) |

## PS5 — Video to insight
*A working tool built on the above.*

| Tool | Where | Output |
|---|---|---|
| Quality filter | `handtrace/ps5/quality.py` | `<clip>/ps5/quality.json` |
| Safety monitor | `handtrace/ps5/safety.py` | `<clip>/ps5/safety_events.csv` |
| Event index + plain-English search | `handtrace/ps5/index.py`, `search.py` | `ps5_events_all.csv` |
| Dashboard | `handtrace/ps5/dashboard.py` | `ps5_dashboard.html` |
| Training video | `handtrace/ps5/training.py` | private MP4 + `.srt` + steps CSV |

## Shared: review and evaluation
`handtrace/review.py`, `review_widgets.py`, `review_html.py`, `render.py` (checking every stage on the original
video) and `eval/` (scoring). See [evaluation.md](evaluation.md).
