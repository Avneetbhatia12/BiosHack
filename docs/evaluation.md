# Evaluation

There are no ground-truth labels for the hackathon footage, so accuracy is measured by **people checking samples**.
Everything the pipeline prints by itself (share of frames with hands, fps, number of contacts) is **coverage and
speed, not accuracy**. This repository contains **no measured accuracy results**; the numbers come from running the
audit below on your own copy of the footage.

## Manual audit (main method)

`handtrace.review` draws every stage on the original frames. The **inspector widget** (notebook §10) or the
standalone **HTML player** (`review_html.py`, numbers only, opens the clip from the reviewer's disk) shows a frame and
records, *among the camera wearer's own visible hands*:

| Field (`review.AUDIT_FIELDS`) | Meaning |
|---|---|
| `pred_hands` | hands the pipeline drew on the frame |
| `missed_hands` | visible wearer hands with no prediction |
| `false_hands` | predictions that are not a wearer hand (background worker, artefact) |
| `side_swaps` | found hands with left/right wrong |
| `bad_keypoints` | found hands whose keypoints sit clearly on the wrong joints |
| `ps2_contact`, `ps3_action`, `safety_flag` | the shown contact, step and flag judged ok / wrong / n.a. |

`sample_audit_frames` picks about 100 frames per clip: 60 % from hard moments (low detector score, interpolated,
side corrected, two hands close together, frame edge, PS2 grab, no hand found), spaced at least one second apart, and
40 % uniformly at random; alternate frames are marked `dev` / `test`. With found = `pred_hands` − `false_hands`,
`audit_report(OUT_ROOT, stems, split=...)` computes:

- **hand recall** = found / (found + missed)
- **hand precision** = found / predicted
- **side accuracy** = (found − side swaps) / found
- **keypoint-OK rate** = (found − bad keypoints) / found
- **fully correct frames** = frames with no miss, no false hand, no swap and no bad keypoints
- PS2 / PS3 / safety accuracy where they were judged

Always report the number of frames reviewed. Tune thresholds on `dev` and report `test`.

## Exact labels (optional)

- `eval/label.py` + `eval/metrics.py`: boxes, sides and clicked keypoints on sampled frames → recall, precision
  (background-hand false positives counted separately), side accuracy, PCK@0.2 of hand size, mean pixel error, ID
  switches. Labels from 480p copies are rescaled to the prediction resolution.
- `eval/label_events.py` + `eval/contact_metrics.py`: grab / release events per hand → event precision / recall,
  start and end within ±0.5 s, object-name accuracy; or a frame-level contact sheet → contact precision / recall.

## Limitations

- The audit is a sample: a few hundred frames cannot show every failure; hard moments are over-sampled on purpose, so
  rates are not uniform-frame averages unless computed on the random part only.
- Judgements such as "keypoints clearly off" are subjective; use the same reviewer guidelines across the team.
- PS2 contact is 2D proximity: depth ambiguity (a fingertip in front of an object) cannot be resolved from boxes.
- PS3 times depend on the model's frame sampling (2 fps); refinement only helps where PS2 saw the matching contact.
- Development observations recorded in the notebook (not accuracy): WiLoR takes roughly 90–100 ms per hand on a T4;
  WiLoR's `--fast` mode moved keypoints by about 3.6–4.2 px in earlier runs; a 1280-pixel detector input found about
  40 % more hand boxes than 640 at about 4× detector time, which must be confirmed as real wearer hands by the audit.
