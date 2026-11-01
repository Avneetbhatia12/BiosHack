# eval/ — measuring accuracy

Numbers printed by the pipeline (frames with hands, fps, contact counts) are **coverage and speed**.
Accuracy comes only from frames or events that a person checked. Always report the count with the number.

| Tool | What it scores | Input you create |
|---|---|---|
| `handtrace.review.audit_report` (notebook §10–12) | PS1 hand recall, precision, left/right accuracy, keypoint-OK rate, fully-correct frames; PS2 / PS3 / safety where judged | the manual audit: inspector widget or HTML player, ~100 frames per clip |
| `eval/label.py` + `eval/metrics.py` | PS1 boxes, side, PCK@0.2 on clicked keypoints, ID switches | exact hand boxes / keypoints on sampled frames |
| `eval/contact_metrics.py` | PS2 contact precision / recall, type and name accuracy; event start/end within ±0.5 s | a filled contact sheet, or an event list from `eval/label_events.py` |

```bash
python eval/label.py --video data/clips_480p/<clip>.mp4 --n 40 --split dev          # local OpenCV tool
python eval/metrics.py --labels-dir eval/labels --pred-root outputs --pred-name hands_final.jsonl
python eval/label_events.py --video data/clips_480p/<clip>.mp4                       # grab / release events
python eval/contact_metrics.py timing --pred outputs/<clip>/ps2_final/contacts.csv --gt eval/labels/<clip>_events.csv
python eval/contact_metrics.py template --contacts outputs/<clip>/ps2_final/contacts.jsonl --n 50 --out eval/labels/<clip>_contacts.csv
python eval/contact_metrics.py score --sheet eval/labels/<clip>_contacts.csv
```

Labels are private (derived from the footage) and stay in `eval/labels/`, which is gitignored.
Tune thresholds on `dev` clips and report `test` (demo) clips. Method and limitations: [docs/evaluation.md](../docs/evaluation.md).
