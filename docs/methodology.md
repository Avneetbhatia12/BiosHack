# Methodology

## PS1 — hand tracking

**WiLoR** (CVPR 2025, official code and weights, pinned commit `fcb9113`) provides two models: a YOLO hand detector
with left/right classes, and a ViT-based reconstruction network that fits the MANO hand model to each hand crop.

1. **Detection** (`wilor/wilor_detector.py`): batched YOLO at `det_imgsz` (default 640), low confidence threshold
   (0.2) so weak detections can still continue a track, class-agnostic NMS (a hand detected as both left and right
   keeps the better box), top-K per frame.
2. **Tracking** (`wilor/hand_tracker.py`): ByteTrack-style two stages with Hungarian assignment. The cost combines box
   IoU against a damped constant-velocity prediction, centre distance, a soft side penalty and box-size consistency.
   Stage 1 matches high-score detections; stage 2 lets low-score detections (blur) continue unmatched tracks. Each
   track keeps an exponentially-weighted left/right vote that overrides isolated label flips.
3. **Reconstruction** (`wilor/wilor_inference.py`): the official crop and forward pass; the 21 joints (OpenPose /
   MediaPipe order: 0 wrist, tips 4/8/12/16/20) are projected to pixels with the official full-image camera. Precision
   modes: fp32, fp16 only, or WiLoR's `--fast` (fp16 + ViT block skipping). `benchmark_modes` measures each mode's
   keypoint deviation from fp32 and picks the fastest one inside a pixel tolerance.
4. **Smoothing** (`wilor/smoothing.py`): a One-Euro filter per track on 2D and 3D keypoints, reset after dropouts.
5. **Gap fill**: linear interpolation of dropouts up to 8 frames, flagged `interpolated: true`.

**Post-processing** (`postprocess.py`, pure Python over the whole clip, so it can use the future as well as the past):

| Step | Rule |
|---|---|
| dedup | a box overlapping a better one by > 0.6 of the smaller box is the same hand |
| wearer filter | track-level score from hand size, height in the frame and forearm direction; background workers' tracks removed (or flagged) |
| side vote | whole-track weighted majority when clear (≥ 75 %), otherwise a centred 15-frame majority |
| relink | a track starting ≤ 2 s after a same-side track ended, near where it ended (or both at a frame edge), continues its ID |
| cap | at most one left and one right wearer hand; two same-side hands → the less certain one is flipped |
| gap fill, renumber | interpolate gaps ≤ 8 frames inside a final ID; IDs 1..n in order of appearance |

Hands whose side changed are re-run through WiLoR with the corrected side (`wilor/refine.py`), because WiLoR mirrors
left-hand crops and a wrong side gives a mirrored fit.

## PS2 — hand-object contact

- **Objects** (`ps2/object_detector.py`): OWLv2 (default, open vocabulary), SAM 3 (optional, adds instance masks;
  gated weights) or Ultralytics YOLO (COCO, or a model fine-tuned on canonical names). Prompts come from
  **workspace profiles** in `configs/ps2_objects.json`, chosen from the clip name; each object maps to portable /
  fixed / unknown. Per-class confidence, size limits and NMS are applied after detection.
- **Object IDs** (`ps2/object_tracker.py`): IoU + centre-distance matching with a type gate (a portable box never
  continues a fixed track); each track reports its confidence-weighted majority name. Fixed objects are re-detected
  every frame, so they follow the moving camera.
- **Fingertip rule** (`ps2/contact_detector.py`): pad = 0.15 × the larger hand-box side; a candidate needs ≥ 2 of
  fingertips 4/8/12/16/20 inside the padded object box (and near the mask, when SAM 3 provides one). Candidates rank by
  object type (portable before fixed, so a tool held above a table is a grab), then fingertip count, distance and
  confidence. Each contact gets a grasp label (pinch / power / hook / partial). Listed pairs (cloth + slab) may be
  touched together.
- **State machine** (`ps2/state_machine.py`): per hand × object ID, a contact opens after 5 consecutive candidate
  frames (notebook default), starts at the first candidate frame, survives a 3-frame release grace and ends at the
  last supported frame. Interpolated hands cannot start a contact. Portable contacts become `grab` / `release`, fixed
  ones `touch_start` / `touch_end`.

## PS3 — task-step timeline

1. **Windows** (`ps3/windows.py`): overlapping 60 s windows (10 s overlap) over the original timeline.
2. **Gemini** (`ps3/gemini_client.py`): uploads a 480p copy with the same timing, sends each window with video offsets,
   `fps=2`, a fixed verb vocabulary and a JSON response schema (`ps3/schema.py`); retries with backoff; caches every
   reply in `ps3/raw/`; deletes the upload when the clip is done. The model is auto-selected from the key's live list.
3. **Validation** (`ps3/validate.py`): JSON recovery from truncated replies, time parsing (`12.5s`, `1:02:03`),
   time-base detection (absolute vs window-relative), window bounds, verb / hand / object normalisation.
4. **Merge** (`ps3/merge.py`): the same action described in two overlapping windows becomes one.
5. **PS2 refinement** (`ps3/refine.py`): action edges snap to the exact grab / release frames of a compatible hand and
   object within 1.5 s; unknown hands / objects are filled from the overlapping contact; PS2 grab→release intervals
   with no action are listed as possible missed steps.
6. **Comparison** (`ps3/compare.py`): missing / out-of-order steps against an expected list (longest increasing
   subsequence), and two workers on the same task (difflib alignment, who is faster).

## PS5 — video to insight

| Tool | Method |
|---|---|
| Quality filter (`ps5/quality.py`) | 2 frames/s on a 480p copy: Laplacian-variance sharpness, mean brightness, PS1 hand count; bad stretches and a clip verdict |
| Safety monitor (`ps5/safety.py`) | hazard objects by name (machine, spindle, blade, press, conveyor…); per hand: touching (PS2), fingertip inside, or near; stretches ≥ `min_s`; optional motion check of the hazard box |
| Event index (`ps5/index.py`) | one table of PS1 visibility, PS2 contacts, PS3 actions and PS5 flags across clips |
| Plain-English search (`ps5/search.py`) | deterministic offline parser: hands, verbs (vocabulary + synonyms), known objects (with plural / close spellings), durations, times (`after 1:30`), clips; the parsed filters are shown with the results |
| Dashboard (`ps5/dashboard.py`) | time per step type, tool use, hand utilisation, quality and safety per clip; HTML with numbers only |
| Training video (`ps5/training.py`) | PS3 steps cut, captioned, with title cards, `.srt` subtitles and a step list (private footage) |
