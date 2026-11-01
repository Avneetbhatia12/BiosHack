"""PS2: hand-object contact and grab/release timeline on top of the PS1 (WiLoR) hands.jsonl.

object_detector (OWLv2 / YOLO) -> object_tracker (object IDs "O<n>") -> contact_detector
(fingertip rule + state machine) -> pipeline (outputs under <ps1 clip dir>/ps2/).
Heavy libraries (torch, transformers, ultralytics) are imported lazily.
"""
