"""Label grab / release (touch start / end) events per hand, fast, with the keyboard. Local and private.

Judges check that every grab and release is found and that times are right (about +-0.5 s), so the
ground truth worth having is the EVENT LIST, not per-frame boxes. Step through the clip, mark when a
hand starts and stops touching something, pick the object from the clip's workspace profile, done.
Output: eval/labels/<clip>_events.csv (gitignored) with clip, hand, object_name, object_type, t_start,
t_end, start_frame, end_frame, which `eval/contact_metrics.py timing` scores against contacts.csv.

    python eval/label_events.py --video data/clips_480p/<clip>.mp4 [--hands-jsonl .../hands.jsonl]

Keys: space play/pause | d / a next / previous frame | D / A +-15 frames | ] / [ +-5 s
      l / r  start (first press) or end (second press) an event for the wearer's LEFT / RIGHT hand
      o / O  next / previous object for the open event (from the profile list)
      u  undo last event | s save | q save and quit
--hands-jsonl draws the PS1 hands so you can see them in blur; PS2 predictions are never shown, so the
labels stay independent of the system being evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ["clip", "hand", "object_name", "object_type", "t_start", "t_end", "start_frame", "end_frame"]


class EventBook:
    """Open / close events per hand; pure logic, no GUI."""

    def __init__(self, clip: str, fps: float, objects: list, type_of: dict):
        self.clip, self.fps, self.objects, self.type_of = clip, fps, objects, type_of
        self.open: dict = {}                 # hand -> {"start_frame", "obj_i"}
        self.events: list = []

    def toggle(self, hand: str, frame: int) -> str:
        if hand in self.open:
            ev = self.open.pop(hand)
            if frame < ev["start_frame"]:
                self.open[hand] = ev
                return "end before start: ignored"
            name = self.objects[ev["obj_i"]]
            self.events.append({"clip": self.clip, "hand": hand, "object_name": name,
                                "object_type": self.type_of.get(name, "unknown"),
                                "start_frame": ev["start_frame"], "end_frame": frame,
                                "t_start": round(ev["start_frame"] / self.fps, 3), "t_end": round(frame / self.fps, 3)})
            return f"closed {hand} {name} {ev['start_frame']}-{frame}"
        self.open[hand] = {"start_frame": frame, "obj_i": 0}
        return f"opened {hand} at {frame}"

    def cycle_object(self, step: int = 1) -> None:
        for ev in self.open.values():
            ev["obj_i"] = (ev["obj_i"] + step) % len(self.objects)

    def undo(self) -> None:
        if self.open:
            self.open.popitem()
        elif self.events:
            self.events.pop()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sorted(self.events, key=lambda e: (e["start_frame"], e["hand"])))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, clip, fps, objects, type_of) -> "EventBook":
        b = cls(clip, fps, objects, type_of)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["start_frame"], r["end_frame"] = int(r["start_frame"]), int(r["end_frame"])
                    b.events.append(r)
        return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--hands-jsonl", type=Path)
    ap.add_argument("--prompts", default=str(ROOT / "configs" / "ps2_objects.json"))
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "eval" / "labels")
    ap.add_argument("--max-width", type=int, default=1280)
    a = ap.parse_args(argv)
    import cv2
    from handtrace.io import VideoCursor
    from handtrace.ps2.config import ObjectVocabulary

    clip = a.video.stem
    vocab = ObjectVocabulary.load(a.prompts, a.profile, clip)
    objects = list(vocab.prompts) + ["other_object"]
    cur = VideoCursor(a.video)
    out = a.labels_dir / f"{clip}_events.csv"
    book = EventBook.load(out, clip, cur.fps, objects, vocab.type_of)
    hands = {}
    if a.hands_jsonl:
        from handtrace.wilor.visualization import draw_hand
        hands = {r["frame"]: r for r in map(json.loads, open(a.hands_jsonl, encoding="utf-8"))}
    f, playing, msg = 0, False, f"profile {vocab.profile}: {len(objects)} objects"
    while True:
        img = cur.read(f)
        if img is None:
            f, playing = max(0, cur.frame_count - 1), False
            img = cur.read(f)
            if img is None:
                break
        for h in hands.get(f, {}).get("hands", []):
            draw_hand(img, h)
        s = min(1.0, a.max_width / img.shape[1])
        view = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
        opened = " | ".join(f"{h.upper()} open @{e['start_frame']}: {objects[e['obj_i']]}" for h, e in book.open.items())
        lines = [f"frame {f}/{cur.frame_count}  t={f / cur.fps:.2f}s  events={len(book.events)}  {'PLAY' if playing else ''}",
                 opened or "l / r = start or end a LEFT / RIGHT hand event", msg]
        for k, line in enumerate(lines):
            cv2.putText(view, line, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, line, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("events", view)
        key = cv2.waitKey(int(1000 / cur.fps) if playing else 0) & 0xFF
        if key == 255 and playing:
            f += 1
            continue
        ch = chr(key) if key < 128 else ""
        step = {"d": 1, "a": -1, "D": 15, "A": -15, "]": int(5 * cur.fps), "[": -int(5 * cur.fps)}
        if ch == " ":
            playing = not playing
        elif ch in step:
            f, playing = max(0, f + step[ch]), False
        elif ch in ("l", "r"):
            msg = book.toggle("left" if ch == "l" else "right", f)
        elif ch in ("o", "O"):
            book.cycle_object(1 if ch == "o" else -1)
        elif ch == "u":
            book.undo(); msg = "undone"
        elif ch == "s":
            book.save(out); msg = f"saved {len(book.events)} events"
        elif ch == "q" or cv2.getWindowProperty("events", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif playing:
            f += 1
    book.save(out)
    cur.close()
    cv2.destroyAllWindows()
    print(f"[events] {len(book.events)} events -> {out}" + (f" ({len(book.open)} unfinished, not saved)" if book.open else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
