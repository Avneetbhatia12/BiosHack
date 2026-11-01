"""Label hands on sampled frames of one clip, locally, for eval/metrics.py.

Labels are written to eval/labels/<video_stem>.json (gitignored). No frame is ever saved.
Label every frame BLIND to predictions, so the labels do not copy the model's mistakes.

    python eval/label.py --video data/clips_480p/<clip>.mp4 --n 40 --split dev
    python eval/label.py --video data/clips_480p/<clip>.mp4 --segment 900:1200:5 --split test

Per frame: draw a tight box around each visible hand (wrist to fingertips), then
    l / r     side of the selected hand (the WEARER's left / right, not the image side)
    b         toggle background worker (not the camera wearer)
    k         click wrist, thumb, index, middle, ring, pinky tips in that order (x = skip point)
    tab       select next hand        d  delete selected hand      u  undo last point
    n / space save frame and go on    p  previous frame            q  save and quit
A frame saved with no boxes means "no wearer hand visible", which is a valid label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KP_ORDER = (0, 4, 8, 12, 16, 20)
KP_NAMES = ("wrist", "thumb", "index", "middle", "ring", "pinky")
DEFAULT_DIR = ROOT / "eval" / "labels"


class LabelStore:
    """Load / atomically save one clip's label file."""

    def __init__(self, path: Path, video: str, width: int, height: int):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if (self.data["width"], self.data["height"]) != (width, height):
                raise SystemExit(f"{path} was labelled at {self.data['width']}x{self.data['height']}, "
                                 f"this video is {width}x{height}; use the same video version")
        else:
            self.data = {"video": video, "width": width, "height": height, "frames": {}}

    def has(self, frame: int) -> bool:
        return str(frame) in self.data["frames"]

    def get(self, frame: int) -> list[dict]:
        return [dict(h) for h in self.data["frames"].get(str(frame), {}).get("hands", [])]

    def set(self, frame: int, hands: list[dict], split: str) -> None:
        self.data["frames"][str(frame)] = {"split": split, "hands": hands}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def pick_frames(frame_count: int, n: int = 0, seed: int = 0,
                segments: list[tuple[int, int, int]] = ()) -> list[int]:
    """n stratified frames (one random frame per equal slice) plus every segment frame."""
    rng = np.random.default_rng(seed)
    out = set()
    if n > 0 and frame_count > 0:
        edges = np.linspace(0, frame_count, n + 1).astype(int)
        for a, b in zip(edges, edges[1:]):
            if b > a:
                out.add(int(rng.integers(a, b)))
    for start, stop, step in segments:
        out.update(range(max(0, start), min(stop, frame_count), max(1, step)))
    return sorted(out)


def parse_segment(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        parts.append(1)
    if len(parts) != 3 or parts[1] <= parts[0]:
        raise argparse.ArgumentTypeError("segment must be START:STOP[:STEP] with STOP > START")
    return tuple(parts)


# ----------------------------------------------------------------------------- GUI
def run_gui(video: Path, frames: list[int], store: LabelStore, split: str, max_w: int) -> None:
    import cv2
    from handtrace.io import read_frame

    st = {"hands": [], "sel": -1, "drag": None, "kp_mode": False, "kp_i": 0, "scale": 1.0}

    def to_img(x, y):
        return [x / st["scale"], y / st["scale"]]

    def on_mouse(ev, x, y, flags, _):
        if st["kp_mode"] and ev == cv2.EVENT_LBUTTONDOWN and st["sel"] >= 0:
            h = st["hands"][st["sel"]]
            h.setdefault("kp", {})[str(KP_ORDER[st["kp_i"]])] = [round(v, 1) for v in to_img(x, y)]
            st["kp_i"] += 1
            if st["kp_i"] >= len(KP_ORDER):
                st["kp_mode"] = False
        elif not st["kp_mode"] and ev == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = (x, y, x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["drag"] = st["drag"][:2] + (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
            x0, y0, x1, y1 = st["drag"]
            st["drag"] = None
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                a, b = to_img(min(x0, x1), min(y0, y1)), to_img(max(x0, x1), max(y0, y1))
                st["hands"].append({"bbox": [round(v, 1) for v in a + b], "side": "?", "wearer": True})
                st["sel"] = len(st["hands"]) - 1

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    i = 0
    while 0 <= i < len(frames):
        f = frames[i]
        img = read_frame(video, f)
        if img is None:
            print(f"[label] frame {f} unreadable, skipped"); i += 1; continue
        st.update(hands=store.get(f), sel=len(store.get(f)) - 1, kp_mode=False, kp_i=0,
                  scale=min(1.0, max_w / img.shape[1]))
        while True:
            view = cv2.resize(img, None, fx=st["scale"], fy=st["scale"]) if st["scale"] < 1 else img.copy()
            s = st["scale"]
            for j, h in enumerate(st["hands"]):
                col = {"left": (255, 160, 40), "right": (40, 120, 255)}.get(h["side"], (200, 200, 200))
                if not h["wearer"]:
                    col = (120, 120, 120)
                b = [int(v * s) for v in h["bbox"]]
                cv2.rectangle(view, b[:2], b[2:], col, 3 if j == st["sel"] else 1)
                tag = f"{j}:{h['side']}{'' if h['wearer'] else ' BG'}"
                cv2.putText(view, tag, (b[0], max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                for k, p in h.get("kp", {}).items():
                    cv2.circle(view, (int(p[0] * s), int(p[1] * s)), 3, col, -1)
            if st["drag"]:
                x0, y0, x1, y1 = st["drag"]
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 1)
            mode = f"click {KP_NAMES[st['kp_i']]}" if st["kp_mode"] else "draw boxes"
            done = "saved" if store.has(f) else "new"
            hud = f"[{i + 1}/{len(frames)}] frame {f} ({done}) | {mode} | l r b k tab d u | n save p prev q quit"
            cv2.rectangle(view, (0, 0), (view.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(view, hud, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("label", view)
            key = cv2.waitKey(20) & 0xFF
            sel = st["hands"][st["sel"]] if 0 <= st["sel"] < len(st["hands"]) else None
            if key in (ord("l"), ord("r")) and sel:
                sel["side"] = "left" if key == ord("l") else "right"
            elif key == ord("b") and sel:
                sel["wearer"] = not sel["wearer"]
            elif key == ord("k") and sel:
                st["kp_mode"], st["kp_i"] = True, 0
                sel["kp"] = {}
            elif key == ord("x") and st["kp_mode"]:
                st["kp_i"] += 1
                st["kp_mode"] = st["kp_i"] < len(KP_ORDER)
            elif key == ord("u") and sel and sel.get("kp"):
                sel["kp"].pop(list(sel["kp"])[-1])
                st["kp_i"] = max(0, st["kp_i"] - 1)
            elif key == 9 and st["hands"]:                     # tab
                st["sel"] = (st["sel"] + 1) % len(st["hands"])
            elif key == ord("d") and sel:
                st["hands"].pop(st["sel"]); st["sel"] = len(st["hands"]) - 1
            elif key in (ord("n"), ord(" "), ord("p"), ord("q")):
                unsided = [h for h in st["hands"] if h["wearer"] and h["side"] == "?"]
                if unsided and key != ord("p"):
                    print("[label] set l/r on every wearer hand before saving"); continue
                if key != ord("p"):
                    store.set(f, st["hands"], split); store.save()
                if key == ord("q"):
                    cv2.destroyAllWindows(); return
                i += -1 if key == ord("p") else 1
                break
            if cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:
                return
    cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Label hands on sampled frames (local, private).")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="stratified random frames to label")
    ap.add_argument("--segment", type=parse_segment, action="append", default=[],
                    help="START:STOP[:STEP] continuous frames (for ID-switch / L-R flip checks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="dev = used for tuning; test = demo clips, only for reported numbers")
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--revisit", action="store_true", help="also show frames already labelled")
    ap.add_argument("--max-width", type=int, default=1280, help="display width")
    a = ap.parse_args(argv)
    if a.n <= 0 and not a.segment:
        ap.error("give --n and/or --segment")
    from handtrace.io import VideoCursor
    try:
        meta = VideoCursor(a.video)
    except IOError as e:
        raise SystemExit(f"cannot read {a.video}: {e}")
    meta.close()
    store = LabelStore(a.labels_dir / f"{a.video.stem}.json", a.video.name, meta.width, meta.height)
    frames = pick_frames(meta.frame_count, a.n, a.seed, a.segment)
    if not a.revisit:
        frames = [f for f in frames if not store.has(f)]
    print(f"[label] {len(frames)} frames to label -> {store.path}")
    if frames:
        run_gui(a.video, frames, store, a.split, a.max_width)
    print(f"[label] {len(store.data['frames'])} frames labelled in {store.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
