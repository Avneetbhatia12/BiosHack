"""PS1: post-processing, WiLoR pipeline modules (fake models), PS1 evaluation. Synthetic data only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from handtrace.postprocess import PostConfig, overlap_small, postprocess, wearer_frame_score
from handtrace.wilor.config import PipelineConfig
from handtrace.wilor.hand_tracker import HandTracker
from handtrace.wilor.pipeline import GapFillBuffer, run_video
from handtrace.wilor.refine import rereconstruct_corrected
from handtrace.wilor.smoothing import OneEuroFilter
from tests.conftest import make_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import label  # noqa: E402
import metrics  # noqa: E402


# ============================================================================= from test_postprocess.py
W, H = 1920, 1080


def hand(tid, side, cx, cy, size=200.0, score=0.9, interp=False):
    """Upright synthetic hand: wrist at the bottom, fingers up (wearer-like forearm direction)."""
    kp = np.zeros((21, 2))
    kp[0] = (cx, cy + size / 2)
    for j in range(1, 21):
        finger, k = (j - 1) // 4, (j - 1) % 4
        kp[j] = (cx - size / 3 + finger * size / 6, cy + size / 4 - (k + 1) * size / 6)
    kp[9] = (cx, cy - size / 8)
    x1, y1 = kp.min(0)
    x2, y2 = kp.max(0)
    return {"id": tid, "side": side, "bbox": [x1, y1, x2, y2], "score": score,
            "kp2d": kp.tolist(), "kp3d": np.zeros((21, 3)).tolist(), "interpolated": interp}


def recs(n, fn):
    return [{"frame": f, "t": f / 30, "hands": fn(f)} for f in range(n)]


def cfg(**kw):
    return PostConfig(frame_wh=(W, H), **kw)


def test_wearer_score_prefers_big_low_hands():
    c = cfg()
    big_low = wearer_frame_score(hand(1, "left", 600, 850, size=300), c)
    small_high = wearer_frame_score(hand(2, "left", 900, 250, size=60), c)
    assert big_low > 0.6 > 0.4 > small_high


def test_background_track_removed_and_flag_mode_keeps_it():
    r = recs(30, lambda f: [hand(1, "left", 600, 850, 300), hand(2, "right", 1300, 850, 300),
                            hand(3, "right", 900, 250, 60)])
    out, rep = postprocess(r, cfg())
    assert rep["wearer"]["background_tracks"] == 1
    assert all(len(x["hands"]) == 2 for x in out)
    out2, _ = postprocess(r, cfg(wearer_filter=False, max_hands_per_side=2))
    assert any(not h["is_wearer"] for h in out2[0]["hands"])


def test_duplicate_box_on_same_hand_removed():
    def two(f):
        a = hand(1, "left", 600, 850, 300, score=0.9)
        b = hand(2, "right", 610, 860, 280, score=0.5)
        return [a, b]
    out, rep = postprocess(recs(10, two), cfg())
    assert rep["duplicates_removed"] == 10
    assert all(len(x["hands"]) == 1 and x["hands"][0]["side"] == "left" for x in out)


def test_isolated_side_flicker_fixed_by_track_majority():
    r = recs(40, lambda f: [hand(1, "right" if f in (10, 11) else "left", 600, 850, 300)])
    out, rep = postprocess(r, cfg())
    assert {h["side"] for x in out for h in x["hands"]} == {"left"}
    assert rep["side_changes"] == 2
    assert out[10]["hands"][0]["side_corrected"] and out[10]["hands"][0]["side_raw"] == "right"


def test_fragmented_track_relinked_and_gap_filled():
    def fn(f):
        if f < 20:
            return [hand(1, "left", 600 + f, 850, 300)]
        if f < 25:
            return []
        return [hand(7, "left", 630 + f, 850, 300)]
    out, rep = postprocess(recs(50, fn), cfg())
    assert rep["ids_in"] == 2 and rep["ids_out"] == 1 and rep["relinks"] == 1
    assert rep["gap_fills"] == 5
    assert all(len(x["hands"]) == 1 for x in out)
    assert out[22]["hands"][0]["interpolated"] is True


def test_no_relink_when_far_or_too_late():
    def far(f):
        return [hand(1, "left", 500, 850, 300)] if f < 10 else ([hand(2, "left", 1500, 500, 300)] if f > 12 else [])
    _, rep = postprocess(recs(30, far), cfg(edge_margin=0.0))
    assert rep["ids_out"] == 2
    def late(f):
        return [hand(1, "left", 600, 850, 300)] if f < 10 else ([hand(2, "left", 600, 850, 300)] if f > 100 else [])
    _, rep = postprocess(recs(120, late), cfg())
    assert rep["ids_out"] == 2


def test_two_same_side_hands_resolved_by_position():
    out, rep = postprocess(recs(5, lambda f: [hand(1, "left", 500, 850, 300), hand(2, "left", 1400, 850, 300)]),
                           cfg())
    sides = {h["kp2d"][0][0] < W / 2: h["side"] for h in out[0]["hands"]}
    assert sides == {True: "left", False: "right"}
    assert rep["pair_side_flips"] == 5


def test_input_not_mutated_and_output_ids_compact():
    r = recs(5, lambda f: [hand(42, "left", 600, 850, 300)])
    out, _ = postprocess(r, cfg())
    assert r[0]["hands"][0]["id"] == 42 and out[0]["hands"][0]["id"] == 1


def test_overlap_small():
    assert overlap_small([0, 0, 10, 10], [2, 2, 5, 5]) == 1.0
    assert overlap_small([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0




# ============================================================================= from test_wilor.py
def det(x, y, s=20.0, score=0.9, right=False):
    return [x - s, y - s, x + s, y + s, score, 1.0 if right else 0.0]


def test_tracker_keeps_ids_and_recovers_through_low_score():
    tr = HandTracker(PipelineConfig(), (640, 480))
    ids = []
    for f in range(30):
        score = 0.25 if 10 <= f < 15 else 0.9            # blur: score drops below det_high_conf
        a = tr.step(np.array([det(100 + f, 300, score=score), det(500 - f, 300, right=True)]), f)
        ids.append(sorted(x.track_id for x in a))
    assert all(i == [1, 2] for i in ids)
    assert tr.next_id == 3


def test_tracker_overrides_isolated_side_flip():
    tr = HandTracker(PipelineConfig(), (640, 480))
    sides = [tr.step(np.array([det(200, 300, right=(f == 20))]), f)[0].is_right for f in range(30)]
    assert not any(sides)


def test_one_euro_constant_signal_is_unchanged():
    f = OneEuroFilter(1.0, 0.0)
    x = np.array([5.0, 7.0])
    for i in range(10):
        y = f(x, i / 30)
    assert np.allclose(y, x)


def test_gap_fill_buffer_interpolates_short_dropout():
    out = []
    buf = GapFillBuffer(delay=6, max_gap=3, enabled=True, sink=out.append)
    h = lambda x: {"id": 1, "side": "left", "bbox": [x, 0, x + 10, 10], "score": 0.9,
                   "kp2d": [[x, 0]] * 21, "kp3d": [[0, 0, 0]] * 21, "interpolated": False}
    for f in range(8):
        buf.push({"frame": f, "t": f / 30, "hands": [h(f * 10)] if f not in (3, 4) else []})
    buf.flush()
    assert [len(r["hands"]) for r in out] == [1] * 8
    assert out[3]["hands"][0]["interpolated"] and out[3]["hands"][0]["bbox"][0] == 30


class FakeDetector:
    last_time_s = 0.0

    def __call__(self, frames):
        return [np.array([det(20, 20, 8, 0.9, False), det(44, 24, 8, 0.9, True)], np.float32)
                for _ in frames]


class FakeRecon:
    last_time_s = 0.0

    def __init__(self):
        self.calls = []

    def reconstruct(self, frames, requests, return_vertices=False):
        self.calls.append(requests)
        out = []
        for r in requests:
            cx, cy = (r.bbox[0] + r.bbox[2]) / 2, (r.bbox[1] + r.bbox[3]) / 2
            out.append({"joints": np.zeros((21, 3)), "cam_t": np.array([0, 0, 1.0]),
                        "kp2d": np.tile([cx, cy], (21, 1))})
        return out


def test_run_video_end_to_end_with_fakes(tmp_path):
    video = make_video(tmp_path / "clip.mp4", n=20, fps=10.0, size=(64, 48))
    cfg = PipelineConfig(video_path=str(video), output_dir=str(tmp_path / "out"), write_video=False)
    res = run_video(cfg, detector=FakeDetector(), reconstructor=FakeRecon(), log=lambda *a: None)
    recs = [json.loads(l) for l in open(res["jsonl"])]
    assert [r["frame"] for r in recs] == list(range(20))
    assert all(sorted((h["id"], h["side"]) for h in r["hands"]) == [(1, "left"), (2, "right")] for r in recs)
    h = recs[0]["hands"][0]
    assert len(h["kp2d"]) == 21 and len(h["bbox"]) == 4 and "kp3d_root_cam" in h
    meta = json.load(open(tmp_path / "out" / "clip" / "hands_meta.json"))
    assert meta["video"]["width"] == 64


def test_rereconstruct_only_corrected_hands(tmp_path):
    video = make_video(tmp_path / "clip.mp4", n=6, fps=10.0, size=(64, 48))
    hand = lambda corr: {"id": 1, "side": "right", "bbox": [10, 10, 30, 30], "score": 0.9,
                         "kp2d": [[0, 0]] * 21, "kp3d": [[0, 0, 0]] * 21, "side_corrected": corr}
    recs = [{"frame": f, "t": f / 10, "hands": [hand(f == 3)]} for f in range(6)]
    fake = FakeRecon()
    n = rereconstruct_corrected(str(video), recs, fake, log=lambda *a: None)
    assert n == 1 and recs[3]["hands"][0]["kp2d"][0] == [20.0, 20.0]
    assert recs[3]["hands"][0]["rereconstructed"] and "rereconstructed" not in recs[2]["hands"][0]
    assert fake.calls[0][0].is_right is True


# ============================================================================= from test_metrics.py
def gt_hand(side, x, wearer=True):
    return {"bbox": [x, 100, x + 100, 200], "side": side, "wearer": wearer,
            "kp": {"0": [x + 50, 200], "8": [x + 50, 100]}}


def pred_hand(tid, side, x, kp_shift=0.0):
    kp = [[x + 50 + kp_shift, 150]] * 21
    kp[0] = [x + 50 + kp_shift, 200]
    kp[8] = [x + 50 + kp_shift, 100]
    return {"id": tid, "side": side, "bbox": [x, 100, x + 100, 200], "score": 0.9, "kp2d": kp}


def labels(frames):
    return {"video": "v.mp4", "width": 1000, "height": 500,
            "frames": {str(f): {"split": "dev", "hands": h} for f, h in frames.items()}}


def test_perfect_prediction():
    lab = labels({0: [gt_hand("left", 100), gt_hand("right", 600)], 5: [gt_hand("left", 110)]})
    recs = [{"frame": 0, "hands": [pred_hand(1, "left", 100), pred_hand(2, "right", 600)]},
            {"frame": 5, "hands": [pred_hand(1, "left", 110)]}]
    m = metrics.score(lab, recs)
    assert m["frames"] == 2 and m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["side_accuracy"] == 1.0 and m["pck"] == 1.0 and m["id_switches"] == 0
    assert m["id_pairs_checked"] == 1


def test_miss_swap_background_fp_and_id_switch():
    lab = labels({0: [gt_hand("left", 100), gt_hand("right", 600, wearer=False)],
                  3: [gt_hand("left", 100)], 4: [gt_hand("left", 100)]})
    recs = [{"frame": 0, "hands": [pred_hand(1, "right", 100), pred_hand(2, "right", 600)]},
            {"frame": 3, "hands": []},
            {"frame": 4, "hands": [pred_hand(9, "left", 100, kp_shift=80)]}]
    m = metrics.score(lab, recs)
    assert (m["tp"], m["fn"], m["fp_background"], m["fp_other"]) == (2, 1, 1, 0)
    assert m["side_accuracy"] == 0.5 and m["pck"] == 0.5 and m["id_switches"] == 1


def test_labels_scaled_to_prediction_resolution():
    lab = labels({0: [gt_hand("left", 100)]})
    recs = [{"frame": 0, "hands": [pred_hand(1, "left", 200)]}]
    recs[0]["hands"][0]["bbox"] = [200, 200, 400, 400]
    m = metrics.score(lab, recs, pred_wh=(2000, 1000))
    assert m["recall"] == 1.0


def test_metrics_cli(tmp_path):
    lab = tmp_path / "v.json"
    lab.write_text(json.dumps(labels({0: [gt_hand("left", 100)]})))
    pred = tmp_path / "v" / "hands_final.jsonl"
    pred.parent.mkdir()
    pred.write_text(json.dumps({"frame": 0, "hands": [pred_hand(1, "left", 100)]}) + "\n")
    out = tmp_path / "r.json"
    assert metrics.main(["--labels-dir", str(tmp_path), "--pred-root", str(tmp_path),
                         "--pred-size", "1000x500", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["v"]["recall"] == 1.0


def test_pick_frames_and_segments():
    f = label.pick_frames(1000, n=10, seed=1, segments=[(100, 120, 5)])
    assert len(f) >= 10 and {100, 105, 110, 115} <= set(f) and f == sorted(f) and max(f) < 1000
    assert label.parse_segment("10:20") == (10, 20, 1)
    with pytest.raises(Exception):
        label.parse_segment("20:10")


def test_label_store_roundtrip_and_resolution_guard(tmp_path):
    p = tmp_path / "labels" / "clip.json"
    s = label.LabelStore(p, "clip.mp4", 854, 480)
    s.set(12, [gt_hand("left", 10)], "test")
    s.save()
    s2 = label.LabelStore(p, "clip.mp4", 854, 480)
    assert s2.has(12) and s2.get(12)[0]["side"] == "left" and not s2.has(13)
    with pytest.raises(SystemExit):
        label.LabelStore(p, "clip.mp4", 1920, 1080)
