"""PS2: contact rule, state machine, profiles, masks, pipeline (fake detector), PS2 evaluation tools. Synthetic only."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from handtrace.ps2.config import ObjectVocabulary, PS2Config
from handtrace.ps2.contact_detector import (CSV_COLUMNS, ContactStateMachine, analyze_contacts, event_rows,
                                            frame_candidates, grasp_type, point_box_distance, select_contacts)
from handtrace.ps2.masks import decode_grid, encode_grid, mask_to_grid, point_near_mask
from handtrace.ps2.object_detector import YoloDetector, decode_owl, filter_detections, nms
from handtrace.ps2.object_tracker import ObjectTracker
from handtrace.ps2.pipeline import AlignmentError, check_hand_records, rerun_contacts, run_ps2
from tests.conftest import make_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import contact_metrics  # noqa: E402
import label_events  # noqa: E402


# ============================================================================= from test_ps2_contact.py
TIPS = (4, 8, 12, 16, 20)


def hand(hid, tips, side="right", bbox=(0, 0, 100, 100), interp=False):
    """tips: up to 5 (x, y) fingertip points in order thumb..pinky; others sit at the wrist (far away)."""
    kp = [[-500.0, -500.0]] * 21
    kp = [list(p) for p in kp]
    for i, p in zip(TIPS, tips):
        kp[i] = list(p)
    for i in TIPS[len(tips):]:
        kp[i] = [-500.0, -500.0]
    return {"id": hid, "side": side, "bbox": list(bbox), "score": 0.9, "kp2d": kp, "interpolated": interp}


def obj(oid, box, name="screwdriver", otype="portable", conf=0.8):
    return {"object_id": oid, "object_name": name, "object_type": otype, "bbox": list(box), "confidence": conf}


BOX = (200, 200, 300, 300)


def test_point_box_distance():
    assert point_box_distance((250, 250), BOX) == 0
    assert point_box_distance((310, 250), BOX) == 10


def test_fingertips_inside_and_padded():
    # hand box 100 px -> pad 15 px. Two tips inside, one 10 px outside (inside padding), two far.
    h = hand(1, [(210, 210), (250, 250), (310, 250)])
    c = frame_candidates(h, [obj("O1", BOX)])
    assert c[0]["fingertip_count"] == 3 and c[0]["tips"] == [4, 8, 12] and c[0]["min_tip_dist_px"] == 0


def test_exactly_two_vs_one_fingertip():
    two = hand(1, [(210, 210), (290, 290)])
    one = hand(1, [(210, 210), (400, 400)])
    assert frame_candidates(two, [obj("O1", BOX)])[0]["fingertip_count"] == 2
    assert frame_candidates(one, [obj("O1", BOX)]) == []
    just_outside_pad = hand(1, [(210, 210), (320, 250)])          # 20 px > 15 px pad
    assert frame_candidates(just_outside_pad, [obj("O1", BOX)]) == []


def test_no_object_and_no_hand_keypoints():
    assert frame_candidates(hand(1, [(210, 210), (250, 250)]), []) == []
    assert frame_candidates({"id": 1, "bbox": [0, 0, 1, 1], "kp2d": []}, [obj("O1", BOX)]) == []


def test_ranking_multiple_objects_near_one_hand():
    h = hand(1, [(210, 210), (250, 250), (290, 290)])
    small = obj("O2", (240, 240, 260, 260), conf=0.9)              # 1 tip inside + padding covers more?
    big = obj("O1", BOX, conf=0.5)
    c = frame_candidates(h, [small, big])
    assert c[0]["object_id"] == "O1"                               # more fingertips wins over confidence


def test_tool_held_above_table_is_portable_contact():
    h = hand(1, [(210, 210), (250, 250), (290, 290), (230, 280), (270, 220)])     # all 5 tips over the table
    tool = obj("O2", (200, 200, 260, 260), conf=0.6)                              # 2 tips on the tool
    table = obj("O1", (0, 0, 1000, 1000), "table", "fixed", conf=0.9)
    assert frame_candidates(h, [table, tool])[0]["object_id"] == "O2"
    assert frame_candidates(h, [table, tool], type_priority=None)[0]["object_id"] == "O1"
    spec = [([hand(1, [(210, 210), (250, 250), (290, 290), (230, 280), (270, 220)])], [table, tool])] * 5
    res = run(spec, type_priority=("portable", "unknown", "fixed"))
    assert [(e["object_id"], e["types"].most_common(1)[0][0]) for e in res["events"]] == [("O2", "portable")]


def test_sticky_previous_object_and_multiple_contacts():
    a = {"object_id": "A", "object_type": "portable", "fingertip_count": 3, "tips": [4, 8, 12]}
    b = {"object_id": "B", "object_type": "portable", "fingertip_count": 2, "tips": [16, 20]}
    assert select_contacts([a, b], prev_ids={"B"})[0]["object_id"] == "B"        # within 1 fingertip
    assert select_contacts([a, b], prev_ids={"B"}, sticky_margin=0)[0]["object_id"] == "A"
    assert [c["object_id"] for c in select_contacts([a, b], set(), allow_multiple=True)] == ["A", "B"]
    b_overlap = {"object_id": "B", "object_type": "portable", "fingertip_count": 3, "tips": [4, 8, 12]}
    assert len(select_contacts([a, b_overlap], set(), allow_multiple=True)) == 1


def test_vocabulary_types_and_unknown(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"portable": ["wrench"], "fixed": ["table"], "unknown": ["thing"]}))
    v = ObjectVocabulary.load(p)
    assert (v.object_type("wrench"), v.object_type("table"), v.object_type("thing"), v.object_type("zzz")) == \
        ("portable", "fixed", "unknown", "unknown")
    p.write_text(json.dumps({"portable": ["box"], "fixed": ["box"]}))
    with pytest.raises(ValueError):
        ObjectVocabulary.load(p)


def test_default_prompt_file_loads():
    v = ObjectVocabulary.load()
    assert v.object_type("screwdriver") == "portable" and v.object_type("workbench") == "fixed"
    assert v.coco_map["dining table"] == "fixed"


def test_object_tracker_stable_ids_and_type_gate():
    tr = ObjectTracker((1000, 1000))
    ids = []
    for f in range(10):
        d = [{"bbox": [100 + 3 * f, 100, 200 + 3 * f, 200], "object_name": "wrench" if f % 2 else "spanner",
              "object_type": "portable", "confidence": 0.6},
             {"bbox": [500, 500, 900, 900], "object_name": "table", "object_type": "fixed", "confidence": 0.5}]
        ids.append([o["object_id"] for o in tr.update(d, f)])
    assert all(i == ["O1", "O2"] for i in ids)
    # a fixed detection on top of the portable track does not inherit its ID
    out = tr.update([{"bbox": [130, 100, 230, 200], "object_name": "shelf", "object_type": "fixed", "confidence": 0.9}], 10)
    assert out[0]["object_id"] == "O3"


def test_object_tracker_keeps_id_through_short_miss_then_drops():
    tr = ObjectTracker((1000, 1000), max_missed=2)
    d = [{"bbox": [100, 100, 200, 200], "object_name": "box", "object_type": "portable", "confidence": 0.6}]
    assert tr.update(d, 0)[0]["object_id"] == "O1"
    tr.update([], 1); tr.update([], 2)
    assert tr.update(d, 3)[0]["object_id"] == "O1"
    for f in range(4, 8):
        tr.update([], f)
    assert tr.update(d, 8)[0]["object_id"] == "O2"


def cfg(**kw):
    base = dict(fingertip_threshold=2, pad_frac=0.15, min_contact_frames=3, min_contact_s=None,
                release_grace_frames=3, allow_multiple=False, multi_min_fingertips=2, sticky_margin=1,
                interpolated_can_start=False, min_contact_object_conf=0.0, hand_ids=None,
                type_priority=("portable", "unknown", "fixed"))
    base.update(kw)
    return SimpleNamespace(**base)


def run(frames_spec, **kw):
    """frames_spec: list of (hands, objects) per frame at 30 fps."""
    hands = [{"frame": f, "t": round(f / 30, 4), "hands": h} for f, (h, _) in enumerate(frames_spec)]
    objs = [{"frame": f, "t": round(f / 30, 4), "objects": o} for f, (_, o) in enumerate(frames_spec)]
    return analyze_contacts(hands, objs, cfg(**kw))


TOUCH = [(210, 210), (250, 250), (290, 290)]
AWAY = [(600, 600), (650, 650)]


def test_noise_rejected_and_stable_contact():
    spec = [([hand(1, AWAY)], [obj("O1", BOX)])] * 2
    spec += [([hand(1, TOUCH)], [obj("O1", BOX)])] * 2              # 2-frame blip
    spec += [([hand(1, AWAY)], [obj("O1", BOX)])] * 5
    spec += [([hand(1, TOUCH)], [obj("O1", BOX)])] * 10             # real contact frames 9..18
    spec += [([hand(1, AWAY)], [obj("O1", BOX)])] * 6
    res = run(spec)
    assert res["rejected_candidates"] == 1
    assert len(res["events"]) == 1
    e = res["events"][0]
    assert (e["start_frame"], e["end_frame"], e["end_reason"]) == (9, 18, "released")
    assert res["frames"][2]["hands"][0]["state"] == "candidate"
    assert res["frames"][9]["hands"][0]["state"] == "touching_portable"   # back-filled from first frame
    assert res["frames"][20]["hands"][0]["state"] == "none"              # event ends at last support


def test_dropout_within_grace_keeps_one_event_longer_dropout_splits():
    seg = lambda n, t: [([hand(1, t)], [obj("O1", BOX)])] * n
    one = run(seg(5, TOUCH) + seg(3, AWAY) + seg(5, TOUCH))
    assert len(one["events"]) == 1 and one["events"][0]["end_frame"] == 12
    gap = one["frames"][6]["hands"][0]
    assert gap["state"] == "touching_portable" and gap["contacts"][0]["in_grace"] and gap["uncertain"]
    two = run(seg(5, TOUCH) + seg(4, AWAY) + seg(5, TOUCH))
    assert len(two["events"]) == 2
    assert two["events"][0]["end_reason"] == "released" and two["events"][1]["end_reason"] == "clip_end"


def test_hand_disappearing_counts_as_missing():
    spec = [([hand(1, TOUCH)], [obj("O1", BOX)])] * 5 + [([], [obj("O1", BOX)])] * 5
    res = run(spec)
    assert res["events"][0]["end_frame"] == 4 and res["events"][0]["end_reason"] == "released"


def test_fixed_object_semantics_and_csv_rows():
    spec = [([hand(1, TOUCH)], [obj("O7", BOX, "workbench", "fixed")])] * 6
    res = run(spec)
    rows = event_rows(res["events"], "clipA")
    r = rows[0]
    assert list(r) == CSV_COLUMNS
    assert (r["state"], r["start_event"], r["end_event"], r["hand"]) == ("touching_fixed", "touch_start", "clip_end", "right")
    assert r["duration_s"] == pytest.approx(5 / 30, abs=1e-3) and "ended_by=clip_end" in r["uncertainty"]


def test_unknown_object_stays_unknown():
    res = run([([hand(1, TOUCH)], [obj("O1", BOX, "thing", "unknown")])] * 4)
    r = event_rows(res["events"], "c")[0]
    assert (r["object_type"], r["start_event"]) == ("unknown", "contact_start")


def test_same_hand_switches_objects_without_merging():
    box2 = (500, 500, 600, 600)
    t2 = [(510, 510), (550, 550), (590, 590)]
    spec = [([hand(1, TOUCH)], [obj("O1", BOX), obj("O2", box2, "wrench")])] * 5
    spec += [([hand(1, t2)], [obj("O1", BOX), obj("O2", box2, "wrench")])] * 5
    res = run(spec)
    assert [(e["object_id"], e["start_frame"], e["end_frame"]) for e in res["events"]] == [("O1", 0, 4), ("O2", 5, 9)]
    assert res["events"][0]["end_reason"] == "switched_object"


def test_two_hands_two_objects():
    box2 = (500, 500, 600, 600)
    t2 = [(510, 510), (550, 550)]
    spec = [([hand(1, TOUCH, "left"), hand(2, t2, "right")], [obj("O1", BOX), obj("O2", box2, "box")])] * 6
    res = run(spec)
    pairs = {(e["hand_id"], e["object_id"]) for e in res["events"]}
    assert pairs == {(1, "O1"), (2, "O2")}
    rows = event_rows(res["events"], "c")
    assert {r["hand"] for r in rows} == {"left", "right"}


def test_interpolated_hand_cannot_start_but_continues_with_flag():
    obs = [obj("O1", BOX)]
    only_interp = run([([hand(1, TOUCH, interp=True)], obs)] * 6)
    assert only_interp["events"] == []
    spec = [([hand(1, TOUCH)], obs)] * 3 + [([hand(1, TOUCH, interp=True)], obs)] * 3
    res = run(spec)
    e = res["events"][0]
    assert e["end_frame"] == 5 and e["interpolated_frames"] == 3
    assert "interpolated_hand_frames=3" in event_rows(res["events"], "c")[0]["uncertainty"]
    assert res["frames"][4]["hands"][0]["uncertain"] is True
    allowed = run([([hand(1, TOUCH, interp=True)], obs)] * 6, interpolated_can_start=True)
    assert len(allowed["events"]) == 1


def test_hand_allowlist_reports_other_hands():
    spec = [([hand(1, TOUCH), hand(9, TOUCH)], [obj("O1", BOX)])] * 4
    res = run(spec, hand_ids=[1])
    states = {h["hand_id"]: h["state"] for h in res["frames"][0]["hands"]}
    assert states == {1: "touching_portable", 9: "not_analyzed"}


def test_min_contact_seconds():
    spec = [([hand(1, TOUCH)], [obj("O1", BOX)])] * 5
    assert run(spec, min_contact_s=0.5)["events"] == []
    assert len(run(spec, min_contact_s=0.1)["events"]) == 1


def test_empty_timeline():
    res = run([([hand(1, AWAY)], [obj("O1", BOX)])] * 5 + [([], [])] * 3)
    assert res["events"] == [] and event_rows(res["events"], "c") == []


def test_state_machine_direct_interpolated_neutral_in_streak():
    sm = ContactStateMachine(min_contact_frames=3)
    c = lambda interp: {"object_id": "O1", "object_name": "box", "object_type": "portable", "score": 0.5,
                        "fingertip_count": 2, "object_conf": 0.5, "interpolated": interp}
    sm.step(0, 0.0, {1: [c(False)]}); sm.step(1, 0.03, {1: [c(True)]}); sm.step(2, 0.07, {1: [c(False)]})
    assert sm.events == [] and sm.keys[(1, "O1")].event is None
    sm.step(3, 0.1, {1: [c(False)]})
    assert sm.keys[(1, "O1")].event["start_frame"] == 0


# ----------------------------------------------------------------------------- detector helpers
def test_nms_and_filter():
    keep = nms(np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]]), np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep == [0, 2]
    vocab = ObjectVocabulary(["table", "workbench", "wrench"], {"table": "fixed", "workbench": "fixed", "wrench": "portable"},
                             per_class_conf={"table": 0.5})
    c = PS2Config(object_conf=0.2, min_object_area_frac=0.001)
    dets = [{"bbox": [0, 0, 500, 300], "object_name": "table", "object_type": "fixed", "confidence": 0.4},   # below class thr
            {"bbox": [0, 0, 500, 300], "object_name": "workbench", "object_type": "fixed", "confidence": 0.6},
            {"bbox": [1, 1, 500, 300], "object_name": "workbench", "object_type": "fixed", "confidence": 0.3},  # NMS
            {"bbox": [10, 10, 12, 12], "object_name": "wrench", "object_type": "portable", "confidence": 0.9},  # tiny
            {"bbox": [600, 400, 700, 520], "object_name": "wrench", "object_type": "portable", "confidence": 0.25}]
    out = filter_detections(dets, vocab, c, (1000, 1000))
    assert [(d["object_name"], d["confidence"]) for d in out] == [("workbench", 0.6), ("wrench", 0.25)]


def test_decode_owl_uses_padded_square_of_original_frame():
    vocab = ObjectVocabulary(["wrench", "table"], {"wrench": "portable", "table": "fixed"})
    logits = np.array([[5.0, -5.0], [-5.0, -5.0]])
    boxes = np.array([[0.5, 0.25, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]])
    # 1920x1080 frame padded (bottom) to a 1920x1920 square: y=0.25 -> 480 px, not 270
    out = decode_owl(logits, boxes, ["wrench", "table"], vocab, 1920.0, 0.1)
    assert len(out) == 1 and out[0]["object_name"] == "wrench" and out[0]["object_type"] == "portable"
    assert np.allclose(out[0]["bbox"], [864, 384, 1056, 576])
    # real models return float32 arrays; detections must still be JSON-serializable
    out32 = decode_owl(logits.astype(np.float32), boxes.astype(np.float32), ["wrench", "table"], vocab, 1920.0, 0.1)
    kept = filter_detections([{**d, "confidence": np.float32(d["confidence"])} for d in out32], vocab,
                             PS2Config(object_conf=0.1), (1920, 1080))
    json.dumps(kept)
    assert all(type(v) is float for v in kept[0]["bbox"]) and type(kept[0]["confidence"]) is float


# ============================================================================= from test_ps2_upgrades.py
# ----------------------------------------------------------------------------- profiles
def test_profiles_resolve_from_clip_name():
    v = ObjectVocabulary.load(clip="mytronlabs-000060-008")
    assert v.profile == "granite_factory"
    assert v.object_type("pendant_controller") == "portable" and v.object_type("conveyor") == "fixed"
    assert v.query_text("pendant_controller") == "a photo of a yellow crane remote control"
    assert ("cloth", "granite_slab") in v.co_contact
    assert ObjectVocabulary.load(clip="mytronlabs-000003-004").profile == "repair_bench"
    assert ObjectVocabulary.load(clip="MTL-SKL00089-013").profile == "electrical_shop"
    assert ObjectVocabulary.load(clip="MTL-SGF00106-027").profile == "generic"         # not reviewed yet
    assert ObjectVocabulary.load(profile="repair_bench", clip="MTL-SKL00089-013").profile == "repair_bench"
    with pytest.raises(ValueError):
        ObjectVocabulary.load(profile="nope")


def test_every_profile_is_consistent():
    d = json.loads(open(PS2Config().prompts_path, encoding="utf-8").read())
    for name in d["profiles"]:
        v = ObjectVocabulary.load(profile=name)
        assert len(v.prompts) == len(set(v.prompts)) >= 5
        assert set(v.type_of.values()) <= {"portable", "fixed", "unknown"}
        for a, b in v.co_contact:
            assert a in v.type_of and b in v.type_of, (name, a, b)
        assert "factory_floor" not in v.type_of           # a floor box would make every hand "touching fixed"


def test_flat_legacy_file_still_loads(tmp_path):
    p = tmp_path / "flat.json"
    p.write_text(json.dumps({"portable": ["wrench"], "fixed": ["table"]}))
    v = ObjectVocabulary.load(p, clip="anything")
    assert v.profile == "" and v.prompts == ["wrench", "table"] and v.query_text("wrench") == "wrench"


def test_detection_fingerprint_depends_on_resolved_profile():
    c = PS2Config()
    assert c.detection_fingerprint("repair_bench") != c.detection_fingerprint("granite_factory")


# ----------------------------------------------------------------------------- masks
def test_mask_grid_roundtrip_and_near_mask():
    mask = np.zeros((200, 200), bool)
    mask[100:110, 0:200] = True                         # thin horizontal strip (an LED profile)
    bbox = [0, 100, 200, 110]
    g = decode_grid(mask_to_grid(mask, bbox, 32), 32)
    assert g.all()                                      # strip fills its own box
    diag = np.eye(32, dtype=bool)
    assert (decode_grid(encode_grid(diag), 32) == diag).all()
    box = [0, 0, 320, 320]                              # 10 px cells
    assert point_near_mask(diag, box, (155, 155), 0)            # on the diagonal
    assert not point_near_mask(diag, box, (300, 20), 15)        # far corner of the box, off the mask
    assert point_near_mask(diag, box, (172, 155), 15)           # 12 px off the diagonal, within pad


def test_mask_rejects_fingertips_in_empty_part_of_box():
    # diagonal object: box is 300x300 but the mask is only the diagonal band
    diag = np.zeros((32, 32), bool)
    for i in range(32):
        diag[i, max(0, i - 1): i + 2] = True
    o = {**obj("O1", (200, 200, 500, 500), "led_strip"), "mask_grid": encode_grid(diag)}
    off = hand(1, [(480, 220), (470, 230)], bbox=(400, 150, 500, 250))      # top-right corner: in box, off mask
    on = hand(1, [(300, 300), (350, 350)], bbox=(280, 280, 380, 380))       # on the diagonal
    assert frame_candidates(off, [o], use_masks=False)                     # box rule alone says contact
    assert frame_candidates(off, [o]) == []                                # mask says no
    c = frame_candidates(on, [o])
    assert c and c[0]["mask_checked"] is True


# ----------------------------------------------------------------------------- grasp + co-contact
def test_grasp_types():
    assert grasp_type([4, 8]) == "pinch"
    assert grasp_type([4, 8, 12, 16]) == "power"
    assert grasp_type([8, 12, 16]) == "hook"
    assert grasp_type([4, 12]) == "partial"


def test_cloth_and_slab_co_contact_kept_together():
    tips = [(210, 210), (250, 250), (290, 290)]
    cloth = obj("O1", (200, 200, 300, 300), "cloth", "portable", 0.6)
    slab = obj("O2", (100, 100, 900, 900), "granite_slab", "portable", 0.8)
    spec = [([hand(1, tips)], [cloth, slab])] * 6
    without = run(spec)
    assert len(without["events"]) == 1                    # one object per hand without the pair rule
    hands = [{"frame": f, "t": f / 30, "hands": h} for f, (h, _) in enumerate(spec)]
    objs = [{"frame": f, "t": f / 30, "objects": o} for f, (_, o) in enumerate(spec)]
    cfg = SimpleNamespace(fingertip_threshold=2, pad_frac=0.15, min_contact_frames=3, min_contact_s=None,
                          release_grace_frames=3, allow_multiple=False, multi_min_fingertips=2, sticky_margin=1,
                          interpolated_can_start=False, min_contact_object_conf=0.0, hand_ids=None,
                          type_priority=("portable", "unknown", "fixed"), co_contact=True)
    res = analyze_contacts(hands, objs, cfg, [("cloth", "granite_slab")])
    assert sorted(e["object_id"] for e in res["events"]) == ["O1", "O2"]
    rows = event_rows(res["events"], "c")
    assert any(r["co_contact"] for r in rows)
    assert len(res["frames"][4]["hands"][0]["contacts"]) == 2


def test_csv_has_grasp_column():
    res = run([([hand(1, [(210, 210), (250, 250)])], [obj("O1", (200, 200, 300, 300))])] * 4)
    r = event_rows(res["events"], "c")[0]
    assert r["grasp"] == "pinch" and r["co_contact"] is False


# ----------------------------------------------------------------------------- fine-tuned YOLO names
def test_yolo_accepts_vocabulary_class_names():
    v = ObjectVocabulary.load(profile="granite_factory")
    y = YoloDetector.__new__(YoloDetector)
    y.vocab, y.cfg, y.names = v, PS2Config(yolo_model="best.pt"), {0: "pendant_controller", 1: "conveyor",
                                                                    2: "dining table", 3: "person"}
    assert y._type("pendant_controller") == "portable" and y._type("conveyor") == "fixed"
    assert y._type("dining table") == "fixed" and y._type("person") is None
    d = y.describe()
    assert d["ignored_classes"] == ["person"]


# ============================================================================= from test_ps2_pipeline.py

FPS = 10.0


class FakeObjDetector:
    name = "fake"
    last_time_s = 0.0

    def __init__(self):
        self.frames_seen = 0

    def describe(self):
        return {"backend": "fake"}

    def __call__(self, frames):
        self.frames_seen += len(frames)
        return [[{"bbox": list(BOX), "object_name": "screwdriver", "object_type": "portable", "confidence": 0.7},
                 {"bbox": [0, 400, 640, 480], "object_name": "workbench", "object_type": "fixed", "confidence": 0.5}]
                for _ in frames]


def write_hands(path: Path, n: int, touch_frames: range):
    with open(path, "w") as f:
        for i in range(n):
            tips = TOUCH if i in touch_frames else AWAY
            f.write(json.dumps({"frame": i, "t": round(i / FPS, 4), "hands": [hand(1, tips)]}) + "\n")


@pytest.fixture
def clip(tmp_path):
    d = tmp_path / "out" / "clipA"
    d.mkdir(parents=True)
    video = make_video(tmp_path / "clipA.mp4", n=30, fps=FPS, size=(640, 480))
    write_hands(d / "hands.jsonl", 30, range(5, 15))
    return video, d


def test_run_ps2_outputs_and_cache(clip):
    video, d = clip
    cfg = PS2Config(video_path=str(video), hands_jsonl=str(d / "hands.jsonl"), det_batch=4)
    det = FakeObjDetector()
    run_ps2(cfg, detector=det, log=lambda *a: None)
    ps2 = d / "ps2"
    assert det.frames_seen == 30
    for name in ("object_detections.jsonl", "object_detections_meta.json", "contacts.jsonl", "contacts.csv",
                 "ps2_metrics.json", "annotated_ps2.mp4", "ps2_config.json"):
        assert (ps2 / name).exists(), name
    assert not list(ps2.glob("*.partial"))
    objs = [json.loads(l) for l in (ps2 / "object_detections.jsonl").read_text().splitlines()]
    assert [o["frame"] for o in objs] == list(range(30))
    assert {o["object_id"] for o in objs[0]["objects"]} == {"O1", "O2"}
    assert all({o["object_id"] for o in r["objects"]} == {"O1", "O2"} for r in objs)
    rows = list(csv.DictReader(open(ps2 / "contacts.csv")))
    assert len(rows) == 1
    r = rows[0]
    assert (r["clip"], r["hand"], r["state"], r["object_name"], r["start_frame"], r["end_frame"]) == \
        ("clipA", "right", "touching_portable", "screwdriver", "5", "14")
    assert (r["start_event"], r["end_event"]) == ("grab", "release")
    assert float(r["t_start"]) == 0.5 and float(r["t_end"]) == 1.4
    cj = [json.loads(l) for l in (ps2 / "contacts.jsonl").read_text().splitlines()]
    assert len(cj) == 30 and cj[0]["hands"][0]["state"] == "none" and cj[7]["hands"][0]["state"] == "touching_portable"
    m = json.loads((ps2 / "ps2_metrics.json").read_text())
    assert m["portable_contacts"] == 1 and m["grab_release_intervals"] == 1 and "2D" in m["disclaimer"]
    assert m["total_frames"] == 30 and m["object_detections"] == 60 and "total_ps2_s" in m["timing"]
    # second run: cached detections, no detector needed
    res2 = run_ps2(cfg, detector=None, log=lambda *a: None)
    assert res2["metrics"]["timing"]["detections_cached"] is True
    # contact-only rerun with stricter duration drops the 10-frame event
    res3 = rerun_contacts(cfg, log=lambda *a: None, min_contact_frames=11)
    assert res3["rows"] == []
    assert list(csv.DictReader(open(ps2 / "contacts.csv"))) == []                 # header only


def test_det_every_carries_boxes_and_flags_them(clip):
    video, d = clip
    cfg = PS2Config(video_path=str(video), hands_jsonl=str(d / "hands.jsonl"), det_every=3, write_video=False,
                    output_dir=str(d / "ps2_skip"))
    det = FakeObjDetector()
    res = run_ps2(cfg, detector=det, log=lambda *a: None)
    assert det.frames_seen == 10
    assert res["metrics"]["skipped_or_invalid"]["frames_with_carried_objects"] == 20
    assert "carried_object_frames" in res["events"][0]["uncertainty"]


def test_alignment_errors(tmp_path, clip):
    video, d = clip
    bad = [{"frame": 0, "t": 0.0, "hands": []}, {"frame": 2, "t": 0.2, "hands": []}]
    with pytest.raises(AlignmentError, match="consecutive"):
        check_hand_records(bad, FPS)
    with pytest.raises(AlignmentError, match="different video"):
        check_hand_records([{"frame": 0, "t": 0.0, "hands": []}, {"frame": 1, "t": 0.5, "hands": []}], FPS)
    long = tmp_path / "long.jsonl"
    write_hands(long, 40, range(0))
    cfg = PS2Config(video_path=str(video), hands_jsonl=str(long), write_video=False, output_dir=str(tmp_path / "x"))
    with pytest.raises(AlignmentError):
        run_ps2(cfg, detector=FakeObjDetector(), log=lambda *a: None)


def test_max_frames_300_style_subset(clip):
    video, d = clip
    cfg = PS2Config(video_path=str(video), hands_jsonl=str(d / "hands.jsonl"), max_frames=12, write_video=False,
                    output_dir=str(d / "ps2_12"))
    res = run_ps2(cfg, detector=FakeObjDetector(), log=lambda *a: None)
    assert res["metrics"]["total_frames"] == 12
    assert res["events"][0]["end_event"] == "clip_end"


def test_contacts_only_rerun_and_eval_tools(clip, tmp_path):
    video, d = clip
    cfg = PS2Config(video_path=str(video), hands_jsonl=str(d / "hands.jsonl"), write_video=False)
    run_ps2(cfg, detector=FakeObjDetector(), log=lambda *a: None)
    res = rerun_contacts(cfg, log=lambda *a: None, fingertip_threshold=3)       # cached detections, no detector
    assert res["metrics"]["timing"]["detections_cached"] is True
    sheet = tmp_path / "sheet.csv"
    assert contact_metrics.main(["template", "--contacts", str(d / "ps2" / "contacts.jsonl"), "--n", "10",
                                 "--out", str(sheet)]) == 0
    rows = list(csv.DictReader(open(sheet)))
    assert len(rows) == 10 and sum(r["pred_state"].startswith("touching") for r in rows) == 5
    for r in rows:
        r["gt_state"] = "portable" if r["pred_state"].startswith("touching") else "none"
        r["gt_object_name"] = "screwdriver" if r["gt_state"] == "portable" else ""
    s = contact_metrics.score_sheet(rows)
    assert (s["contact_precision"], s["contact_recall"], s["type_accuracy_on_tp"], s["labelled_hand_frames"]) == (1.0, 1.0, 1.0, 10)
    pred = list(csv.DictReader(open(d / "ps2" / "contacts.csv")))
    gt = [{"hand_id": "1", "t_start": "0.6", "t_end": "1.4"}]
    t = contact_metrics.timing_error(pred, gt)
    assert t["matched"] == 1 and t["mean_abs_start_error_s"] == pytest.approx(0.1)


# ============================================================================= from test_ps2_labelling.py


def test_event_book_open_close_cycle_undo_save(tmp_path):
    b = label_events.EventBook("c", 30.0, ["cloth", "chain", "other_object"], {"cloth": "portable"})
    assert "opened" in b.toggle("right", 30)
    b.cycle_object(1)
    assert "ignored" in b.toggle("right", 10)                            # end before start
    assert "closed" in b.toggle("right", 60)
    b.toggle("left", 90); b.toggle("left", 120)
    assert [(e["hand"], e["object_name"], e["t_start"], e["t_end"]) for e in b.events] == \
        [("right", "chain", 1.0, 2.0), ("left", "cloth", 3.0, 4.0)]
    b.undo()
    p = tmp_path / "c_events.csv"
    b.save(p)
    b2 = label_events.EventBook.load(p, "c", 30.0, ["cloth"], {})
    assert len(b2.events) == 1 and b2.events[0]["start_frame"] == 30


def test_timing_matches_by_side_and_reports_tolerance():
    pred = [{"hand_id": "3", "hand": "right", "t_start": "1.2", "t_end": "2.9", "object_name": "chain"}]
    gt = [{"hand": "right", "t_start": "1.0", "t_end": "2.0", "object_name": "chain"}]
    t = contact_metrics.timing_error(pred, gt)
    assert t["matched"] == 1 and t["start_within_0.5s"] == 1.0 and t["end_within_0.5s"] == 0.0
    assert t["object_name_accuracy"] == 1.0
    assert contact_metrics.timing_error(pred, [{**gt[0], "hand": "left"}])["matched"] == 0
