"""PS5: event index built from synthetic PS1/PS2/PS3/PS5 files, plain-English query parsing and search."""
import csv
import json
import os

from handtrace import results as R
from handtrace.ps5.index import build_index, known_objects, ps1_events, ps2_events, ps3_events
from handtrace.ps5.search import merge_moments, parse_query, search


def _write_clip(out_root, stem):
    p = R.paths(str(out_root), stem)
    os.makedirs(os.path.dirname(p["contacts"]), exist_ok=True)
    os.makedirs(os.path.dirname(p["actions_final"]), exist_ok=True)
    with open(p["hands_final"], "w") as f:                      # right hand 0-3 s, both hands 1-2 s, none 3-5 s
        for k in range(51):
            t = k / 10
            hands = []
            if t <= 3.0:
                hands.append({"id": 1, "side": "right"})
            if 1.0 <= t <= 2.0:
                hands.append({"id": 2, "side": "left"})
            f.write(json.dumps({"frame": k, "t": t, "hands": hands}) + "\n")
    with open(p["contacts"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t_start", "t_end", "hand", "state", "object_name", "object_type"])
        w.writeheader()
        w.writerow({"clip": stem, "t_start": 0.5, "t_end": 2.5, "hand": "right", "state": "touching_portable",
                    "object_name": "screwdriver", "object_type": "portable"})
    with open(p["actions_final"], "w") as f:
        json.dump({"actions": [{"start_s": 0.6, "end_s": 2.4, "verb": "fasten", "object": "screw", "hand": "right",
                                "confidence": 0.8},
                               {"start_s": 3.0, "end_s": 4.0, "verb": "pick", "object": "screwdriver", "hand": "left",
                                "confidence": 0.7}]}, f)


def test_ps1_ps2_ps3_events():
    hands = [{"t": k / 10, "hands": [{"side": "right"}] if k <= 30 else []} for k in range(51)]
    ev = ps1_events("c", hands)
    kinds = {(e["kind"], e["hand"]) for e in ev}
    assert ("hands_visible", "right") in kinds and ("no_hands", "") in kinds
    assert ps2_events("c", [{"t_start": "1", "t_end": "2", "hand": "left", "object_name": "cloth",
                             "object_type": "portable"}])[0]["duration_s"] == 1.0
    a = ps3_events("c", [{"start_s": 1, "end_s": 3, "verb": "wipe", "object": "slab", "hand": "both", "refined": ["x"]}])
    assert a[0]["verb"] == "wipe" and a[0]["detail"] == "x"


def test_build_index_reads_every_stage_and_writes_csv(tmp_path):
    _write_clip(tmp_path, "clipA")
    ev = build_index(str(tmp_path), ["clipA"])
    sources = {e["source"] for e in ev}
    assert sources == {"ps1", "ps2", "ps3"}
    assert (tmp_path / "ps5_events_all.csv").exists()
    assert known_objects(ev) == ["screw", "screwdriver"]
    assert build_index(str(tmp_path), ["missing_clip"], write=False) == []        # absent stages: empty, no crash


def test_parse_query_hands_objects_verbs_durations_and_times():
    f = parse_query("show every time someone uses a screwdriver with the right hand", objects=["screwdriver"])
    assert f["objects"] == {"screwdriver"} and f["hands"] == {"right"}
    f = parse_query("left hand fasten screws longer than 5 s after 1:30", objects=["screw"])
    assert f["verbs"] == {"fasten"} and f["objects"] == {"screw"} and f["hands"] == {"left"}
    assert f["min_dur"] == 5.0 and f["after"] == 90.0
    f = parse_query("pick between 1:00 and 2:30")
    assert (f["after"], f["before"]) == (60.0, 150.0) and parse_query("pick before 0:45")["before"] == 45.0
    assert parse_query("no hands")["kinds"] == {"no_hands"}
    assert parse_query("both hands")["hands"] == {"both"}
    assert "safety" in parse_query("unsafe moments")["kinds"]


def test_search_end_to_end(tmp_path):
    _write_clip(tmp_path, "clipA")
    ev = build_index(str(tmp_path), ["clipA"], write=False)
    r = search(ev, "show every time someone uses a screwdriver with the right hand")
    assert r["filters"]["objects"] == ["screwdriver"] and r["filters"]["hands"] == ["right"]
    assert r["events"] and all(e["object"] == "screwdriver" and e["hand"] == "right" for e in r["events"])
    assert not search(ev, "left hand fasten screw")["events"]                   # the fasten action was right-handed
    assert search(ev, "no hands")["events"][0]["kind"] == "no_hands"


def test_merge_moments_joins_close_events():
    ev = [{"clip": "c", "t_start": 0.0, "t_end": 1.0, "kind": "action", "source": "ps3", "hand": "right", "verb": "fasten",
           "object": "screw", "object_type": "", "confidence": 0.8, "detail": "", "duration_s": 1.0},
          {"clip": "c", "t_start": 1.2, "t_end": 2.0, "kind": "contact", "source": "ps2", "hand": "right", "verb": "touch",
           "object": "screw", "object_type": "portable", "confidence": None, "detail": "", "duration_s": 0.8}]
    m = merge_moments(ev, gap_s=0.5)
    assert len(m) == 1 and (m[0]["t_start"], m[0]["t_end"]) == (0.0, 2.0)
