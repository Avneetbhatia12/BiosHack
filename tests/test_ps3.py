"""PS3: reply parsing, validation, time-base handling, windows, merging, PS2 refinement, step comparison.

No API call is made: these test the pure logic around the Gemini client with synthetic replies.
"""
import pytest

from handtrace.ps3.compare import check_sequence, compare_clips, step_list
from handtrace.ps3.config import PS3Config, VERBS
from handtrace.ps3.merge import can_merge, merge_actions, same_object
from handtrace.ps3.prompt import build_prompt
from handtrace.ps3.refine import refine_actions
from handtrace.ps3.validate import (normalise_hand, normalise_object, normalise_verb, parse_model_json,
                                    resolve_time_base, to_seconds, validate_actions)
from handtrace.ps3.windows import make_windows


def act(s, e, verb="fasten", obj="screw", hand="right", conf=0.8, **kw):
    return {"start_s": s, "end_s": e, "verb": verb, "object": obj, "hand": hand, "confidence": conf, **kw}


# ----------------------------------------------------------------------------- parsing and normalisation
def test_parse_model_json_strict_fenced_and_truncated():
    assert parse_model_json('{"actions": [{"verb": "pick"}]}') == [{"verb": "pick"}]
    assert parse_model_json('```json\n[{"verb": "place"}]\n```') == [{"verb": "place"}]
    cut = '{"actions": [{"verb": "pick", "start_s": 1}, {"verb": "place", "start_s": 2}, {"verb": "fas'
    assert [d["verb"] for d in parse_model_json(cut)] == ["pick", "place"]          # complete objects salvaged
    assert parse_model_json('{"actions": []}') == []
    with pytest.raises(ValueError):
        parse_model_json("")
    with pytest.raises(ValueError):
        parse_model_json("no json here")


@pytest.mark.parametrize("value,expected", [(12, 12.0), (12.5, 12.5), ("12.5s", 12.5), ("00:12", 12.0),
                                            ("1:02:03", 3723.0), ("abc", None), (True, None)])
def test_to_seconds(value, expected):
    assert to_seconds(value) == expected


def test_verb_hand_object_normalisation():
    assert normalise_verb("fasten") == ("fasten", False)
    assert normalise_verb("unscrew") == ("loosen", False)                 # synonym
    assert normalise_verb("fastening") == ("fasten", False)               # inflection
    assert normalise_verb("pick up") == ("pick", False)
    assert normalise_verb("juggle") == ("other", True)                    # unmapped is flagged, not invented
    assert set(VERBS) >= {normalise_verb(v)[0] for v in ("grab", "put down", "rotate", "clean", "check")}
    assert (normalise_hand("Left"), normalise_hand("both hands"), normalise_hand("R"), normalise_hand("?")) == \
        ("left", "both", "right", "unknown")
    assert normalise_object("  Small   Screw ") == "small screw" and normalise_object("none") == "unknown"


# ----------------------------------------------------------------------------- time base and validation
def test_resolve_time_base():
    assert resolve_time_base([(5, 8), (10, 12)], 0.0, "auto") == (0.0, "absolute")
    assert resolve_time_base([(5, 8), (10, 12)], 50.0, "auto") == (50.0, "relative(auto)")   # before the window
    assert resolve_time_base([(52, 55), (60, 62)], 50.0, "auto") == (0.0, "absolute(auto)")
    assert resolve_time_base([(5, 8)], 50.0, "relative") == (50.0, "relative")
    assert resolve_time_base([(5, 8)], 50.0, "absolute") == (0.0, "absolute")


def test_validate_actions_relative_window_and_drops():
    cfg = PS3Config()
    raw = [{"start_s": 2, "end_s": 6, "verb": "screwing", "object": "Screw", "hand": "R", "confidence": 1.4},
           {"start_s": "0:07", "end_s": "0:09", "verb": "inspect", "object": "", "hand": "left"},
           {"start_s": 8, "end_s": 8.2, "verb": "pick", "object": "nut", "hand": "left"},          # too short
           {"start_s": 9, "end_s": 7, "verb": "pick", "object": "nut", "hand": "left"},            # end <= start
           {"start_s": "x", "end_s": 3, "verb": "pick"}]                                           # unreadable
    clean, dropped, info = validate_actions(raw, 50.0, 110.0, 180.0, cfg, window_index=1)
    assert info["time_base"] == "relative(auto)" and info["offset_added_s"] == 50.0
    assert [(a["start_s"], a["end_s"], a["verb"], a["hand"]) for a in clean] == [(52.0, 56.0, "fasten", "right"),
                                                                                (57.0, 59.0, "inspect", "left")]
    assert clean[0]["confidence"] == 1.0 and clean[1]["object"] == "unknown" and clean[0]["windows"] == [1]
    assert sorted(d["reason"] for d in dropped) == ["end <= start", "too short", "unreadable timestamps"]


def test_validate_drops_actions_outside_window_and_clamps_to_video():
    clean, dropped, _ = validate_actions([act(0, 5), act(178, 185)], 120.0, 180.0, 180.0,
                                         PS3Config(time_base="absolute"))
    assert [d["reason"] for d in dropped] == ["outside window"]
    assert clean[0]["end_s"] == 180.0                                                          # clamped


# ----------------------------------------------------------------------------- windows and merging
def test_make_windows():
    assert make_windows(0, 60, 10) == []
    assert make_windows(70, 60, 10) == [(0.0, 70.0)]                      # short clip: one window
    w = make_windows(180, 60, 10)
    assert w[0] == (0.0, 60.0) and w[1][0] == 50.0 and w[-1][1] == 180.0
    assert all(b[0] < a[1] for a, b in zip(w, w[1:]))                     # consecutive windows overlap


def test_merge_overlapping_window_duplicates():
    cfg = PS3Config()
    merged = merge_actions([act(10, 14, conf=0.6, windows=[0]), act(13.5, 18, obj="unknown", conf=0.9, windows=[1]),
                            act(30, 32, verb="pick", obj="screwdriver")], cfg)
    assert len(merged) == 2
    m = merged[0]
    assert (m["start_s"], m["end_s"], m["object"], m["confidence"], m["windows"]) == (10, 18, "screw", 0.9, [0, 1])


def test_merge_rules():
    assert same_object("screw", "unknown") and same_object("small screw", "screw") and not same_object("screw", "screwdriver")
    assert not can_merge(act(0, 5, hand="left"), act(1, 6, hand="right"), 1.0)          # two hands acting at once
    assert not can_merge(act(0, 5), act(8, 9), 1.0)                                    # gap too large
    assert can_merge(act(0, 5, hand="unknown"), act(5.5, 9), 1.0)


# ----------------------------------------------------------------------------- PS2 refinement
def test_refine_snaps_edges_to_ps2_grab_release_and_fills_unknowns():
    contacts = [{"t_start": 10.0, "t_end": 14.0, "hand": "right", "object_name": "screwdriver", "object_type": "portable"},
                {"t_start": 40.0, "t_end": 45.0, "hand": "left", "object_name": "cloth", "object_type": "portable"}]
    actions = [act(10.6, 13.3, verb="fasten", obj="screwdriver", hand="unknown"),
               act(20, 22, verb="inspect", obj="panel", hand="left")]
    out, rep = refine_actions(actions, contacts, PS3Config())
    a = out[0]
    assert (a["start_s"], a["end_s"], a["hand"]) == (10.0, 14.0, "right") and a["refined"]
    assert rep["edges_snapped"] == 2 and rep["hands_filled"] == 1
    assert out[1]["start_s"] == 20 and "refined" not in out[1]                        # no compatible contact: unchanged
    assert [c["object"] for c in rep["portable_contacts_without_action"]] == ["cloth"]


def test_refine_does_not_snap_beyond_tolerance():
    contacts = [{"t_start": 10.0, "t_end": 14.0, "hand": "right", "object_name": "screw", "object_type": "portable"}]
    out, rep = refine_actions([act(5.0, 20.0, obj="screw")], contacts, PS3Config(snap_tol_s=1.5))
    assert (out[0]["start_s"], out[0]["end_s"]) == (5.0, 20.0) and rep["edges_snapped"] == 0


# ----------------------------------------------------------------------------- comparison and prompt
def test_check_sequence_missing_and_out_of_order():
    actions = [act(0, 2, verb="pick", obj="screwdriver"), act(5, 8, verb="place", obj="screwdriver"),
               act(3, 4, verb="fasten", obj="screw")]
    rep = check_sequence(actions, ["pick screwdriver", "place screwdriver", "fasten screw", "inspect panel"])
    assert rep["missing"] == ["inspect panel"]
    assert [x["expected_step"] for x in rep["out_of_order"]] == ["place screwdriver"]
    assert rep["complete_and_in_order"] is False
    ok = check_sequence(actions, ["pick screwdriver", "fasten screw", "place screwdriver"])
    assert ok["complete_and_in_order"] is True


def test_compare_clips_and_step_list():
    a = [act(0, 2, verb="pick", obj="screwdriver"), act(2, 6, obj="screw"), act(6, 9, obj="screw")]
    b = [act(0, 1, verb="pick", obj="screwdriver"), act(1, 3, obj="screw")]
    assert [s["count"] for s in step_list(a)] == [1, 2]                               # repeats collapsed
    rows, summ = compare_clips(a, b)
    assert summ["matched_steps"] == 2 and summ["faster"] == "B" and summ["sequence_similarity"] == 1.0


def test_prompt_mentions_vocabulary_window_and_hints():
    p = build_prompt(50.0, 110.0, 180.0, object_hints=["screwdriver", "cloth"])
    assert all(v in p for v in VERBS) and "screwdriver" in p and "NOT mirrored" in p
