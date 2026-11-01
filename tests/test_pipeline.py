"""Pipeline plumbing: CLI parsing and config files, runner clip discovery / ordering / settings / markers,
result loaders, and the environment helpers. No GPU, model or private data needed."""
import json
import os

import pytest

from handtrace import cli, environment, results as R, runner


def test_cli_stage_subcommands_and_config_file(tmp_path):
    a = cli.parse_args(["ps2", "--clips", "data/clips"])
    assert a.stages == "ps2"
    assert cli.parse_args(["all", "--clips", "x"]).stages == "ps1,ps2,ps3,ps5"
    assert cli.parse_args(["--clips", "x", "--stages", "ps1,ps3"]).stages == "ps1,ps3"     # legacy form
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"_comment": "x", "clips": ["data/clips"], "obj_backend": "yolo", "det_imgsz": 1280,
                               "stages": ["ps1", "ps2"]}))
    a = cli.parse_args(["--config", str(cfg)])
    assert a.clips == ["data/clips"] and a.obj_backend == "yolo" and a.det_imgsz == 1280 and a.stages == "ps1,ps2"
    assert cli.parse_args(["--config", str(cfg), "--det-imgsz", "640"]).det_imgsz == 640     # flag wins
    cfg.write_text(json.dumps({"clips": ["x"], "not_an_option": 1}))
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", str(cfg)])
    with pytest.raises(SystemExit):
        cli.parse_args(["ps1"])                                                               # no clips anywhere


def test_example_config_is_valid():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = cli.parse_args(["--config", os.path.join(root, "configs", "example_config.json")])
    assert a.stages == "ps1,ps2,ps3,ps5" and a.mode in runner.MODES


def test_cli_reports_missing_videos(tmp_path):
    assert cli.main(["ps5", "--clips", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_discover_keeps_largest_duplicate_and_orders_demo_first(tmp_path):
    (tmp_path / "full").mkdir(); (tmp_path / "small").mkdir()
    (tmp_path / "full" / "b.mp4").write_bytes(b"x" * 100)
    (tmp_path / "small" / "b.mp4").write_bytes(b"x" * 10)
    (tmp_path / "full" / "a.mp4").write_bytes(b"x")
    (tmp_path / "full" / "notes.txt").write_text("not a video")
    clips = runner.discover_clips([str(tmp_path)])
    assert [os.path.basename(c) for c in clips] == ["a.mp4", "b.mp4"]
    assert os.path.getsize(clips[1]) == 100
    assert [os.path.basename(c) for c in runner.order_clips(clips, ["b"])] == ["b.mp4", "a.mp4"]
    assert runner.stem_of("/x/y/clip-01.mp4") == "clip-01"


def test_settings_roundtrip_and_validation(tmp_path):
    s = runner.Settings(clips=["a.mp4"], out_root=str(tmp_path), mode="fp16", ps2={"backend": "yolo"})
    p = tmp_path / "settings.json"
    s.save(str(p))
    s2 = runner.Settings.load(str(p))
    assert s2 == s
    cfg = runner.ps1_config(s2, "a.mp4")
    assert cfg.fp16 is True and cfg.fast is False and cfg.write_video is False
    with pytest.raises(ValueError):
        runner.ps1_config(runner.Settings(mode="turbo"), "a.mp4")


def test_stage_markers(tmp_path):
    d = tmp_path / "clipA"
    d.mkdir()
    assert not runner.ps1_done(str(tmp_path), "clipA")
    (d / "hands_final.jsonl").write_text("")
    (d / "postprocess_report.json").write_text("{}")
    assert runner.ps1_done(str(tmp_path), "clipA") and not runner.ps2_done(str(tmp_path), "clipA")


def test_result_loaders_never_raise_on_missing_files(tmp_path):
    assert R.load_hands(str(tmp_path), "nope") == []
    assert R.load_contacts(str(tmp_path), "nope") == []
    assert R.load_actions(str(tmp_path), "nope") == []
    p = R.paths(str(tmp_path), "clip")
    assert p["contacts"].endswith(os.path.join("ps2_final", "contacts.csv"))
    assert p["actions_final"].endswith(os.path.join("ps3", "actions_final.json"))


def test_secret_lookup_reads_environment_without_printing(monkeypatch, capsys):
    monkeypatch.setenv("HANDTRACE_TEST_SECRET", "s3cr3t")
    assert environment.get_secret("HANDTRACE_TEST_SECRET") == "s3cr3t"
    assert environment.load_secret("HANDTRACE_TEST_SECRET") is True
    assert "s3cr3t" not in capsys.readouterr().out
    monkeypatch.delenv("HANDTRACE_TEST_SECRET")
    assert environment.get_secret("HANDTRACE_TEST_SECRET") is None


def test_constraints_pin_installed_protected_packages(tmp_path):
    path = environment.write_constraints(str(tmp_path / "c.txt"))
    pins = open(path).read().split()
    assert any(p.startswith("numpy==") for p in pins)
    assert all(p.split("==")[0] in environment.PROTECTED for p in pins)
