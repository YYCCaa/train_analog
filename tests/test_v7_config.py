"""Tests for the v7 YAML config layer (v7_config.py)."""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v7_config as vc  # noqa: E402

RANGE_TYPES = ("int_range", "float_range")


def _leaves(d):
    """Flatten a nested section->key dict into {"section.key": leaf}."""
    return {f"{s}.{k}": v for s, sec in d.items() for k, v in sec.items()}


def test_default_matches_schema():
    cfg = vc.default_config()
    schema = vc.load_schema()
    vc.validate_config(cfg, schema)
    assert set(_leaves(cfg)) == set(_leaves(schema))


def test_named_keys_exist():
    cfg = vc.default_config()
    assert cfg["canvas"]["img_size"] == 256
    assert vc.load_schema()["canvas"]["img_size"]["readonly"] is True
    assert cfg["dataset"]["angle_deg_range"] == [0, 359]
    assert cfg["ticks"]["len_range"] == [6, 9]
    assert cfg["ticks"]["per_digit_range"] == [1, 5]
    assert cfg["digits"]["show_prob"] == 0.5
    assert cfg["pointer"]["gray_range"] == [0, 255]
    assert cfg["shadows"]["prob"] == 0.5
    assert cfg["tri_artifacts"]["enabled"] is True
    assert cfg["tri_artifacts"]["count_range"] == [0, 6]
    assert cfg["geometry"]["scale_range"] == [0.9, 1.1]


def test_every_range_default_inside_bounds():
    cfg = vc.default_config()
    schema = vc.load_schema()
    n = 0
    for name, leaf in _leaves(schema).items():
        if leaf["type"] not in RANGE_TYPES:
            continue
        sec, key = name.split(".")
        lo, hi = cfg[sec][key]
        assert leaf["min"] <= lo <= hi <= leaf["max"], name
        n += 1
    assert n > 20


def test_default_config_is_fresh_copy():
    a = vc.default_config()
    a["ticks"]["len_range"][0] = 99
    assert vc.default_config()["ticks"]["len_range"] == [6, 9]


def test_partial_override_merges():
    cfg = vc.parse_config_text("ticks:\n  len_range: [10, 12]\n")
    assert cfg["ticks"]["len_range"] == [10, 12]
    base = vc.default_config()
    base["ticks"]["len_range"] = [10, 12]
    assert cfg == base


def test_merge_does_not_mutate_inputs():
    base = vc.default_config()
    over = {"ticks": {"len_range": [7, 8]}}
    snap_b, snap_o = yaml.safe_dump(base), yaml.safe_dump(over)
    out = vc.merge_config(base, over)
    assert out["ticks"]["len_range"] == [7, 8]
    assert yaml.safe_dump(base) == snap_b and yaml.safe_dump(over) == snap_o


@pytest.mark.parametrize("text,key", [
    ("ticks:\n  len_range: [9, 6]\n", "ticks.len_range"),
    ("jitter:\n  saturation_range: [0.7, 1.2]\n", "jitter.saturation_range"),
    ("digits:\n  show_prob: yes-please\n", "digits.show_prob"),
    ("ticks:\n  len_range: [6.5, 9]\n", "ticks.len_range"),
    ("ticks: [\n", "yaml"),
    ("ticks:\n  len_range: [6, 999]\n", "ticks.len_range"),
    ("digits:\n  show_prob: true\n", "digits.show_prob"),
    ("bogus:\n  a: 1\n", "bogus"),
    ("canvas:\n  img_size: 128\n", "canvas.img_size"),
])
def test_invalid_configs_report_key(text, key):
    with pytest.raises(vc.ConfigError) as ei:
        vc.parse_config_text(text)
    assert key in str(ei.value)
    assert any(e.startswith(key) for e in ei.value.errors)


def test_all_errors_collected():
    text = "ticks:\n  len_range: [9, 6]\njitter:\n  nope: 1\n"
    with pytest.raises(vc.ConfigError) as ei:
        vc.parse_config_text(text)
    assert len(ei.value.errors) == 2


def test_missing_key_reported():
    cfg = vc.default_config()
    del cfg["ticks"]["len_range"]
    with pytest.raises(vc.ConfigError) as ei:
        vc.validate_config(cfg)
    assert "ticks.len_range" in str(ei.value)


def test_equal_range_ends_valid():
    cfg = vc.parse_config_text("ticks:\n  len_range: [5, 5]\njitter:\n  contrast_range: [1.0, 1.0]\n")
    assert cfg["ticks"]["len_range"] == [5, 5]


def test_dump_roundtrip():
    cfg = vc.default_config()
    assert vc.parse_config_text(vc.dump_config(cfg)) == cfg
    text = vc.dump_config(cfg, run={"num_images": 3})
    assert "run:" in text
    assert vc.strip_run(yaml.safe_load(text)) == cfg
    assert list(yaml.safe_load(text))[:2] == ["canvas", "dataset"]


def test_load_config_none_and_path(tmp_path):
    assert vc.load_config(None) == vc.default_config()
    p = tmp_path / "c.yaml"
    p.write_text("shadows:\n  prob: 0.9\n", encoding="utf-8")
    assert vc.load_config(str(p))["shadows"]["prob"] == 0.9
