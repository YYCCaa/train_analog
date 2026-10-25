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
    # reflections / glares ship off so default datasets need no Places365
    assert cfg["reflections"]["prob"] == 0.0 and cfg["glares"]["prob"] == 0.0
    assert cfg["reflections"]["darken_range"] == [10, 40]
    assert cfg["glares"]["value_range"] == [220, 255]
    assert cfg["glares"]["clear_distance"] == 12.0


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
    # removed by the physical shadow model: old YAMLs are rejected
    ("shadows:\n  gray_range: [11, 60]\n", "shadows.gray_range"),
    ("shadows:\n  opacity_range: [160, 180]\n", "shadows.opacity_range"),
    ("shadows:\n  rim_opacity_range: [200, 220]\n", "shadows.rim_opacity_range"),
    ("shadows:\n  strength_range: [0.2, 1.5]\n", "shadows.strength_range"),
    ("shadows:\n  strength_jitter: 0.6\n", "shadows.strength_jitter"),
    # removed by earlier shape sources: old Places365-patch / bank YAMLs are rejected
    ("reflections:\n  threshold_pct_range: [55.0, 80.0]\n", "reflections.threshold_pct_range"),
    ("reflections:\n  patch_size_range: [24, 64]\n", "reflections.patch_size_range"),
    ("glares:\n  threshold_pct_range: [70.0, 92.0]\n", "glares.threshold_pct_range"),
    ("glares:\n  patch_size_range: [16, 48]\n", "glares.patch_size_range"),
    # removed with the glyph shapes (warp is now relative): old YAMLs are rejected
    ("reflections:\n  warp_amplitude: 4.0\n", "reflections.warp_amplitude"),
    ("glares:\n  warp_amplitude: 3.0\n", "glares.warp_amplitude"),
    # glyph-transform keys are bounded
    ("glares:\n  aspect_range: [0.05, 1.0]\n", "glares.aspect_range"),
    ("reflections:\n  perspective: 0.95\n", "reflections.perspective"),
    ("glares:\n  warp_strength: 1.5\n", "glares.warp_strength"),
    ("reflections:\n  warp_grid: 13\n", "reflections.warp_grid"),
    ("glares:\n  size_range: [8, 300]\n", "glares.size_range"),
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


FLOAT_RANGES = [  # keys the old code draws with random.uniform
    "geometry.displacement_range", "geometry.ptr_displacement_range",
    "geometry.hole_disp_range", "geometry.arc_gap_range",
    "ticks.angle_jitter_range", "digits.outside_jitter_range",
    "digits.inside_jitter_range",
    "digits.angle_jitter_range", "digits.gap_range",
    "pointer.gradient_range", "shadows.offset_range",
    "geometry.scale_range", "jitter.contrast_range",
    "shadows.strength_range",
    "reflections.rotation_range", "reflections.aspect_range", "reflections.shear_range",
    "glares.rotation_range", "glares.aspect_range", "glares.shear_range",
    "reflections.perspective_range", "reflections.warp_strength_range",
    "glares.perspective_range", "glares.warp_strength_range",
]
INT_RANGES = [  # keys the old code draws with random.randint
    "ticks.len_range", "ticks.per_digit_range", "cut_tip.radius_range",
    "tri_artifacts.count_range", "dataset.angle_deg_range",
    "digits.font_size_range",
    "reflections.count_range", "reflections.size_range",
    "reflections.darken_range", "glares.count_range",
    "glares.size_range", "glares.value_range",
    "reflections.warp_grid_range", "glares.warp_grid_range",
]


def test_glyph_shape_keys_and_defaults():
    # pinned defaults + schema bounds of the glyph-transform keys (brief table)
    cfg = vc.default_config()
    shared = {"rotation_range": [0.0, 360.0], "aspect_range": [0.25, 4.0],
              "shear_range": [-1.0, 1.0], "perspective_range": [0.0, 0.4],
              "warp_strength_range": [0.05, 0.2], "warp_grid_range": [3, 6]}  # key -> default value, same in both sections
    want = {  # section -> key -> default value
        "reflections": {"size_range": [24, 96], **shared},
        "glares": {"size_range": [16, 72], **shared},
    }
    bounds = {  # key -> (type, min, max, step) in the schema
        "size_range": ("int_range", 8, 256, 1), "rotation_range": ("float_range", 0, 360, 1),
        "aspect_range": ("float_range", 0.1, 10, 0.05), "shear_range": ("float_range", -3, 3, 0.05),
        "perspective_range": ("float_range", 0, 0.9, 0.01),
        "warp_strength_range": ("float_range", 0, 1.0, 0.01),
        "warp_grid_range": ("int_range", 2, 12, 1),
    }
    schema = vc.load_schema()
    for sec, keys in want.items():
        for gone in ("threshold_pct_range", "patch_size_range", "warp_amplitude",
                     "perspective", "warp_strength", "warp_grid"):
            assert gone not in cfg[sec] and gone not in schema[sec], (sec, gone)
        for key, val in keys.items():
            assert cfg[sec][key] == val, (sec, key)
        for key, (typ, lo, hi, step) in bounds.items():
            leaf = schema[sec][key]  # dict, schema leaf
            assert (leaf["type"], leaf["min"], leaf["max"], leaf["step"]) == (typ, lo, hi, step),                 (sec, key)
            assert leaf.get("help"), (sec, key)


def test_tip_label_keys_and_defaults():
    # blur-aware tip label section: right after cut_tip, pinned defaults + bounds
    cfg = vc.default_config()
    schema = vc.load_schema()
    assert cfg["tip_label"] == {"contrast_frac": 0.5, "base_blur": 1.0}
    order = list(cfg)  # list[str], section order of the default YAML
    assert order.index("tip_label") == order.index("cut_tip") + 1
    assert list(schema).index("tip_label") == list(schema).index("cut_tip") + 1
    bounds = {"contrast_frac": ("float", 0.05, 0.95, 0.01),
              "base_blur": ("float", 0, 4, 0.1)}  # key -> (type, min, max, step)
    for key, (typ, lo, hi, step) in bounds.items():
        leaf = schema["tip_label"][key]  # dict, schema leaf
        assert (leaf["type"], leaf["min"], leaf["max"], leaf["step"]) == (typ, lo, hi, step), key
        assert leaf.get("help"), key
    assert "wide enough to see after blur" in schema["cut_tip"]["enabled"]["help"]


def test_range_types_match_draw_kind():
    schema = _leaves(vc.load_schema())
    cfg = _leaves(vc.default_config())
    for name in FLOAT_RANGES:
        assert schema[name]["type"] == "float_range", name
        assert all(isinstance(x, float) for x in cfg[name]), name
    for name in INT_RANGES:
        assert schema[name]["type"] == "int_range", name
        assert all(isinstance(x, int) for x in cfg[name]), name
