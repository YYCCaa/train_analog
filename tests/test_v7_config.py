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
    """validate_config stays strict: text merged over defaults, no migration."""
    try:
        data = yaml.safe_load(text)  # parsed YAML (dict | None on syntax error)
    except yaml.YAMLError:
        data = None
    with pytest.raises(vc.ConfigError) as ei:
        if data is None:
            vc.parse_config_text(text)
        else:
            vc.validate_config(vc.merge_config(vc.default_config(), data))
    assert key in str(ei.value)
    assert any(e.startswith(key) for e in ei.value.errors)


def test_all_errors_collected():
    cfg = vc.merge_config(vc.default_config(),
                          {"ticks": {"len_range": [9, 6]}, "jitter": {"nope": 1}})
    with pytest.raises(vc.ConfigError) as ei:
        vc.validate_config(cfg)
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


# ---------------------------------------------------------------- old YAMLs
# directory (str) holding sample YAML files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# old-format YAML text (str): scalar perspective / warp keys, a dropped key, an unknown section
OLD_TEXT = (
    "reflections:\n  perspective: 0.4\n  warp_strength: 0.15\n  warp_grid: 4\n"
    "  warp_amplitude: 3\nfoo:\n  bar: 1\n"
)


def _warned(warnings, prefix):
    """The warnings (list[str]) starting with prefix (str)."""
    return [w for w in warnings if w.startswith(prefix)]


def test_old_yaml_migrates_and_warns():
    cfg, warns = vc.parse_config_text_with_warnings(OLD_TEXT)
    vc.validate_config(cfg)
    r = cfg["reflections"]
    assert r["perspective_range"] == [0, 0.4]
    assert r["warp_strength_range"] == [0.15, 0.15]
    assert r["warp_grid_range"] == [4, 4]
    assert "perspective" not in r and "warp_amplitude" not in r
    for key in ("reflections.perspective", "reflections.warp_strength",
                "reflections.warp_grid", "reflections.warp_amplitude", "foo"):
        assert _warned(warns, key), (key, warns)
    assert "removed in a newer version" in _warned(warns, "reflections.warp_amplitude:")[0]
    assert "unknown" in _warned(warns, "foo")[0]


def test_parse_config_text_still_returns_cfg_only():
    cfg = vc.parse_config_text(OLD_TEXT)
    assert isinstance(cfg, dict) and cfg["reflections"]["warp_grid_range"] == [4, 4]


def test_glares_migrates_same_as_reflections():
    cfg, warns = vc.parse_config_text_with_warnings("glares:\n  perspective: 0.2\n  warp_grid: 3\n")
    assert cfg["glares"]["perspective_range"] == [0, 0.2]
    assert cfg["glares"]["warp_grid_range"] == [3, 3]
    assert len(warns) == 2


def test_renamed_noise_and_jitter_keys():
    text = ("pointer:\n  noise_range: 20\n  edge_noise_range: 4\n"
            "digits:\n  dist_jitter_range: [0, 6]\n")
    cfg, warns = vc.parse_config_text_with_warnings(text)
    assert cfg["pointer"]["noise_amplitude"] == 20
    assert cfg["pointer"]["edge_noise_amplitude"] == 4
    assert cfg["digits"]["outside_jitter_range"] == [0, 6]
    assert cfg["digits"]["inside_jitter_range"] == [0, 6]
    assert len(warns) == 3


def test_new_key_wins_over_old_alias():
    cfg, _ = vc.parse_config_text_with_warnings(
        "reflections:\n  perspective: 0.4\n  perspective_range: [0.1, 0.2]\n")
    assert cfg["reflections"]["perspective_range"] == [0.1, 0.2]


def test_missing_sections_and_keys_default_quietly():
    cfg, warns = vc.parse_config_text_with_warnings("ticks:\n  len_range: [7, 8]\n")
    base = vc.default_config()
    base["ticks"]["len_range"] = [7, 8]
    assert cfg == base and warns == []
    cfg, warns = vc.parse_config_text_with_warnings("")
    assert cfg == vc.default_config() and warns == []


def test_out_of_range_is_clamped():
    cfg, warns = vc.parse_config_text_with_warnings(
        "ticks:\n  len_range: [6, 999]\ndigits:\n  show_prob: 7\n")
    assert cfg["ticks"]["len_range"] == [6, 20]
    assert cfg["digits"]["show_prob"] == 1.0
    assert _warned(warns, "ticks.len_range:") and _warned(warns, "digits.show_prob:")


def test_string_where_int_expected_gets_default():
    cfg, warns = vc.parse_config_text_with_warnings("outer:\n  rim_width: wide\n")
    assert cfg["outer"]["rim_width"] == vc.default_config()["outer"]["rim_width"]
    assert _warned(warns, "outer.rim_width:")


def test_reversed_range_is_swapped():
    cfg, warns = vc.parse_config_text_with_warnings("ticks:\n  len_range: [9, 6]\n")
    assert cfg["ticks"]["len_range"] == [6, 9]
    assert _warned(warns, "ticks.len_range:")


def test_integral_float_becomes_int_and_scalar_range_repair():
    cfg, warns = vc.parse_config_text_with_warnings(
        "outer:\n  rim_width: 3.0\nticks:\n  len_range: 8\ngeometry:\n  canvas_margin: [10, 20]\n")
    assert cfg["outer"]["rim_width"] == 3 and isinstance(cfg["outer"]["rim_width"], int)
    assert cfg["ticks"]["len_range"] == [8, 8]
    assert cfg["geometry"]["canvas_margin"] == 15
    assert not _warned(warns, "outer.rim_width:")
    assert _warned(warns, "ticks.len_range:") and _warned(warns, "geometry.canvas_margin:")


def test_float_ends_in_int_range_are_rounded():
    cfg, _ = vc.parse_config_text_with_warnings("ticks:\n  len_range: [6.0, 9.4]\n")
    assert cfg["ticks"]["len_range"] == [6, 9]


def test_bad_section_and_run_block():
    cfg, warns = vc.parse_config_text_with_warnings("ticks: 5\nrun:\n  num_images: 3\n")
    assert cfg == vc.default_config()
    assert _warned(warns, "ticks:") and not _warned(warns, "run")


def test_still_errors_on_syntax_and_non_mapping():
    for text in ("ticks: [\n", "- a\n- b\n"):
        with pytest.raises(vc.ConfigError):
            vc.parse_config_text_with_warnings(text)


def test_migrate_config_is_schema_driven_for_new_sections():
    schema = vc.load_schema()
    schema["extra"] = {"gain": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.1, "help": "x"}}
    defaults = vc.default_config()
    defaults["extra"] = {"gain": 0.5}
    cfg, warns = vc.migrate_config({"extra": {"gain": 3, "old": 1}}, schema, defaults)
    assert cfg["extra"]["gain"] == 1.0 and len(warns) == 2
    vc.validate_config(cfg, schema)
    cfg, warns = vc.migrate_config({}, schema, defaults)
    assert cfg["extra"] == {"gain": 0.5} and warns == []


@pytest.mark.parametrize("data", [
    {"geometry": {"displacement_range": "x", "canvas_margin": None}, "digits": {"show_prob": [1, 2]}},
    {"pointer": {"gray_range": [1, 2, 3], "noise_amplitude": 1e9}, "cut_tip": {"enabled": "yes"}},
    {"ticks": {"len_range": [-5, 500]}, "canvas": {"img_size": 128}},
])
def test_every_migrated_result_validates(data):
    cfg, warns = vc.migrate_config(data)
    vc.validate_config(cfg)
    assert warns


def test_validate_config_still_rejects_unknown_keys():
    cfg = vc.default_config()
    cfg["reflections"]["perspective"] = 0.4
    with pytest.raises(vc.ConfigError, match="reflections.perspective"):
        vc.validate_config(cfg)


def test_load_config_warns_on_stderr(tmp_path, capsys):
    p = tmp_path / "old.yaml"  # old-format YAML file (Path)
    p.write_text(OLD_TEXT, encoding="utf-8")
    cfg = vc.load_config(str(p))
    assert cfg["reflections"]["perspective_range"] == [0, 0.4]
    err = capsys.readouterr().err
    assert "[v7_config] warning: reflections.perspective:" in err
    assert "[v7_config] warning: foo" in err


def test_load_config_quiet_when_clean(tmp_path, capsys):
    p = tmp_path / "clean.yaml"  # current-format YAML file (Path)
    p.write_text(vc.dump_config(vc.default_config()), encoding="utf-8")
    assert vc.load_config(str(p)) == vc.default_config()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("name", ["user_sample_config.yaml", "old_v1_config.yaml"])
def test_old_sample_configs_load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        cfg, warns = vc.parse_config_text_with_warnings(f.read())
    vc.validate_config(cfg)
    if name == "old_v1_config.yaml":
        assert cfg["reflections"]["perspective_range"] == [0, 0.4]
        assert cfg["pointer"]["noise_amplitude"] == 20
        assert _warned(warns, "foo") and _warned(warns, "shadows.opacity_range:")


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
    # pixel-count tip label section: right after cut_tip, one key min_pixels
    cfg = vc.default_config()
    schema = vc.load_schema()
    assert cfg["tip_label"] == {"min_pixels": 2}
    order = list(cfg)  # list[str], section order of the default YAML
    assert order.index("tip_label") == order.index("cut_tip") + 1
    assert list(schema).index("tip_label") == list(schema).index("cut_tip") + 1
    leaf = schema["tip_label"]["min_pixels"]  # dict, schema leaf
    assert (leaf["type"], leaf["min"], leaf["max"], leaf["step"]) == ("int", 0, 30, 1)
    assert leaf["help"] == ("Tip label = point on the pointer triangle's height nearest the "
                            "apex where the line parallel to the base crosses more than this "
                            "many pointer pixels.")
    assert "min_pixels" in schema["cut_tip"]["enabled"]["help"]


def test_old_tip_label_keys_removed_with_warning():
    # a YAML from the blur-aware label era loads; its four keys are dropped
    old = ("tip_label:\n  contrast_frac: 0.6\n  base_blur: 2.0\n  patch_size: 11\n"
           "  min_ratio: 0.3\n")  # str, old tip_label section
    cfg, warns = vc.parse_config_text_with_warnings(old)
    assert cfg["tip_label"] == {"min_pixels": 2}
    for key in ("contrast_frac", "base_blur", "patch_size", "min_ratio"):
        w = _warned(warns, f"tip_label.{key}:")  # list[str]
        assert w and "removed in a newer version" in w[0], (key, warns)


def test_range_types_match_draw_kind():
    schema = _leaves(vc.load_schema())
    cfg = _leaves(vc.default_config())
    for name in FLOAT_RANGES:
        assert schema[name]["type"] == "float_range", name
        assert all(isinstance(x, float) for x in cfg[name]), name
    for name in INT_RANGES:
        assert schema[name]["type"] == "int_range", name
        assert all(isinstance(x, int) for x in cfg[name]), name
