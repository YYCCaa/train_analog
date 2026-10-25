"""YAML config layer for the native-gray v7 generator and the tuning studio.

cfg shape: dict[section: str, dict[key: str, value]] where value is
int | float | bool | [lo, hi] (list of 2 ints or 2 floats).
Schema shape: same nesting; each leaf is
{type, min, max, step, help, readonly?} with type in
int | float | bool | int_range | float_range.
"""

import copy
import math
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
# absolute path (str) of the default parameter values YAML
DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "v7_default.yaml")
# absolute path (str) of the UI / validation schema YAML
SCHEMA_PATH = os.path.join(_HERE, "configs", "v7_schema.yaml")

# top-level key (str) under which run settings are appended by dump_config
RUN_KEY = "run"


class ConfigError(ValueError):
    """Invalid config. .errors is list[str]; each starts with "section.key: "."""

    def __init__(self, errors):
        # list[str], one message per problem found
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def _read_yaml(path):
    """Load a YAML file into a dict (nested section->key mapping)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_schema():
    """Return the schema dict (section -> key -> leaf dict), fresh each call."""
    return _read_yaml(SCHEMA_PATH)


def default_config():
    """Return a fresh deep copy of the default cfg (section -> key -> value)."""
    return _read_yaml(DEFAULT_CONFIG_PATH)


def merge_config(base, override):
    """Deep-merge override onto base; returns a new dict, inputs untouched.

    Key order of base is kept; keys only in override are appended.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_int(v):
    """True for real ints (bool excluded, it subclasses int)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    """True for ints or floats (bool excluded)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_leaf(name, value, leaf):
    """Return list[str] of problems for one value against its schema leaf."""
    t, lo, hi = leaf["type"], leaf["min"], leaf["max"]
    if t == "bool":
        return [] if isinstance(value, bool) else [f"{name}: expected true/false, got {value!r}"]
    if t in ("int", "float"):
        ok = _is_int(value) if t == "int" else _is_num(value)
        if not ok:
            return [f"{name}: expected {t}, got {value!r}"]
        if not lo <= value <= hi:
            return [f"{name}: {value} outside [{lo}, {hi}]"]
        return []
    if t in ("int_range", "float_range"):
        if not (isinstance(value, list) and len(value) == 2):
            return [f"{name}: expected a [lo, hi] list, got {value!r}"]
        check = _is_int if t == "int_range" else _is_num
        if not all(check(x) for x in value):
            kind = "integers" if t == "int_range" else "numbers"
            return [f"{name}: both ends must be {kind}, got {value!r}"]
        a, b = value
        if a > b:
            return [f"{name}: lo > hi ({a} > {b})"]
        if a < lo or b > hi:
            return [f"{name}: {value} outside [{lo}, {hi}]"]
        return []
    return [f"{name}: unknown schema type {t!r}"]


def validate_config(cfg, schema=None):
    """Raise ConfigError (all problems collected) unless cfg matches schema."""
    schema = load_schema() if schema is None else schema
    errors = []
    if not isinstance(cfg, dict):
        raise ConfigError(["config: top level must be a mapping"])
    for sec, body in cfg.items():
        if sec not in schema:
            errors.append(f"{sec}: unknown section")
            continue
        if not isinstance(body, dict):
            errors.append(f"{sec}: section must be a mapping, got {body!r}")
            continue
        for key, value in body.items():
            if key not in schema[sec]:
                errors.append(f"{sec}.{key}: unknown key")
            else:
                errors += _check_leaf(f"{sec}.{key}", value, schema[sec][key])
    for sec, keys in schema.items():
        for key in keys:
            if sec in cfg and isinstance(cfg[sec], dict) and key not in cfg[sec]:
                errors.append(f"{sec}.{key}: missing key")
        if sec not in cfg:
            errors.append(f"{sec}: missing section")
    if errors:
        raise ConfigError(errors)


def strip_run(cfg_with_run):
    """Return a shallow copy of cfg without the top-level run block."""
    return {k: v for k, v in cfg_with_run.items() if k != RUN_KEY}


def _same(v):
    """Migration transform: copy the old value unchanged."""
    return v


def _to_range(lo_fn):
    """Migration transform factory: scalar p -> [lo_fn(p), p]; non-numbers pass through
    unchanged so the value repair step reports them."""
    def fn(v):
        return [lo_fn(v), v] if _is_num(v) else v
    return fn


# Old key -> successors, shape dict[(section str, old key str)] -> list[(new key str,
# transform callable value -> value)]. The old value is renamed / reshaped into every
# successor that the YAML does not already set itself. Found with
# `git log -p --follow configs/v7_schema.yaml`:
#   reflections/glares  perspective p     -> perspective_range   [0, p]   (per-item draw)
#   reflections/glares  warp_strength s   -> warp_strength_range [s, s]
#   reflections/glares  warp_grid g       -> warp_grid_range     [g, g]
#   pointer  noise_range n                -> noise_amplitude     n   (renamed)
#   pointer  edge_noise_range n           -> edge_noise_amplitude n  (renamed)
#   digits   dist_jitter_range [a, b]     -> outside_jitter_range + inside_jitter_range
#                                            (the bidirectional jitter became two one-way ones)
_MIGRATIONS = {
    ("pointer", "noise_range"): [("noise_amplitude", _same)],
    ("pointer", "edge_noise_range"): [("edge_noise_amplitude", _same)],
    ("digits", "dist_jitter_range"): [("outside_jitter_range", _same),
                                      ("inside_jitter_range", _same)],
}
for _sec in ("reflections", "glares"):
    _MIGRATIONS[(_sec, "perspective")] = [("perspective_range", _to_range(lambda p: 0))]
    _MIGRATIONS[(_sec, "warp_strength")] = [("warp_strength_range", _to_range(lambda s: s))]
    _MIGRATIONS[(_sec, "warp_grid")] = [("warp_grid_range", _to_range(lambda g: g))]

# Old keys with no successor, shape set[(section str, key str)]; dropped with a
# "removed in a newer version" warning instead of the generic "unknown key".
_REMOVED = {
    ("reflections", "warp_amplitude"), ("glares", "warp_amplitude"),
    ("reflections", "threshold_pct_range"), ("glares", "threshold_pct_range"),
    ("reflections", "patch_size_range"), ("glares", "patch_size_range"),
    ("shadows", "gray_range"), ("shadows", "opacity_range"), ("shadows", "rim_opacity_range"),
    # blur-aware tip label keys, replaced by the pixel-count rule tip_label.min_pixels
    ("tip_label", "contrast_frac"), ("tip_label", "base_blur"),
    ("tip_label", "patch_size"), ("tip_label", "min_ratio"),
    # px pointer-shadow offset / constant blur, replaced by the modelled shadow
    # (cast distance as a fraction of the dial radius, penumbra growing with it);
    # px -> fraction has no exact conversion, so they are dropped with a hint
    ("shadows", "offset_range"), ("shadows", "blur_radius"),
}

# Replacement hint per removed key, shape dict[(section str, key str), str];
# appended to the "removed" warning so old YAMLs say what to set instead.
_REMOVED_HINTS = {
    ("shadows", "offset_range"): "use shadows.distance_frac_range (fraction of the dial radius)",
    ("shadows", "blur_radius"): "use shadows.blur_min and shadows.blur_per_px",
}


def _fmt(v):
    """Short repr of a value for warning text."""
    return repr(v)


def _repair_number(name, v, leaf, default, warns, what):
    """Repair one scalar v against a numeric leaf; returns int | float.

    what is "int" or "float". Wrong type -> default, integral float -> int,
    fractional float for an int -> rounded, out of [min, max] -> clamped.
    """
    lo, hi = leaf["min"], leaf["max"]
    if not _is_num(v) or not math.isfinite(v):
        warns.append(f"{name}: expected {what}, got {_fmt(v)}; using default {_fmt(default)}")
        return default
    if what == "int" and not _is_int(v):
        r = int(round(v))  # nearest int (int)
        if r != v:
            warns.append(f"{name}: expected int, got {v}; rounded to {r}")
        v = r
    if v < lo or v > hi:
        c = min(max(v, lo), hi)  # clamped value
        warns.append(f"{name}: {v} outside [{lo}, {hi}]; clamped to {c}")
        v = c
    return v


def _repair_leaf(name, value, leaf, default, warns):
    """Return value repaired to satisfy schema leaf (dict); problems appended to
    warns (list[str]). default is the fallback when value is unusable."""
    t = leaf["type"]
    if t == "bool":
        if isinstance(value, bool):
            return value
        warns.append(f"{name}: expected true/false, got {_fmt(value)}; using default {_fmt(default)}")
        return default
    if t in ("int", "float"):
        if isinstance(value, list) and len(value) == 2 and all(_is_num(x) for x in value):
            mid = (value[0] + value[1]) / 2  # midpoint of the range (float)
            mid = int(round(mid)) if t == "int" else mid
            warns.append(f"{name}: expected a single value, got range {value}; used midpoint {mid}")
            value = mid
        return _repair_number(name, value, leaf, default, warns, t)
    if t in ("int_range", "float_range"):
        what = "int" if t == "int_range" else "float"  # element type
        if _is_num(value):
            warns.append(f"{name}: expected a [lo, hi] list, got {value}; used [{value}, {value}]")
            value = [value, value]
        if not (isinstance(value, list) and len(value) == 2):
            warns.append(f"{name}: expected a [lo, hi] list, got {_fmt(value)}; "
                         f"using default {_fmt(default)}")
            return list(default)
        if not all(_is_num(x) and math.isfinite(x) for x in value):
            warns.append(f"{name}: both ends must be numbers, got {_fmt(value)}; "
                         f"using default {_fmt(default)}")
            return list(default)
        a, b = value
        if a > b:
            warns.append(f"{name}: lo > hi ({a} > {b}); swapped")
            a, b = b, a
        return [_repair_number(name, a, leaf, default[0], warns, what),
                _repair_number(name, b, leaf, default[1], warns, what)]
    raise ConfigError([f"{name}: unknown schema type {t!r}"])


def migrate_config(data, schema=None, defaults=None):
    """Lenient load of a possibly old / hand-edited cfg dict -> (cfg, warnings).

    data: dict[section str, dict[key str, value]] (not modified); schema / defaults
    default to the current files. Steps: known renames / reshapes (_MIGRATIONS), drop
    removed or unknown sections and keys, fill anything missing from defaults, repair
    each value against its schema leaf (type, int/float, scalar vs range, order,
    bounds). Returns cfg (dict, section order of the schema; always passes
    validate_config) and warnings (list[str], each "section.key: message").
    A top-level run: block is ignored silently.
    """
    schema = load_schema() if schema is None else schema
    defaults = default_config() if defaults is None else defaults
    warns = []  # list[str]
    # section -> key -> raw value after renames; unknown keys already dropped
    pending = {sec: {} for sec in schema}
    for sec, body in data.items():
        if sec == RUN_KEY:
            continue
        if sec not in schema:
            warns.append(f"{sec}: unknown section, ignored")
            continue
        if not isinstance(body, dict):
            warns.append(f"{sec}: section must be a mapping, got {_fmt(body)}; using defaults")
            continue
        for key, value in body.items():
            if key in schema[sec]:
                pending[sec][key] = value
    for sec, body in data.items():
        if sec not in schema or not isinstance(body, dict):
            continue
        for key, value in body.items():
            if key in schema[sec]:
                continue
            name = f"{sec}.{key}"
            moves = _MIGRATIONS.get((sec, key))  # successors or None
            if moves:
                done = []  # successor keys actually filled (list[str])
                for new_key, fn in moves:
                    if new_key in schema[sec] and new_key not in body:
                        pending[sec][new_key] = fn(copy.deepcopy(value))
                        done.append(new_key)
                if done:
                    warns.append(f"{name}: renamed to {', '.join(done)}")
                else:
                    warns.append(f"{name}: replaced by {moves[0][0]} which is also set; ignored")
            elif (sec, key) in _REMOVED:
                hint = _REMOVED_HINTS.get((sec, key))  # str | None, replacement keys
                warns.append(f"{name}: removed in a newer version, ignored"
                             + (f"; {hint}" if hint else ""))
            else:
                warns.append(f"{name}: unknown key, ignored")
    cfg = {}
    for sec, keys in schema.items():
        cfg[sec] = {}
        for key, leaf in keys.items():
            default = defaults[sec][key]
            if key in pending[sec]:
                cfg[sec][key] = _repair_leaf(f"{sec}.{key}", pending[sec][key], leaf,
                                             default, warns)
            else:
                cfg[sec][key] = copy.deepcopy(default)
    return cfg, warns


def parse_config_text_with_warnings(text):
    """YAML text -> (cfg, warnings) via the lenient migrate_config.

    A top-level run: block (as written by dump_config) is ignored so a
    generated config.yaml can be loaded back. Only YAML syntax errors and
    non-mapping documents raise ConfigError. warnings is list[str].
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError([f"yaml: {str(e).splitlines()[0] if str(e) else 'syntax error'}"]) from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(["yaml: top level must be a mapping of sections"])
    cfg, warns = migrate_config(strip_run(data))
    validate_config(cfg)
    return cfg, warns


def parse_config_text(text):
    """Like parse_config_text_with_warnings but returns only the cfg dict."""
    return parse_config_text_with_warnings(text)[0]


def load_config(path):
    """Load cfg from a YAML file path; None -> defaults. Old-YAML adjustments are
    printed to stderr as "[v7_config] warning: <message>"."""
    if path is None:
        return default_config()
    with open(path, "r", encoding="utf-8") as f:
        cfg, warns = parse_config_text_with_warnings(f.read())
    for w in warns:
        print(f"[v7_config] warning: {w}", file=sys.stderr)
    return cfg


class _FlowListDumper(yaml.SafeDumper):
    """SafeDumper that writes lists in flow style, e.g. [6, 9]."""


_FlowListDumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True),
)


def dump_config(cfg, run=None):
    """cfg -> YAML text (section order kept); run dict appended as run: block."""
    doc = dict(cfg)
    if run is not None:
        doc[RUN_KEY] = run
    return yaml.dump(doc, Dumper=_FlowListDumper, sort_keys=False,
                     default_flow_style=False, allow_unicode=True)
