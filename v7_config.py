"""YAML config layer for the native-gray v7 generator and the tuning studio.

cfg shape: dict[section: str, dict[key: str, value]] where value is
int | float | bool | [lo, hi] (list of 2 ints or 2 floats).
Schema shape: same nesting; each leaf is
{type, min, max, step, help, readonly?} with type in
int | float | bool | int_range | float_range.
"""

import copy
import os

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


def parse_config_text(text):
    """YAML text -> defaults merged with text -> validated cfg.

    A top-level run: block (as written by dump_config) is ignored so a
    generated config.yaml can be loaded back. YAML syntax errors and
    non-mapping documents become ConfigError.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError([f"yaml: {str(e).splitlines()[0] if str(e) else 'syntax error'}"]) from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(["yaml: top level must be a mapping of sections"])
    cfg = merge_config(default_config(), strip_run(data))
    validate_config(cfg)
    return cfg


def load_config(path):
    """Load cfg from a YAML file path; None -> defaults."""
    if path is None:
        return default_config()
    with open(path, "r", encoding="utf-8") as f:
        return parse_config_text(f.read())


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
