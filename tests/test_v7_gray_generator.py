"""CLI-level tests for generate_synthetic_v7_mp.py (native-gray, YAML-driven).

Drives the generator as a user would (`python generate_synthetic_v7_mp.py ...`
via subprocess) and inspects the files it wrote.  Only the cancel test calls
generate_dataset directly (a cancel needs an in-process threading.Event).
"""

import csv
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (pathlib.Path)
sys.path.insert(0, str(REPO_ROOT))

import v7_config  # noqa: E402
from generate_synthetic_v7_mp import (  # noqa: E402
    _seed_for_index, generate_dataset, render_image)

SEED = 3  # base RNG seed passed via --seed (int)


def run_cli(*args):
    """Run the generator CLI -> subprocess.CompletedProcess (text output)."""
    return subprocess.run(
        [sys.executable, "generate_synthetic_v7_mp.py", *map(str, args)],
        cwd=REPO_ROOT, timeout=300, capture_output=True, text=True)


def read_labels(folder):
    """labels.csv -> list[dict[str, str]], one dict per row."""
    with open(Path(folder) / "labels.csv", newline="") as f:
        return list(csv.DictReader(f))


def read_run(folder):
    """config.yaml -> parsed dict (cfg sections plus 'run')."""
    return yaml.safe_load((Path(folder) / "config.yaml").read_text(encoding="utf-8"))


def test_cli_outputs(tmp_path):
    out = tmp_path / "o"
    r = run_cli("--output-dir", out, "--num-images", 6, "--workers", 2, "--seed", SEED)
    assert r.returncode == 0, r.stderr
    rows = read_labels(out)
    assert len(rows) == 6
    assert list(rows[0].keys()) == ["filename", "angle_deg", "tip_x", "tip_y"]
    for row in rows:
        im = Image.open(out / row["filename"])
        assert im.mode == "L" and im.size == (256, 256)
        assert 0 <= float(row["tip_x"]) < 256 and 0 <= float(row["tip_y"]) < 256
    doc = read_run(out)
    run = doc["run"]
    assert v7_config.strip_run(doc) == v7_config.default_config()
    assert run["completed"] is True and run["num_images"] == 6


def test_workers_bit_exact(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_cli("--output-dir", a, "--num-images", 8, "--workers", 1, "--seed", SEED).returncode == 0
    assert run_cli("--output-dir", b, "--num-images", 8, "--workers", 4, "--seed", SEED).returncode == 0
    assert (a / "labels.csv").read_text() == (b / "labels.csv").read_text()
    for i in range(8):
        n = f"dial_{i:04d}.png"
        assert np.array_equal(np.array(Image.open(a / n)), np.array(Image.open(b / n)))


def test_preview_equals_disk(tmp_path):
    out = tmp_path / "o"
    assert run_cli("--output-dir", out, "--num-images", 6, "--workers", 2, "--seed", SEED).returncode == 0
    cfg = v7_config.default_config()
    for idx in range(6):
        want = render_image(cfg, _seed_for_index(SEED, idx)).image
        got = np.array(Image.open(out / f"dial_{idx:04d}.png"))
        assert np.array_equal(want, got)


def test_cli_partial_config(tmp_path):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("ticks:\n  len_range: [12, 14]\n")
    out = tmp_path / "o"
    r = run_cli("--config", cfgp, "--output-dir", out, "--num-images", 2, "--workers", 1)
    assert r.returncode == 0, r.stderr
    assert read_run(out)["ticks"]["len_range"] == [12, 14]


def test_cli_bad_config(tmp_path):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("ticks: [\n")  # YAML syntax error: the only thing load_config still rejects
    out = tmp_path / "o"
    r = run_cli("--config", cfgp, "--output-dir", out, "--num-images", 2)
    assert r.returncode == 2
    assert "yaml" in r.stderr + r.stdout


def test_overwrite_semantics(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "dial_0099.png").write_bytes(b"x")
    (out / "labels.csv").write_text("old")
    (out / "notes.txt").write_text("keep")
    r = run_cli("--output-dir", out, "--num-images", 2, "--workers", 1)
    assert r.returncode != 0
    assert "--overwrite" in r.stderr + r.stdout
    assert (out / "dial_0099.png").exists()
    r = run_cli("--output-dir", out, "--num-images", 2, "--workers", 1, "--overwrite")
    assert r.returncode == 0, r.stderr
    assert not (out / "dial_0099.png").exists()
    assert (out / "notes.txt").read_text() == "keep"


def test_cancel_leaves_consistent_folder(tmp_path):
    out = tmp_path / "o"
    ev = threading.Event()  # cancel flag set by the progress callback

    def cb(done, total):
        if done >= 20:
            ev.set()

    res = generate_dataset(v7_config.default_config(), str(out), 400, workers=4,
                           cancel_event=ev, progress_cb=cb)
    assert res["completed"] is False
    pngs = {p.name for p in out.glob("dial_*.png")}
    assert pngs == {r["filename"] for r in read_labels(out)}
    assert len(pngs) >= 20
    assert read_run(out)["run"]["completed"] is False


def test_worker_error_leaves_consistent_folder(tmp_path):
    out = tmp_path / "o"

    def cb(done, total):
        if done >= 10:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        generate_dataset(v7_config.default_config(), str(out), 200, workers=4,
                         progress_cb=cb)
    pngs = {p.name for p in out.glob("dial_*.png")}
    assert pngs == {r["filename"] for r in read_labels(out)}
    run = read_run(out)["run"]
    assert run["completed"] is False and run["finished_at"] is not None


def test_glob_metachars_in_output_dir(tmp_path):
    """A '[1]' in --output-dir is a literal folder name, not a glob pattern:
    'run[1]' must never match (and delete / count) files in the sibling 'run1'."""
    decoy = tmp_path / "run1"  # sibling folder a pattern 'run[1]' would match (Path)
    decoy.mkdir()
    (decoy / "dial_0000.png").write_bytes(b"decoy")
    (decoy / "labels.csv").write_text("decoy")
    target = tmp_path / "run[1]"  # literal folder name with glob metachars (Path)

    # fresh run[1] beside a populated run1: must not be refused as not-empty
    r = run_cli("--output-dir", target, "--num-images", 3, "--workers", 1)
    assert r.returncode == 0, r.stderr
    # overwrite run: only run[1]'s own files may be removed
    r = run_cli("--output-dir", target, "--num-images", 3, "--workers", 1, "--overwrite")
    assert r.returncode == 0, r.stderr

    assert (decoy / "dial_0000.png").read_bytes() == b"decoy"
    assert (decoy / "labels.csv").read_text() == "decoy"
    assert len(read_labels(target)) == 3
    assert sorted(p.name for p in target.iterdir() if p.name.startswith("dial_")) == \
        [f"dial_{i:04d}.png" for i in range(3)]


# ── --save-params / params.jsonl ──────────────────────────────────────────

def read_params(folder):
    """params.jsonl -> list[dict], one parsed record per line."""
    text = (Path(folder) / "params.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def test_save_params_cli(tmp_path):
    out = tmp_path / "o"
    r = run_cli("--output-dir", out, "--num-images", 6, "--workers", 2, "--seed", SEED,
                "--save-params")
    assert r.returncode == 0, r.stderr
    rows = read_labels(out)
    recs = read_params(out)
    assert len(recs) == 6 == len(rows)
    cfg = v7_config.default_config()
    for i, (rec, row) in enumerate(zip(recs, rows)):
        assert rec["filename"] == row["filename"]
        assert rec["seed"] == _seed_for_index(SEED, i)
        assert rec["angle_deg"] == int(row["angle_deg"])
        assert rec["tip_x"] == float(row["tip_x"]) and rec["tip_y"] == float(row["tip_y"])
        want = json.loads(json.dumps(render_image(cfg, rec["seed"]).params))
        assert rec["sampled"] == want
    assert read_run(out)["run"]["save_params"] is True


def test_no_save_params_identical_outputs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_cli("--output-dir", a, "--num-images", 6, "--workers", 2, "--seed", SEED).returncode == 0
    assert run_cli("--output-dir", b, "--num-images", 6, "--workers", 2, "--seed", SEED,
                   "--save-params").returncode == 0
    assert not (a / "params.jsonl").exists() and (b / "params.jsonl").exists()
    assert read_run(a)["run"]["save_params"] is False
    assert (a / "labels.csv").read_bytes() == (b / "labels.csv").read_bytes()
    for i in range(6):
        n = f"dial_{i:04d}.png"
        assert (a / n).read_bytes() == (b / n).read_bytes()
    assert v7_config.strip_run(read_run(a)) == v7_config.strip_run(read_run(b))
    # run blocks (dict): equal except run.save_params (the option under test) and
    # run.finished_at (str timestamp, differs per run)
    run_a, run_b = read_run(a)["run"], read_run(b)["run"]
    assert run_a["save_params"] is False and run_b["save_params"] is True
    for run in (run_a, run_b):
        run.pop("save_params")
        run.pop("finished_at")
    assert run_a == run_b


def test_help_lists_params_output_and_flag():
    out = run_cli("--help").stdout  # str, full --help text
    assert "--save-params" in out
    # argparse wraps help text, so collapse whitespace before looking for the --output-dir sentence
    flat = " ".join(out.split())
    assert "labels.csv, params.jsonl (with --save-params), config.yaml" in flat


def test_save_params_workers_bit_exact(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out, w in ((a, 1), (b, 4)):
        assert run_cli("--output-dir", out, "--num-images", 8, "--workers", w, "--seed", SEED,
                       "--save-params").returncode == 0
    assert (a / "params.jsonl").read_bytes() == (b / "params.jsonl").read_bytes()


def test_cancel_and_error_keep_params_aligned(tmp_path):
    cfg = v7_config.default_config()

    def check(out):
        pngs = {p.name for p in out.glob("dial_*.png")}
        names = [r["filename"] for r in read_labels(out)]
        assert [r["filename"] for r in read_params(out)] == names
        assert pngs == set(names)

    out = tmp_path / "cancel"
    ev = threading.Event()  # cancel flag set by the progress callback

    def cancel_cb(done, total):
        if done >= 20:
            ev.set()

    res = generate_dataset(cfg, str(out), 400, seed=SEED, workers=2, save_params=True,
                           cancel_event=ev, progress_cb=cancel_cb)
    assert res["completed"] is False
    check(out)

    out2 = tmp_path / "error"

    def err_cb(done, total):
        if done >= 5:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        generate_dataset(cfg, str(out2), 200, seed=SEED, workers=2, save_params=True,
                         progress_cb=err_cb)
    check(out2)


def test_overwrite_and_not_empty_with_params(tmp_path):
    out = tmp_path / "o"
    assert run_cli("--output-dir", out, "--num-images", 3, "--workers", 1,
                   "--save-params").returncode == 0
    assert (out / "params.jsonl").exists()
    r = run_cli("--output-dir", out, "--num-images", 3, "--workers", 1, "--overwrite")
    assert r.returncode == 0, r.stderr
    assert not (out / "params.jsonl").exists()

    only = tmp_path / "only"
    only.mkdir()
    (only / "params.jsonl").write_text("{}\n")
    r = run_cli("--output-dir", only, "--num-images", 2, "--workers", 1)
    assert r.returncode == 1
    assert "params.jsonl" in r.stderr


def test_params_non_finite_strict_json(tmp_path, monkeypatch):
    import generate_synthetic_v7_mp as g
    real = g.render_image

    def fake(cfg, seed):
        res = real(cfg, seed)
        res.params = {"a": float("nan"), "n": {"b": float("inf")},
                      "l": [1.5, -float("inf")]}
        return res

    monkeypatch.setattr(g, "render_image", fake)

    def boom(const):
        raise AssertionError(f"non-strict constant {const}")

    out = tmp_path / "o"
    g.generate_dataset(v7_config.default_config(), str(out), 3, seed=SEED, workers=1,
                       save_params=True)
    lines = [ln for ln in (out / "params.jsonl").read_text(encoding="utf-8").split("\n") if ln]
    assert len(lines) == 3
    for ln in lines:
        s = json.loads(ln, parse_constant=boom)["sampled"]
        assert s == {"a": "NaN", "n": {"b": "Infinity"}, "l": [1.5, "-Infinity"]}
