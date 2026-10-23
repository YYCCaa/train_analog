"""CLI-level tests for generate_synthetic_v7_mp.py (native-gray, YAML-driven).

Drives the generator as a user would (`python generate_synthetic_v7_mp.py ...`
via subprocess) and inspects the files it wrote.  Only the cancel test calls
generate_dataset directly (a cancel needs an in-process threading.Event).
"""

import csv
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
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
        assert 0 <= float(row["tip_x"]) <= 256 and 0 <= float(row["tip_y"]) <= 256
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
    cfgp.write_text("ticks:\n  len_range: [9, 6]\n")
    out = tmp_path / "o"
    r = run_cli("--config", cfgp, "--output-dir", out, "--num-images", 2)
    assert r.returncode == 2
    assert "ticks.len_range" in r.stderr + r.stdout


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
