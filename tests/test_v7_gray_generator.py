"""CLI-level tests for generate_synthetic_v7_mp.py (grayscale v2_mp port).

Drives the generator exactly as a user would -- `python
generate_synthetic_v7_mp.py --output-dir ... --num-images ...` -- and then
inspects the files it wrote.
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# REPO_ROOT: repo root containing the generator scripts (pathlib.Path)
REPO_ROOT = Path(__file__).resolve().parents[1]
# V7_SCRIPT: the grayscale generator under test (pathlib.Path)
V7_SCRIPT = REPO_ROOT / "generate_synthetic_v7_mp.py"
# V2_CANDIDATES: places the (possibly untracked) RGB v2_mp generator may live,
#                list[pathlib.Path], first existing one wins
V2_CANDIDATES = [
    REPO_ROOT / "generate_synthetic_v2_mp.py",
    Path("C:/Users/Administrator/PycharmProjects/train_dial/generate_synthetic_v2_mp.py"),
]
NUM_IMAGES = 6   # images per run (int)
SEED = 3         # base RNG seed passed via --seed (int)
IMG_SIZE = 256   # expected output width/height in px (int)


def _run(script, out_dir, workers):
    """Run a generator CLI like a user would; assert it exits cleanly."""
    cmd = [sys.executable, str(script),
           "--output-dir", str(out_dir),
           "--num-images", str(NUM_IMAGES),
           "--workers", str(workers),
           "--seed", str(SEED)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, (
        f"{script.name} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _read_labels(out_dir):
    """Return (header list[str], rows list[dict[str, str]]) of labels.csv."""
    with open(out_dir / "labels.csv", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


@pytest.fixture(scope="module")
def v7_run(tmp_path_factory):
    """v7 output from --workers 2; returns the output dir (pathlib.Path)."""
    out = tmp_path_factory.mktemp("v7") / "out"
    _run(V7_SCRIPT, out, workers=2)
    return out


def test_outputs_are_grayscale_256_with_valid_labels(v7_run):
    header, rows = _read_labels(v7_run)
    assert header == ["filename", "angle_deg", "tip_x", "tip_y"]
    assert len(rows) == NUM_IMAGES
    for row in rows:
        with Image.open(v7_run / row["filename"]) as im:
            assert im.mode == "L"
            assert im.size == (IMG_SIZE, IMG_SIZE)
            arr = np.array(im)  # grayscale pixels, (256, 256) uint8
            assert arr.shape == (IMG_SIZE, IMG_SIZE) and arr.dtype == np.uint8
        assert 0 <= float(row["tip_x"]) < IMG_SIZE
        assert 0 <= float(row["tip_y"]) < IMG_SIZE
        assert 0 <= int(row["angle_deg"]) <= 359


def test_deterministic_across_worker_counts(v7_run, tmp_path):
    out1 = tmp_path / "out_w1"
    _run(V7_SCRIPT, out1, workers=1)
    assert (out1 / "labels.csv").read_bytes() == (v7_run / "labels.csv").read_bytes()
    _, rows = _read_labels(v7_run)
    for row in rows:
        a = np.array(Image.open(v7_run / row["filename"]))
        b = np.array(Image.open(out1 / row["filename"]))
        assert np.array_equal(a, b), row["filename"]


def test_pixel_parity_with_v2_mp_converted_to_L(v7_run, tmp_path):
    v2_script = next((p for p in V2_CANDIDATES if p.exists()), None)
    if v2_script is None:
        pytest.skip("generate_synthetic_v2_mp.py not available")
    out2 = tmp_path / "out_v2"
    _run(v2_script, out2, workers=2)
    assert (out2 / "labels.csv").read_bytes() == (v7_run / "labels.csv").read_bytes()
    _, rows = _read_labels(v7_run)
    for row in rows:
        with Image.open(out2 / row["filename"]) as im2:
            assert im2.mode == "RGB"
            expected = np.array(im2.convert("L"))  # (256, 256) uint8
        actual = np.array(Image.open(v7_run / row["filename"]))  # (256, 256) uint8
        assert np.array_equal(expected, actual), row["filename"]
