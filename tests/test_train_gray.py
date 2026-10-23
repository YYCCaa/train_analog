"""End-to-end test of the grayscale Model E trainer, driven through its CLI.

Builds a tiny fake grayscale dataset, runs ``train_e_inv_res_gray.py`` the way
a user would from a terminal, then checks the produced artifacts.
"""

import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]  # Path: repo root (cwd for the CLI)
sys.path.insert(0, str(REPO))

N_IMAGES = 12   # int: fake dataset size
SIZE = 256      # int: image side length in px


def _make_gray_dataset(out_dir: Path) -> Path:
    """Write N_IMAGES mode-"L" 256x256 dial PNGs + labels.csv into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []  # list[tuple(filename:str, angle_deg:float, tip_x:float, tip_y:float)]
    cx = cy = SIZE / 2.0
    for i in range(N_IMAGES):
        angle = i * 360.0 / N_IMAGES                       # float: CW from up, deg
        tip_x = cx + 90 * math.sin(math.radians(angle))    # float: tip px
        tip_y = cy - 90 * math.cos(math.radians(angle))
        img = Image.new("L", (SIZE, SIZE), 30)              # dark background
        d = ImageDraw.Draw(img)
        d.ellipse([20, 20, SIZE - 20, SIZE - 20], fill=180)  # light gray disc
        d.line([cx, cy, tip_x, tip_y], fill=10, width=4)     # dark needle
        d.ellipse([tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5], fill=0)  # tip blob
        fname = f"dial_{i:04d}.png"
        img.save(out_dir / fname)
        rows.append((fname, angle, tip_x, tip_y))
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "angle_deg", "tip_x", "tip_y"])
        w.writerows(rows)
    return out_dir


def test_model_shapes():
    from train_e_inv_res_gray import InvResNetGray
    x = torch.rand(2, 1, SIZE, SIZE)  # (2, 1, 256, 256) float32
    with torch.no_grad():
        assert InvResNetGray(half_res=False).eval()(x).shape == (2, 1, 256, 256)
        assert InvResNetGray(half_res=True).eval()(x).shape == (2, 1, 128, 128)


def test_dataset_item_shape(tmp_path):
    from train_utils_v2_gray import DialDatasetGray
    ds = DialDatasetGray(str(_make_gray_dataset(tmp_path / "ds")))
    img, hm, fname = ds[0]
    assert img.shape == (1, SIZE, SIZE) and img.dtype == torch.float32
    assert hm.shape == (1, SIZE, SIZE)
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_cli_trains_and_exports_gray_onnx(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    ds = _make_gray_dataset(tmp_path / "ds")
    model_base = tmp_path / "m_gray"   # Path: --model-name (artifacts derive from it)
    diag = REPO / "diag" / "m_gray"     # Path: diag dir created relative to cwd
    try:
        proc = subprocess.run(
            [sys.executable, "train_e_inv_res_gray.py",
             "--data-dir", str(ds), "--epochs", "1", "--batch-size", "4",
             "--num-workers", "0", "--model-name", str(model_base)],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        assert proc.returncode == 0, proc.stderr[-4000:]

        pth = tmp_path / "m_gray.pth"
        onnx_path = tmp_path / "m_gray.onnx"
        assert pth.exists()
        assert onnx_path.exists()

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert list(sess.get_inputs()[0].shape) == [1, 1, 256, 256]
        x = np.random.rand(1, 1, SIZE, SIZE).astype(np.float32)  # (1,1,256,256) float32
        heatmap = sess.run(None, {"input": x})[0]                # (1,1,256,256) float32
        assert heatmap.shape == (1, 1, 256, 256)
    finally:
        shutil.rmtree(diag, ignore_errors=True)
        try:
            diag.parent.rmdir()  # drop diag/ only if the test left it empty
        except OSError:
            pass
