"""Tests for the AutoCalibDataset."""

import csv
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from autocal_utils import AutoCalibDataset, N_MARKS


def _write_tiny_dataset(tmp_dir, n=3):
    """Write `n` blank 256x256 images and a labels.csv with hand-set values."""
    fieldnames = ["filename", "cx", "cy", "a", "b", "theta", "phi_0"]
    for i in range(N_MARKS):
        fieldnames += [f"mark_{i}_x", f"mark_{i}_y"]
    rows = []
    for k in range(n):
        fname = f"img_{k}.png"
        Image.new("RGB", (256, 256), (k * 20, 0, 0)).save(
            os.path.join(tmp_dir, fname))
        row = {"filename": fname, "cx": 128.0, "cy": 100.0 + k,
               "a": 60.0, "b": 55.0, "theta": 0.1, "phi_0": 0.5}
        for i in range(N_MARKS):
            row[f"mark_{i}_x"] = 100.0 + i
            row[f"mark_{i}_y"] = 200.0 + i
        rows.append(row)
    with open(os.path.join(tmp_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_loads_and_returns_correct_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_dataset(tmp, n=3)
        ds = AutoCalibDataset(tmp)
        assert len(ds) == 3
        img, params_raw, marks_raw, fname = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 256, 256)
        assert img.dtype == torch.float32
        assert (img >= 0).all() and (img <= 1).all()
        assert params_raw.shape == (6,)
        assert marks_raw.shape == (10, 2)
        assert fname == "img_0.png"
        # Marks are written as mark_i_x = 100.0 + i, mark_i_y = 200.0 + i.
        assert marks_raw[0, 0].item() == pytest.approx(100.0)
        assert marks_raw[0, 1].item() == pytest.approx(200.0)
        assert marks_raw[9, 0].item() == pytest.approx(109.0)
        assert marks_raw[9, 1].item() == pytest.approx(209.0)


def test_dataset_param_ordering():
    """Returned params are [cx, cy, a, b, theta, phi_0] in that order."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_dataset(tmp, n=1)
        ds = AutoCalibDataset(tmp)
        _, params_raw, _, _ = ds[0]
        assert params_raw[0].item() == pytest.approx(128.0)
        assert params_raw[1].item() == pytest.approx(100.0)
        assert params_raw[2].item() == pytest.approx(60.0)
        assert params_raw[3].item() == pytest.approx(55.0)
        assert params_raw[4].item() == pytest.approx(0.1)
        assert params_raw[5].item() == pytest.approx(0.5)
