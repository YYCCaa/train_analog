"""Shared fixtures: a fake 200-image dataset + a randomly initialised TipNet run, all in tmp_path.

    fake_data(tmp_path) --> data_dir/ dial_00000.png ... + labels.csv
                            run_dir/tip.pth   (seeded TipNet state_dict)
                            cache_dir/        (val_viewer rank cache, not yet created)
    client(fake_data)   --> TestClient(create_app(ValData(..., device="cpu")))
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import train_v9_tip as tv
from val_viewer.app.data import ValData
from val_viewer.app.main import create_app

# N_IMAGES: int, number of fake dataset rows (val split 0.15 -> 30 val rows)
N_IMAGES = 200


def make_model_file(path, seed: int) -> None:
    """Save a randomly initialised TipNet state_dict (seeded by `seed`, int) to `path`."""
    torch.manual_seed(seed)
    torch.save(tv.build_model("tipnet").state_dict(), path)


def build_fake_data(tmp_path) -> SimpleNamespace:
    """Write the fake dataset + run under tmp_path (Path); returns SimpleNamespace(data_dir: Path, run_dir: Path, cache_dir: Path, val_idx: (30,) int64 ndarray,
    labels: (200, 2) float32 (as read back from labels.csv), names: list[str] (200,))."""
    rng = np.random.default_rng(7)
    np.random.seed(7)
    data_dir = tmp_path / "fake_ds"
    data_dir.mkdir()
    run_dir = tmp_path / "runs" / "fake_run"
    run_dir.mkdir(parents=True)
    cache_dir = tmp_path / "vv_cache"

    val_idx = tv.split_indices(N_IMAGES, 0.15)[1].numpy()  # (30,) int64: dataset row of each val position
    imgs = rng.integers(0, 256, size=(N_IMAGES, 256, 256), dtype=np.uint8)  # (200, 256, 256) uint8
    labels = rng.uniform(5, 250, size=(N_IMAGES, 2)).round(3)  # (200, 2) float64 tip x, y
    imgs[val_idx[1]] = imgs[val_idx[0]]      # val positions 0 and 1: identical image ...
    labels[val_idx[1]] = labels[val_idx[0]]  # ... and label -> exact error tie
    labels[val_idx[2]] = (2.0, 3.0)          # val position 2: tip near the corner
    names = [f"dial_{i:05d}.png" for i in range(N_IMAGES)]  # list[str] (200,)
    for name, img in zip(names, imgs):
        Image.fromarray(img).save(data_dir / name)
    with open(data_dir / "labels.csv", "w", newline="") as f:
        f.write("filename,tip_x,tip_y\n")
        for name, (x, y) in zip(names, labels):
            f.write(f"{name},{x:.3f},{y:.3f}\n")
    make_model_file(run_dir / "tip.pth", 1234)
    _, csv_labels = tv.read_labels_csv(str(data_dir))
    return SimpleNamespace(data_dir=data_dir, run_dir=run_dir, cache_dir=cache_dir,
                           val_idx=val_idx, labels=csv_labels, names=names)


@pytest.fixture
def fake_data(tmp_path):
    """Function-scoped fake dataset: see build_fake_data."""
    return build_fake_data(tmp_path)


def open_data(fd, **kw) -> ValData:
    """ValData on the fake dataset (fd: fake_data namespace), cpu device; kw overrides ValData kwargs."""
    args = dict(data_dir=fd.data_dir, run_dir=fd.run_dir, cache_dir=fd.cache_dir, device="cpu")
    args.update(kw)
    return ValData(**args)


@pytest.fixture
def client(fake_data):
    """TestClient on a fresh app over the fake dataset (cpu); `.data` is the ValData behind it."""
    data = open_data(fake_data)
    c = TestClient(create_app(data))
    c.data = data
    return c
