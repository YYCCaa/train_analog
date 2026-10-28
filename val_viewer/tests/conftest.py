"""Shared fixtures: a fake 200-image dataset + a randomly initialised TipNet run, all in tmp_path.

    fake_data(tmp_path) --> data_dir/ dial_00000.png ... + labels.csv
                            run_dir/tip.pth   (seeded TipNet state_dict)
                            cache_dir/        (val_viewer rank cache, not yet created)
    client(fake_data)   --> TestClient(create_app(ValData(..., device="cpu")))
"""

from __future__ import annotations

import json
from pathlib import Path
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


def build_fake_data(tmp_path, n: int = N_IMAGES) -> SimpleNamespace:
    """Write the fake dataset + run under tmp_path (Path) with `n` (int, >= 20 so there are >= 3 val rows) images;
    returns SimpleNamespace(n: int, data_dir: Path, run_dir: Path, cache_dir: Path, val_idx: (30,) int64 ndarray,
    train_idx: (170,) int64 ndarray (dataset row of each train position), labels: (200, 2) float32 (as read back from labels.csv), names: list[str] (200,))."""
    rng = np.random.default_rng(7)
    np.random.seed(7)
    data_dir = tmp_path / "fake_ds"
    data_dir.mkdir()
    run_dir = tmp_path / "runs" / "fake_run"
    run_dir.mkdir(parents=True)
    cache_dir = tmp_path / "vv_cache"

    train_idx, val_idx = (t.numpy() for t in tv.split_indices(n, 0.15))  # (n_train,), (n_val,) int64 dataset rows
    imgs = rng.integers(0, 256, size=(n, 256, 256), dtype=np.uint8)  # (n, 256, 256) uint8
    labels = rng.uniform(5, 250, size=(n, 2)).round(3)  # (n, 2) float64 tip x, y
    imgs[val_idx[1]] = imgs[val_idx[0]]      # val positions 0 and 1: identical image ...
    labels[val_idx[1]] = labels[val_idx[0]]  # ... and label -> exact error tie
    labels[val_idx[2]] = (2.0, 3.0)          # val position 2: tip near the corner
    names = [f"dial_{i:05d}.png" for i in range(n)]  # list[str] (n,)
    for name, img in zip(names, imgs):
        Image.fromarray(img).save(data_dir / name)
    with open(data_dir / "labels.csv", "w", newline="") as f:
        f.write("filename,tip_x,tip_y\n")
        for name, (x, y) in zip(names, labels):
            f.write(f"{name},{x:.3f},{y:.3f}\n")
    make_model_file(run_dir / "tip.pth", 1234)
    _, csv_labels = tv.read_labels_csv(str(data_dir))
    return SimpleNamespace(n=n, data_dir=data_dir, run_dir=run_dir, cache_dir=cache_dir,
                           train_idx=train_idx, val_idx=val_idx, labels=csv_labels, names=names)


def params_record(fd, row: int) -> dict:
    """The fake params.jsonl record (dict) of dataset row `row`: pointer.gray is 100 + row."""
    return {"filename": fd.names[row], "seed": 1000 + row, "angle_deg": row % 360,
            "tip_x": float(fd.labels[row][0]), "tip_y": float(fd.labels[row][1]),
            "sampled": {"pointer": {"gray": 100 + row, "forbidden": [[74, 144]]},
                        "glares": {"items": [{"gray": 200 + row, "x": 5}, {"gray": 10, "x": 6}]},
                        "angle": row % 360}}


def write_params_jsonl(fd, rows=None, mutate=None) -> Path:
    """Write <fd.data_dir>/params.jsonl, one JSON line per dataset row -> its Path.

    rows: iterable[int] | None, the rows that get a line (default all fd.n; the others are left out, so later
    lines shift up); mutate: callable(row: int, rec: dict) -> dict | str | None, may replace a record (dict) or its
    whole line (str), returning None keeps the record."""
    lines = []  # list[str]: one text line per written row
    for row in (range(getattr(fd, "n", N_IMAGES)) if rows is None else rows):
        rec = params_record(fd, row)
        out = mutate(row, rec) if mutate else None
        lines.append(out if isinstance(out, str) else json.dumps(out if out is not None else rec))
    path = fd.data_dir / "params.jsonl"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


@pytest.fixture
def fake_data(tmp_path):
    """Function-scoped fake dataset: see build_fake_data."""
    return build_fake_data(tmp_path)


@pytest.fixture
def small_fake_data(tmp_path):
    """Function-scoped SMALL fake dataset (SimpleNamespace, see build_fake_data) with n=40 images (6 val, 34 train):
    ranking it costs a fraction of the 200-image set. For tests that do not depend on the 200-row layout."""
    return build_fake_data(tmp_path, n=40)


@pytest.fixture
def small_client(small_fake_data):
    """Function-scoped TestClient on a fresh app over small_fake_data (cpu); `.data` is the ValData behind it."""
    data = open_data(small_fake_data)
    c = TestClient(create_app(data))
    c.data = data
    return c


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


@pytest.fixture(scope="module")
def shared_fake_data(tmp_path_factory):
    """Module-scoped fake dataset (SimpleNamespace, see build_fake_data), built once per test file.

    For READ-ONLY tests only: they must not write under data_dir / run_dir / cache_dir."""
    return build_fake_data(tmp_path_factory.mktemp("shared_vv"))


@pytest.fixture(scope="module")
def shared_client(shared_fake_data):
    """Module-scoped TestClient (cpu) over shared_fake_data, ranked once; `.data` is the ValData behind it.

    For READ-ONLY tests only (GET requests, reading `.data`); anything that re-opens, writes or corrupts
    the dataset, run, cache or params.jsonl must use the function-scoped `client` / `fake_data`."""
    data = open_data(shared_fake_data)
    c = TestClient(create_app(data))
    c.data = data
    return c
