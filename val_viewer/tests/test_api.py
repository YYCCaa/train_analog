"""Backend tests: ranking, numbers vs the trainer, PNGs, bad requests, cache, weights, concurrency, CLI.

    fake dataset (conftest) --> ValData --> create_app --> TestClient (HTTP, like the page)
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import train_v9_tip as tv
from val_viewer.tests.conftest import make_model_file, open_data

# REPO_ROOT: Path, repo root (cwd of the CLI subprocesses)
REPO_ROOT = Path(__file__).resolve().parents[2]


def get_list(client, side, n):
    """GET /api/list -> list[dict] entries."""
    r = client.get("/api/list", params={"side": side, "n": n})
    assert r.status_code == 200, r.text
    return r.json()


def png_array(client, url):
    """GET url -> (PIL mode str, ndarray of the decoded PNG)."""
    r = client.get(url)
    assert r.status_code == 200
    im = Image.open(io.BytesIO(r.content))
    return im.mode, np.asarray(im)


def test_lists_sorted(client):
    allerr = client.data.err
    best, worst = get_list(client, "best", 5), get_list(client, "worst", 5)
    assert [e["rank"] for e in best] == [1, 2, 3, 4, 5] == [e["rank"] for e in worst]
    be, we = [e["err"] for e in best], [e["err"] for e in worst]
    assert be == sorted(be) and we == sorted(we, reverse=True)
    assert be[0] == allerr.min() and we[0] == allerr.max()


def test_full_list_and_tie_order(client):
    best, worst = get_list(client, "best", 1000), get_list(client, "worst", 1000)
    for lst in (best, worst):
        assert sorted(e["pos"] for e in lst) == list(range(30))
        order = [e["pos"] for e in lst]
        assert order.index(0) + 1 == order.index(1)  # tie: 0 then 1, adjacent
        assert lst[order.index(0)]["err"] == lst[order.index(1)]["err"]


def test_numbers_match_trainer(client, fake_data):
    data = client.data
    model = tv.build_model("tipnet")
    model.load_state_dict(torch.load(fake_data.run_dir / "tip.pth", weights_only=True))
    model.eval()
    for e in get_list(client, "best", 1000):
        pos, row = e["pos"], fake_data.val_idx[e["pos"]]
        img = torch.from_numpy(data.image(pos)).unsqueeze(0)
        with torch.no_grad():
            lg = model(tv.to_input(img))
        tip = tv.decode_logits(lg)[0].numpy()
        np.testing.assert_allclose(e["tip"], tip, atol=1e-4)
        np.testing.assert_allclose(e["label"], fake_data.labels[row], atol=1e-6)
        assert e["name"] == fake_data.names[row]
        assert abs(e["err"] - np.hypot(*(np.asarray(e["tip"]) - np.asarray(e["label"])))) < 1e-6
        # independent numpy 5x5 clipped softmax mass around the argmax cell
        z = data.logits(pos)
        p = np.exp(z - z.max())
        p /= p.sum()
        iy, ix = np.unravel_index(z.argmax(), z.shape)
        conf = p[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3].sum()
        assert abs(e["conf"] - conf) < 1e-6


def test_bad_requests(client):
    for params in ({"side": "foo", "n": 5}, {"side": "best", "n": 0}, {"side": "best", "n": 1001}):
        assert client.get("/api/list", params=params).status_code == 422
    assert client.get("/api/heatmap/0.png", params={"scale": "foo"}).status_code == 422
    assert client.get("/api/image/30.png").status_code == 404
    assert client.get("/api/heatmap/30.png").status_code == 404


def test_pngs(client, fake_data):
    mode, arr = png_array(client, "/api/image/3.png")
    row = fake_data.val_idx[3]
    ref = np.asarray(Image.open(fake_data.data_dir / fake_data.names[row]))
    assert mode == "L" and arr.shape == (256, 256) and np.array_equal(arr, ref)
    mode, default = png_array(client, "/api/heatmap/3.png")
    _, log = png_array(client, "/api/heatmap/3.png?scale=log")
    _, lin = png_array(client, "/api/heatmap/3.png?scale=linear")
    assert mode == "RGB" and default.shape == (128, 128, 3)
    assert np.array_equal(default, log) and not np.array_equal(log, lin)


def test_heatmap_brightest_at_argmax(client):
    for pos in (0, 5, 11, 20, 29):
        _, rgb = png_array(client, f"/api/heatmap/{pos}.png")
        lum = rgb.astype(float) @ [0.299, 0.587, 0.114]
        z = client.data.logits(pos)
        assert np.unravel_index(lum.argmax(), lum.shape) == np.unravel_index(z.argmax(), z.shape)


def test_rank_cache(fake_data):
    first = open_data(fake_data)
    second = open_data(fake_data)
    assert not first.cache_hit and second.cache_hit
    np.testing.assert_array_equal(first.err, second.err)
    make_model_file(fake_data.run_dir / "tip.pth", 99)
    third = open_data(fake_data)
    assert not third.cache_hit and not np.array_equal(third.err, first.err)
    assert open_data(fake_data).cache_hit
    csv = fake_data.data_dir / "labels.csv"
    lines = csv.read_text().splitlines()
    name, x, y = lines[1].split(",")
    lines[1] = f"{name},{float(x) + 1:.3f},{y}"
    csv.write_text("\n".join(lines) + "\n")
    assert not open_data(fake_data).cache_hit


def test_weights_fallback_and_summary(fake_data):
    (fake_data.run_dir / "tip.pth").rename(fake_data.run_dir / "best_p99.pth")
    assert open_data(fake_data).summary()["weights"] == "best_p99.pth"
    shutil.copy(fake_data.run_dir / "best_p99.pth", fake_data.run_dir / "tip.pth")
    data = open_data(fake_data)
    s = data.summary()
    assert s["weights"] == "tip.pth" and s["run"] == "fake_run" and s["n_val"] == 30
    assert s["n_over_3"] == int((data.err > 3).sum()) and s["n_over_7"] == int((data.err > 7).sum())
    assert s["p99"] == pytest.approx(np.percentile(data.err, 99)) and s["max_err"] == data.err.max()


def test_concurrent_heatmaps(client):
    positions = [i % 10 for i in range(40)]  # list[int] (40,): requested val positions
    with ThreadPoolExecutor(8) as ex:
        resps = list(ex.map(lambda p: (p, client.get(f"/api/heatmap/{p}.png")), positions))
    seen = {}  # dict[int, bytes]: first PNG bytes seen per position
    for p, r in resps:
        assert r.status_code == 200
        assert seen.setdefault(p, r.content) == r.content


def run_cli(*args):
    """Run `python -m val_viewer *args` from the repo root -> CompletedProcess (text output)."""
    return subprocess.run([sys.executable, "-m", "val_viewer", *map(str, args)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)


def assert_error_exit(proc):
    """proc must have exited 2 with exactly one stderr line starting 'error:'."""
    assert proc.returncode == 2, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error:"), proc.stderr


def test_cli(fake_data, tmp_path, capsys):
    common = ["--cache-dir", tmp_path / "cc", "--device", "cpu"]
    assert_error_exit(run_cli("--data-dir", tmp_path / "nope", "--run", fake_data.run_dir, *common))
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    assert_error_exit(run_cli("--data-dir", fake_data.data_dir, "--run", empty_run, *common))
    no_csv = tmp_path / "no_csv"
    no_csv.mkdir()
    assert_error_exit(run_cli("--data-dir", no_csv, "--run", fake_data.run_dir, *common))
    capsys.readouterr()
    open_data(fake_data)
    assert re.search(r"\[rank\] 30/30", capsys.readouterr().out)
    open_data(fake_data)
    assert re.search(r"^\[rank\] cached", capsys.readouterr().out, re.M)
