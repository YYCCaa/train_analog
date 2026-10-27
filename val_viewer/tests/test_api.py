"""Backend tests: ranking, numbers vs the trainer, PNGs, bad requests, cache, weights, concurrency, CLI.

    fake dataset (conftest) --> ValData --> create_app --> TestClient (HTTP, like the page)
"""

from __future__ import annotations

import io
import re
import shutil
import os
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import train_v9_tip as tv
from val_viewer.app import data as vdata, render
from val_viewer.tests.conftest import make_model_file, open_data  # noqa: E402

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
    # labels.csv edit: the images file (size, mtime) is NOT touched here, so only the key can notice
    data = open_data(fake_data)
    key_args = (data.data_dir, 0.15, data._img_path, data.weights_path, "cpu")
    stat = (os.stat(data._img_path).st_size, os.stat(data._img_path).st_mtime_ns)
    key_before = vdata.rank_key(*key_args)
    csv = fake_data.data_dir / "labels.csv"
    lines = csv.read_text().splitlines()
    name, x, y = lines[1].split(",")
    lines[1] = f"{name},{float(x) + 1:.3f},{y}"
    csv.write_text("\n".join(lines) + "\n")
    assert (os.stat(data._img_path).st_size, os.stat(data._img_path).st_mtime_ns) == stat
    assert vdata.rank_key(*key_args) != key_before


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


def only_cache_file(fd):
    """The single rank_*.npz under fd.cache_dir -> Path (asserts there is exactly one)."""
    files = list(fd.cache_dir.glob("rank_*.npz"))
    assert len(files) == 1, files
    return files[0]


def test_summary_key_changes_with_weights(fake_data):
    """Restart with the same inputs -> same key; retrained weights -> a different key."""
    from fastapi.testclient import TestClient
    from val_viewer.app.main import create_app
    key = lambda: TestClient(create_app(open_data(fake_data))).get("/api/summary").json()["key"]
    k1 = key()
    assert isinstance(k1, str) and k1 and key() == k1
    make_model_file(fake_data.run_dir / "tip.pth", 99)
    assert key() != k1


def test_rank_key_inputs(fake_data, tmp_path):
    """The key reacts to labels.csv bytes (images file size + mtime fixed), device and trainer code."""
    data = open_data(fake_data)
    img = data._img_path
    st = os.stat(img)
    args = dict(data_dir=data.data_dir, val_split=0.15, img_path=img, weights_path=data.weights_path)
    base = vdata.rank_key(device="cpu", **args)
    assert vdata.rank_key(device="cpu", **args) == base
    assert vdata.rank_key(device="cuda", **args) != base
    trainer = tmp_path / "trainer.py"
    trainer.write_text("a = 1\n")
    t1 = vdata.rank_key(device="cpu", trainer_file=trainer, **args)
    trainer.write_text("a = 2\n")
    assert vdata.rank_key(device="cpu", trainer_file=trainer, **args) != t1
    csv = fake_data.data_dir / "labels.csv"
    csv.write_text(csv.read_text().replace("dial_00000.png,", "dial_00000.png, ", 1))
    assert (os.stat(img).st_size, os.stat(img).st_mtime_ns) == (st.st_size, st.st_mtime_ns)
    assert vdata.rank_key(device="cpu", **args) != base


def test_rank_cache_with_wrong_row_is_recomputed(fake_data):
    first = open_data(fake_data)
    f = only_cache_file(fake_data)
    with np.load(f) as z:
        arrays = {k: z[k] for k in z.files}
    assert {"tip", "err", "conf", "row", "label"} <= set(arrays)
    arrays["row"] = arrays["row"][::-1].copy()
    arrays["err"] = arrays["err"] + 100.0
    np.savez(f, **arrays)
    again = open_data(fake_data)
    assert not again.cache_hit
    np.testing.assert_array_equal(again.err, first.err)
    assert open_data(fake_data).cache_hit


def test_corrupt_rank_cache_is_recomputed(fake_data):
    first = open_data(fake_data)
    only_cache_file(fake_data).write_bytes(b"PK" + b"garbage" * 50)
    again = open_data(fake_data)
    assert not again.cache_hit
    np.testing.assert_array_equal(again.err, first.err)
    # a real npz cut in half (starts with PK, numpy raises zipfile.BadZipFile)
    f = only_cache_file(fake_data)
    f.write_bytes(f.read_bytes()[:f.stat().st_size // 2])
    cut = open_data(fake_data)
    assert not cut.cache_hit
    np.testing.assert_array_equal(cut.err, first.err)


def test_render_emits_no_deprecation_warning(client):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        render.dial_png(client.data.image(0))
        render.heatmap_png(client.data.logits(0), "log")
        render.heatmap_png(client.data.logits(0), "linear")
    assert [w for w in rec if issubclass(w.category, DeprecationWarning)] == []


def test_trainer_cache_prefix_is_abspath(fake_data):
    open_data(fake_data)
    prefix = os.path.abspath(fake_data.data_dir) + "_cache"
    assert all(os.path.exists(p) for p in tv.cache_paths(prefix))
