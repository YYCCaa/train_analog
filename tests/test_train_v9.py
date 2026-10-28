"""Tests for train_v9_tip.py (cached-tensor tip trainer + tipnet).

CPU-friendly; the CLI smoke uses CUDA when present.  What is covered:

    split      split_indices == torch.utils.data.random_split (seed 42)
    cache      PNG -> uint8 .npy cache round trip, stale-cache rebuild, --limit
    augment    dihedral_images / dihedral_points agree for all 8 transforms
               (one bright pixel, its argmax must land on the transformed label)
    decode     sub-pixel Gaussian peaks decode to < 0.15 image px, also at the border
    model      tipnet params <= 115k, best_p99.pth < 500 KB, ONNX export == ORT
    CLI        a user runs `python train_v9_tip.py ...` on a 96-image v9 dataset
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import random_split

REPO = Path(__file__).resolve().parents[1]  # Path: repo root (cwd for the CLI)
sys.path.insert(0, str(REPO))

import train_v9_tip as tv  # noqa: E402

# v9 generator config candidates (list[Path]), first existing one wins:
# $TRAIN_V9_CONFIG, the repo-root copy, the main checkout's copy named in the brief.
V9_CONFIG_CANDIDATES = [Path(p) for p in (
    os.environ.get("TRAIN_V9_CONFIG", ""),
    str(REPO / "v9_config.yaml"),
    "C:/Users/Administrator/PycharmProjects/train_dial/v9_config.yaml",
) if p]
SMOKE_DIR = REPO / "tmp_smoke" / "test_train_v9"  # Path: CLI smoke scratch (deleted after)


# ── split ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [96, 20000, 150000])
def test_split_matches_random_split(n):
    train_idx, val_idx = tv.split_indices(n, 0.15)
    n_train, n_val = tv.split_sizes(n, 0.15)
    assert n_val == max(1, int(n * 0.15))
    ref_train, ref_val = random_split(range(n), [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    assert train_idx.tolist() == list(ref_train.indices)
    assert val_idx.tolist() == list(ref_val.indices)


# ── cache ────────────────────────────────────────────────────────────────

def _write_pngs(folder, n, seed=0):
    """n random mode-L 256x256 PNGs + labels.csv -> (images uint8 (n,256,256), labels (n,2))."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n, 256, 256), dtype=np.uint8)
    labels = rng.uniform(20, 230, size=(n, 2)).round(2).astype(np.float32)
    with open(folder / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "angle_deg", "tip_x", "tip_y"])
        for i in range(n):
            Image.fromarray(imgs[i]).save(folder / f"dial_{i:04d}.png")  # 2-D uint8 -> mode L
            w.writerow([f"dial_{i:04d}.png", 0, f"{labels[i, 0]:.2f}", f"{labels[i, 1]:.2f}"])
    return imgs, labels


def test_cache_roundtrip_and_rebuild(tmp_path):
    data = tmp_path / "ds"
    imgs, labels = _write_pngs(data, 6)
    prefix = str(tmp_path / "ds_cache")
    x, y, names = tv.load_cache(str(data), prefix)
    assert x.dtype == torch.uint8 and x.shape == (6, 256, 256)
    assert np.array_equal(x.numpy(), imgs)
    assert np.allclose(y.numpy(), labels) and names[0] == "dial_0000.png"
    assert tv.cache_is_valid(str(data), prefix)
    x3, y3, names3 = tv.load_cache(str(data), prefix, limit=3)
    assert x3.shape == (3, 256, 256) and len(names3) == 3 and np.array_equal(x3.numpy(), imgs[:3])
    # labels.csv changes (one more image) -> the cache is stale and gets rebuilt
    imgs2, _ = _write_pngs(data, 7, seed=1)
    assert not tv.cache_is_valid(str(data), prefix)
    x7, _, _ = tv.load_cache(str(data), prefix)
    assert x7.shape[0] == 7 and np.array_equal(x7.numpy(), imgs2)


# ── dihedral augmentation ────────────────────────────────────────────────

POINTS = [(0, 0), (255, 0), (0, 255), (37, 200), (180, 61), (128, 127)]  # (x, y) px


@pytest.mark.parametrize("k", range(8))
def test_dihedral_bright_pixel(k):
    n = len(POINTS)
    img = torch.zeros(n, 1, 256, 256)                       # (n, 1, 256, 256)
    for i, (x, y) in enumerate(POINTS):
        img[i, 0, y, x] = 1.0
    ks = torch.full((n,), k, dtype=torch.long)
    out = tv.dihedral_images(img, ks)
    pts = tv.dihedral_points(torch.tensor(POINTS, dtype=torch.float32), ks)
    for i in range(n):
        flat = int(out[i, 0].argmax())
        assert (flat % 256, flat // 256) == (int(pts[i, 0]), int(pts[i, 1])), (k, POINTS[i])


def test_dihedral_mixed_batch_and_distinct():
    x, y = 37, 200
    img = torch.zeros(8, 256, 256, dtype=torch.uint8)      # uint8, no channel axis
    img[:, y, x] = 255
    ks = torch.arange(8)
    out = tv.dihedral_images(img, ks)
    pts = tv.dihedral_points(torch.tensor([[x, y]] * 8, dtype=torch.float32), ks)
    seen = set()
    for i in range(8):
        flat = int(out[i].argmax())
        got = (flat % 256, flat // 256)
        assert got == (int(pts[i, 0]), int(pts[i, 1]))
        seen.add(got)
    assert len(seen) == 8                                   # all 8 transforms differ
    # sub-pixel labels: the transform is affine, so a fractional point maps exactly
    p = tv.dihedral_points(torch.tensor([[10.25, 3.5]]), torch.tensor([5]))  # flip + rot90
    assert torch.allclose(p, torch.tensor([[3.5, 10.25]]))


# ── decode ───────────────────────────────────────────────────────────────

def _peak_logits(uv, sigma=1.0, size=tv.HM_SIZE):
    """Log-Gaussian logit maps (B, 1, size, size) peaking at uv (B, 2) heatmap px.

    softmax of these logits is the Gaussian of std `sigma` heatmap px, i.e. what a
    model trained on the sigma-1 CE target converges to; +5 is an arbitrary offset.
    """
    g = torch.arange(size, dtype=torch.float32)
    u, v = uv[:, 0, None, None], uv[:, 1, None, None]
    d2 = (g[None, None, :] - u) ** 2 + (g[None, :, None] - v) ** 2
    return (-d2 / (2 * sigma ** 2))[:, None] + 5.0


@pytest.mark.parametrize("sigma", [0.8, 1.0])
def test_decode_subpixel_interior_and_border(sigma):
    rng = np.random.default_rng(0)
    interior = rng.uniform(10, 117, size=(20, 2))
    # heatmap cells run 0..127; >= 1.3 cells from an edge cell the 5x5 window either fits
    # or loses at most one column/row to the -inf padding
    border = np.array([[1.3, 64.2], [2.45, 70.7], [125.4, 30.3], [125.7, 90.1],
                       [50.6, 1.6], [80.2, 124.55], [1.6, 1.45], [125.5, 125.6]])
    uv = torch.tensor(np.concatenate([interior, border]), dtype=torch.float32)
    logits = _peak_logits(uv, sigma)
    pred = tv.decode_logits(logits)                         # (B, 2) image px
    want = tv.heatmap_to_image(uv)
    err = torch.linalg.vector_norm(pred - want, dim=1)
    assert err.max() < 0.15, err


def test_heatmap_image_mapping_roundtrip():
    xy = torch.tensor([[0.5, 0.5], [255.5, 100.0], [37.3, 200.9]])
    assert torch.allclose(tv.heatmap_to_image(tv.image_to_heatmap(xy)), xy)
    assert torch.allclose(tv.image_to_heatmap(torch.tensor([[0.5, 2.5]])), torch.tensor([[0.0, 1.0]]))


def test_loss_prefers_the_label():
    xy = torch.tensor([[100.3, 60.8], [30.0, 200.0]])
    good = _peak_logits(tv.image_to_heatmap(xy))
    bad = _peak_logits(tv.image_to_heatmap(xy + 12.0))
    assert tv.tip_loss(good, xy) < tv.tip_loss(bad, xy)

    def mse_logits(p):  # logits whose sigmoid is the peak-1 Gaussian at p (B, 2) image px
        g = tv.gaussian_target(tv.image_to_heatmap(p), tv.HM_SIZE, tv.HM_SIZE, 1.0, normalise=False)
        return torch.logit(g.clamp(1e-4, 1 - 1e-4))[:, None]
    assert tv.tip_loss(mse_logits(xy), xy, kind="mse") < 1e-6
    assert tv.tip_loss(mse_logits(xy), xy, kind="mse") < tv.tip_loss(mse_logits(xy + 12.0), xy, kind="mse")


# ── model budget + ONNX ──────────────────────────────────────────────────

def test_tipnet_budget_and_shape():
    m = tv.build_model("tipnet", 1.0)
    assert tv.count_params(m) <= tv.PARAM_BUDGET
    assert tv.state_dict_bytes(m) < tv.PTH_LIMIT_BYTES
    with torch.no_grad():
        assert m.eval()(torch.rand(2, 1, 256, 256)).shape == (2, 1, 128, 128)
    with torch.no_grad():   # the 'e' adapter emits logits of the same shape
        assert tv.build_model("e").eval()(torch.rand(1, 1, 256, 256)).shape == (1, 1, 128, 128)


def test_onnx_export_matches_ort(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    m = tv.build_model("tipnet").eval()
    yy, xx = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
    blob = torch.exp(-((xx - 170) ** 2 + (yy - 60) ** 2) / 50.0)          # (256, 256)
    samples = torch.stack([blob, 1 - blob, torch.rand(256, 256)])[:, None]  # (3, 1, 256, 256)
    ok, diff, kb = tv.export_and_check_onnx(m, str(tmp_path / "m.onnx"), samples)
    assert ok, diff
    assert diff < 1e-3 and kb < 500


# ── CLI smoke, the way a user runs it ────────────────────────────────────

def _v9_config():
    for p in V9_CONFIG_CANDIDATES:
        if p.is_file():
            return p
    return None


def test_cli_smoke_two_epochs():
    cfg_path = _v9_config()
    if cfg_path is None:
        pytest.skip("v9_config.yaml not found (set TRAIN_V9_CONFIG)")
    import v7_config
    from generate_synthetic_v7_mp import generate_dataset

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    data = SMOKE_DIR / "data"           # Path: 96-image dataset
    out = SMOKE_DIR / "run"             # Path: --out
    try:
        res = generate_dataset(v7_config.load_config(str(cfg_path)), str(data), 96,
                               seed=123, workers=4)
        assert res["completed"]
        proc = subprocess.run(
            [sys.executable, "train_v9_tip.py", "--data-dir", str(data),
             "--epochs", "2", "--batch-size", "16", "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, timeout=900)
        log = proc.stdout[-6000:] + "\n" + proc.stderr[-4000:]
        assert proc.returncode == 0, log
        with open(out / "metrics.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2, log
        assert list(rows[0].keys()) == tv.METRIC_FIELDS
        assert [int(r["epoch"]) for r in rows] == [1, 2]
        assert all(np.isfinite(float(r["p99"])) for r in rows)
        best = out / "best_p99.pth"
        assert best.is_file() and (out / "model.onnx").is_file()
        assert (out / "last.pth").is_file() and (out / "worst_val.csv").is_file()
        assert (out / "train.log").is_file()
        assert os.path.getsize(best) < tv.PTH_LIMIT_BYTES
        sd = torch.load(best, map_location="cpu", weights_only=True)
        tv.build_model("tipnet").load_state_dict(sd)            # a plain state_dict
        assert "FINAL best_p99=" in proc.stdout, log
    finally:
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)


# ── stale cache after an in-place regeneration ───────────────────────────

def _run_trainer_cli(data, out):
    """Run train_v9_tip.py for 1 epoch on `data` -> (returncode int, stdout+stderr str)."""
    proc = subprocess.run(
        [sys.executable, "train_v9_tip.py", "--data-dir", str(data), "--epochs", "1",
         "--batch-size", "8", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=900)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def test_cli_rebuilds_cache_after_in_place_regeneration(tmp_path):
    data = tmp_path / "ds"                       # Path: tiny 16-image dataset
    prefix = str(data) + "_cache"                # str: the CLI's default cache prefix
    img_path, _ = tv.cache_paths(prefix)         # str: cache images .npy
    imgs, _ = _write_pngs(data, 16)
    rc, log = _run_trainer_cli(data, tmp_path / "run1")
    assert rc == 0, log
    assert "[cache] building" in log
    assert tv.cache_is_valid(str(data), prefix) and not tv.cache_is_stale(str(data), prefix)
    assert np.array_equal(np.load(img_path), imgs)

    # untouched dataset: the next run reuses the cache silently
    rc, log = _run_trainer_cli(data, tmp_path / "run2")
    assert rc == 0, log
    assert "[cache] building" not in log and "labels.csv is newer" not in log

    # regenerate in place: identical labels.csv text, new pixels, labels.csv newer than the cache
    labels_text = (data / "labels.csv").read_bytes()
    new_imgs, _ = _write_pngs(data, 16)          # same seed -> same labels, same names
    rng = np.random.default_rng(99)
    new_imgs = rng.integers(0, 256, size=new_imgs.shape, dtype=np.uint8)
    for i in range(16):
        Image.fromarray(new_imgs[i]).save(data / f"dial_{i:04d}.png")
    assert (data / "labels.csv").read_bytes() == labels_text
    t = os.stat(img_path).st_mtime_ns + 1_000_000      # int ns: 1 ms after the cache (newer, yet older than any rebuild)
    os.utime(data / "labels.csv", ns=(t, t))
    assert tv.cache_is_stale(str(data), prefix) and not tv.cache_is_valid(str(data), prefix)
    rc, log = _run_trainer_cli(data, tmp_path / "run3")
    assert rc == 0, log
    assert "[cache] labels.csv is newer than" in log and "[cache] building" in log
    assert np.array_equal(np.load(img_path), new_imgs)

    # right after the rebuild the cache is fresh again and a further run does not rebuild
    assert tv.cache_is_valid(str(data), prefix) and not tv.cache_is_stale(str(data), prefix)
    rc, log = _run_trainer_cli(data, tmp_path / "run4")
    assert rc == 0, log
    assert "[cache] building" not in log


def test_ensure_cache_reports_labels_dated_in_the_future(tmp_path, capsys):
    data = tmp_path / "ds"                       # Path: tiny 6-image dataset
    prefix = str(data) + "_cache"                # str: cache prefix
    _write_pngs(data, 6)
    tv.ensure_cache(str(data), prefix)
    img_path, _ = tv.cache_paths(prefix)                 # str: cache images .npy
    before = os.stat(img_path).st_mtime_ns               # int ns: cache mtime before the failing call
    capsys.readouterr()                                  # drop the output of the first build
    t = os.stat(data / "labels.csv").st_mtime_ns + 3600 * 10**9   # int ns: one hour ahead
    os.utime(data / "labels.csv", ns=(t, t))
    with pytest.raises(RuntimeError, match="labels.csv is dated in the future"):
        tv.ensure_cache(str(data), prefix)
    # the error comes before any decoding: no rebuild started, the npy is untouched
    assert "[cache] building" not in capsys.readouterr().out
    assert os.stat(img_path).st_mtime_ns == before
