"""Tests for train_v9_tip.py (cached-tensor tip trainer + tipnet).

CPU-friendly; the CLI smoke uses CUDA when present.  What is covered:

    split      split_indices == torch.utils.data.random_split (seed 42)
    cache      PNG -> uint8 .npy cache round trip, stale-cache rebuild, --limit
    augment    dihedral_images / dihedral_points agree for all 8 transforms
               (one bright pixel, its argmax must land on the transformed label)
    decode     sub-pixel Gaussian peaks decode to < 0.15 image px, also at the border
    model      tipnet params <= 115k, best_p99.pth < 500 KB, ONNX export == ORT
    CLI        a user runs `python train_v9_tip.py ...` on a 96-image v9 dataset
    TipNetX    tipnet_d2x / tipnet_m32d2x presets (param counts + key/shape fingerprints),
               --max-pth-kb, --mmap-cache, --decode-win, --hard-weight (tiny tmp_path datasets)
"""

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


# ── TipNetX presets, --max-pth-kb, --mmap-cache, --decode-win, --hard-weight ──

def _fingerprint(model):
    """sha256 hex (str) of every state_dict key + shape, sorted by key."""
    text = "\n".join(f"{k} {tuple(t.shape)}" for k, t in sorted(model.state_dict().items()))
    return hashlib.sha256(text.encode()).hexdigest()


def _run_cli(args, timeout=900):
    """Run `python train_v9_tip.py <args>` from the repo root -> CompletedProcess (text output)."""
    return subprocess.run([sys.executable, "train_v9_tip.py", *map(str, args)],
                          cwd=REPO, capture_output=True, text=True, timeout=timeout)


def _argline(text):
    """The 'args: k=v ...' line (str) of a train.log / stdout."""
    return next(ln for ln in text.splitlines() if ln.startswith("args: "))


def test_cli_tipnetx_with_every_new_flag(tmp_path):
    data, out = tmp_path / "ds", tmp_path / "run"       # Path: 16-image dataset, --out
    _write_pngs(data, 16)
    proc = _run_cli(["--data-dir", data, "--arch", "tipnet_d2x", "--max-pth-kb", "0",
                     "--mmap-cache", "--decode-win", "7", "--hard-weight", "1",
                     "--epochs", "1", "--batch-size", "8", "--cpu", "--out", out])
    log = proc.stdout[-5000:] + "\n" + proc.stderr[-3000:]
    assert proc.returncode == 0, log
    with open(out / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and np.isfinite(float(rows[0]["p99"])), log
    args_line = _argline((out / "train.log").read_text(encoding="utf-8"))
    for want in ("arch=tipnet_d2x", "mmap_cache=True", "decode_win=7", "hard_weight=1.0"):
        assert want in args_line, args_line
    best = out / "best_p99.pth"
    assert os.path.getsize(best) > tv.PTH_LIMIT_BYTES
    m = tv.build_model("tipnet_d2x")
    m.load_state_dict(torch.load(best, map_location="cpu", weights_only=True))
    assert "-> OK" in next(ln for ln in proc.stdout.splitlines() if ln.startswith("ONNX:")), log


def test_cli_big_arch_refused_without_max_pth_kb(tmp_path):
    data, out = tmp_path / "ds", tmp_path / "run"
    _write_pngs(data, 16)
    proc = _run_cli(["--data-dir", data, "--arch", "tipnet_d2x", "--epochs", "1",
                     "--batch-size", "8", "--cpu", "--out", out])
    log = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 2, log
    refused = next((ln for ln in log.splitlines() if ln.startswith("REFUSED")), "")
    assert "KB" in refused and "--max-pth-kb" in refused, log
    rows = []
    if (out / "metrics.csv").exists():
        with open(out / "metrics.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    assert rows == []


@pytest.mark.parametrize("bad", ["4", "0"])
def test_cli_decode_win_must_be_odd_and_positive(bad):
    proc = _run_cli(["--decode-win", bad])
    assert proc.returncode == 2
    assert "--decode-win" in proc.stderr and "odd" in proc.stderr, proc.stderr


def test_tipnetx_presets_match_the_trained_prototype():
    want = {"tipnet_d2x": (546_993, "bb6423b5b196da904653e0df6e6c06cffd61628feec15d6991cf968a626f1e01"),
            "tipnet_m32d2x": (556_273, "b5517d17184c160ffe00fdaf7727b6ff77a105944ed851902ff1c8b7e4622116")}
    for arch, (n_params, digest) in want.items():
        m = tv.build_model(arch)
        assert tv.count_params(m) == n_params, arch
        assert _fingerprint(m) == digest, arch
        with torch.no_grad():
            assert m.eval()(torch.rand(2, 1, 256, 256)).shape == (2, 1, 128, 128)
    # the old archs keep their keys: tipnet is still the 100k model
    assert tv.count_params(tv.build_model("tipnet")) <= tv.PARAM_BUDGET


def test_tipnetx_onnx_export_matches_ort(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    m = tv.build_model("tipnet_m32d2x").eval()
    ok, diff, _ = tv.export_and_check_onnx(m, str(tmp_path / "x.onnx"), torch.rand(2, 1, 256, 256))
    assert ok and diff < 1e-3, diff


def test_load_cache_mmap_is_a_view_of_the_npy(tmp_path):
    data = tmp_path / "ds"
    imgs, labels = _write_pngs(data, 6)
    prefix = str(tmp_path / "ds_cache")
    ram = tv.load_cache(str(data), prefix)
    mm = tv.load_cache(str(data), prefix, mmap=True)
    assert mm[0].dtype == torch.uint8 and mm[0].shape == (6, 256, 256)
    assert np.array_equal(mm[0].numpy(), ram[0].numpy())
    assert torch.equal(mm[1], ram[1]) and mm[2] == ram[2]
    # backed by the file, not a copy: a byte written to the .npy shows in the tensor
    img_path, _ = tv.cache_paths(prefix)
    writer = np.load(img_path, mmap_mode="r+")
    writer[2, 5, 7] = 255 - int(imgs[2, 5, 7])
    writer.flush()
    assert int(mm[0][2, 5, 7]) == int(writer[2, 5, 7]) != int(ram[0][2, 5, 7])
    del writer
    first3 = tv.load_cache(str(data), prefix, limit=3, mmap=True)
    assert first3[0].shape == (3, 256, 256) and len(first3[2]) == 3 and first3[1].shape == (3, 2)
    assert np.array_equal(first3[0].numpy()[:2], mm[0].numpy()[:2])


def test_batchfeeder_runs_on_a_mmap_cache(tmp_path):
    data = tmp_path / "ds"
    imgs, labels = _write_pngs(data, 6)
    x, y, _ = tv.load_cache(str(data), str(tmp_path / "ds_cache"), mmap=True)
    feeder = tv.BatchFeeder(x, y, torch.device("cpu"), 4)
    got = list(feeder.batches(torch.tensor([5, 1, 3, 0, 2])))
    assert [b[0].shape[0] for b in got] == [4, 1]
    assert np.array_equal(got[0][0].numpy(), imgs[[5, 1, 3, 0]])
    assert np.allclose(got[1][1].numpy(), labels[[2]])


class _FixedLogits(torch.nn.Module):
    """Stub model: ignores the image, returns the same (1, 1, 128, 128) logit map for every sample."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits                              # (1, 1, 128, 128) float32

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1, -1, -1)


def test_evaluate_uses_decode_win():
    # a peak at (40, 40) plus a lower shoulder 4 cells to its right: invisible to a 5x5
    # window (+-2), inside a 9x9 window (+-4)
    uv = torch.tensor([[40.0, 40.0]])
    g = torch.arange(tv.HM_SIZE, dtype=torch.float32)
    shoulder = -(((g[None, None, :] - 44.0) ** 2 + (g[None, :, None] - 40.0) ** 2) / 8.0)
    logits = _peak_logits(uv, sigma=3.0) + 2.0 * shoulder.exp()[:, None] - 0.5
    images = torch.zeros(2, 256, 256, dtype=torch.uint8)
    labels = torch.tensor([[80.5, 80.5], [10.0, 20.0]])
    feeder = tv.BatchFeeder(images, labels, torch.device("cpu"), 2)
    idx = torch.tensor([0, 1])
    errs = {}
    for win in (5, 9):
        args = SimpleNamespace(loss="ce", sigma=1.0, coord_weight=0.1, decode_win=win)
        res = tv.evaluate(_FixedLogits(logits), feeder, idx, args, amp=False)
        want = (tv.decode_logits(logits.expand(2, -1, -1, -1), win=win) - labels).norm(dim=1)
        assert np.allclose(res["err"], want.numpy(), atol=1e-5), win
        errs[win] = res["err"]
    assert not np.allclose(errs[5], errs[9], atol=1e-3)


def test_tip_loss_hard_weight():
    xy = torch.tensor([[100.3, 60.8], [30.0, 200.0]])
    easy_hard = _peak_logits(tv.image_to_heatmap(xy + torch.tensor([[0.0, 0.0], [14.0, 14.0]])))
    old = tv.tip_loss(easy_hard, xy)
    assert torch.equal(tv.tip_loss(easy_hard, xy, hard_weight=0.0), old)      # G=0: bit-identical
    assert torch.equal(tv.tip_loss(easy_hard, xy, "ce", 1.0, 0.1, 0.0, 10.0), old)

    def share(g):   # fraction of the gradient norm that lands on the hard sample (index 1)
        lg = easy_hard.clone().requires_grad_(True)
        tv.tip_loss(lg, xy, hard_weight=g).backward()
        n = lg.grad.flatten(1).norm(dim=1)
        return float(n[1] / n.sum())
    assert share(1.0) > share(0.0) + 0.01
    assert tv.tip_loss(easy_hard, xy, hard_weight=1.0) != old

    # weights: batch mean 1, the harder sample larger, clamped at hard_weight_max
    kl = torch.tensor([0.0, 1.0, 5.0])
    w = tv.hard_sample_weights(kl, 1.0, 10.0)
    assert float(w.mean()) == pytest.approx(1.0) and w[0] < w[1] < w[2]
    kl = torch.tensor([1.0, 1.0, 10.0])               # ratios 0.25, 0.25, 2.5 -> clamp 2 -> /mean
    w = tv.hard_sample_weights(kl, 1.0, 2.0)
    assert torch.allclose(w, torch.tensor([0.25, 0.25, 2.0]) / (2.5 / 3))
    assert torch.allclose(tv.hard_sample_weights(kl, 0.0, 2.0), torch.ones(3))


# ── --ws-cap-gb: hard working-set cap for a memmap run (Windows) ──────────

def _ws_limits():
    """(min_bytes, max_bytes, flags) of this process's working-set quota (ints), via ctypes."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(wintypes.DWORD)]
    lo, hi, fl = ctypes.c_size_t(), ctypes.c_size_t(), wintypes.DWORD()
    assert k32.GetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), ctypes.byref(lo),
                                          ctypes.byref(hi), ctypes.byref(fl))
    return lo.value, hi.value, fl.value


def _ws_release():
    """Drop the hard working-set max set by a test (QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x8)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t,
                                               ctypes.c_size_t, wintypes.DWORD]
    k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), 256 << 20, 1 << 40, 0x2 | 0x8)


def test_cli_ws_cap_with_mmap_cache(tmp_path):
    data, out = tmp_path / "ds", tmp_path / "run"
    _write_pngs(data, 16)
    proc = _run_cli(["--data-dir", data, "--mmap-cache", "--ws-cap-gb", "2", "--epochs", "1",
                     "--batch-size", "8", "--cpu", "--out", out])
    log = proc.stdout[-4000:] + "\n" + proc.stderr[-3000:]
    assert proc.returncode == 0, log
    with open(out / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and np.isfinite(float(rows[0]["p99"])), log
    want = "ok" if sys.platform == "win32" else "unavailable"
    assert f"[mem] working-set cap 2 GB -> {want}" in proc.stdout, log


def test_cli_ws_cap_needs_mmap_cache():
    proc = _run_cli(["--ws-cap-gb", "2"])
    assert proc.returncode == 2
    err = proc.stderr.splitlines()[-1]               # the argparse "error: ..." line
    assert "--ws-cap-gb" in err and "--mmap-cache" in err and "requires" in err, proc.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set API")
def test_set_working_set_cap_sets_a_hard_max():
    try:
        assert tv.set_working_set_cap(64) is True
        lo, hi, fl = _ws_limits()
        assert hi == 64 * 2**30
        assert fl & 0x4                      # QUOTA_LIMITS_HARDWS_MAX_ENABLE
    finally:
        _ws_release()
