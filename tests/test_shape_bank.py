"""Tests for tools/build_shape_bank.py (SAM shape-bank bake).

No SAM is loaded: the helpers are pure, and the CLI is driven end to end with
its hidden `--fake-generator` flag, the way a user runs it from a terminal.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (pathlib.Path)
sys.path.insert(0, str(REPO_ROOT))
from tools.build_shape_bank import (  # noqa: E402
    FRAME, filter_masks, load_bank, normalise_mask, pack_bank, save_bank,
)

SCRIPT = REPO_ROOT / "tools" / "build_shape_bank.py"  # CLI under test (Path)
CLI_TIMEOUT = 120  # per CLI run timeout, seconds (int)


def _iou(a, b):
    """IoU of two same-shape bool arrays (float)."""
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def _ellipse(hw, cy, cx, ry, rx):
    """Bool (H, W) filled axis-aligned ellipse."""
    yy, xx = np.mgrid[0:hw[0], 0:hw[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _rec(seg, iou=0.9, stab=0.95):
    """SAM-style mask record dict for a bool (H, W) segmentation."""
    return {"segmentation": seg, "area": int(seg.sum()),
            "predicted_iou": iou, "stability_score": stab}


# ---------------------------------------------------------------- normalise_mask

@pytest.mark.parametrize("ry,rx", [(30, 10), (10, 30), (20, 20)])
def test_normalise_mask_aspect_longest_side_topleft(ry, rx):
    m = _ellipse((200, 200), 100, 90, ry, rx)  # (200, 200) bool
    frame, h, w = normalise_mask(m)
    assert frame.shape == (FRAME, FRAME) and frame.dtype == bool
    assert max(h, w) == FRAME and 1 <= min(h, w) <= FRAME
    ys, xs = np.nonzero(m)
    bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1  # tight bbox size (int, int)
    assert abs(h / w - bh / bw) < 0.05 * (bh / bw) + 1.0 / FRAME
    # anchored top-left: content touches row 0 and col 0, nothing outside [:h, :w]
    assert frame[0, :].any() and frame[:, 0].any()
    assert not frame[h:, :].any() and not frame[:, w:].any()
    # shape preserved vs a reference resize of the crop (bilinear then threshold)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ref = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255)
                     .resize((w, h), Image.BILINEAR)) >= 128
    assert _iou(frame[:h, :w], ref) > 0.95


def test_normalise_mask_upscales_small_and_rejects_empty():
    m = np.zeros((50, 50), bool)
    m[10:14, 20:22] = True  # 4x2 block
    frame, h, w = normalise_mask(m)
    assert (h, w) == (128, 64) and frame[:h, :w].all()
    with pytest.raises(ValueError):
        normalise_mask(np.zeros((10, 10), bool))


# ---------------------------------------------------------------- filter_masks

def test_filter_masks_area_rule():
    hw = (100, 100)
    tiny = np.zeros(hw, bool); tiny[50, 50] = True  # 0.0001 < 0.003
    ok = np.zeros(hw, bool); ok[40:60, 40:60] = True  # 0.04
    huge = np.zeros(hw, bool); huge[10:90, 10:90] = True  # 0.64 > 0.35
    out = filter_masks([_rec(tiny), _rec(ok), _rec(huge)], hw, 10)
    assert len(out) == 1 and out[0]["segmentation"] is ok


def test_filter_masks_border_rule():
    hw = (100, 100)
    two = np.zeros(hw, bool); two[0:20, 0:20] = True  # top + left: kept
    three = np.zeros(hw, bool); three[0:100, 0:10] = True  # top+left+bottom: dropped
    three_b = np.zeros(hw, bool); three_b[0:10, 0:100] = True  # top+left+right
    out = filter_masks([_rec(two), _rec(three), _rec(three_b)], hw, 10)
    assert [r["segmentation"] is two for r in out] == [True]


def test_filter_masks_topk_by_iou_times_stability():
    hw = (100, 100)
    recs = []
    for i, (iou, st) in enumerate([(0.9, 0.9), (0.99, 0.99), (0.87, 0.93), (0.95, 0.97)]):
        m = np.zeros(hw, bool); m[10 * i:10 * i + 8, 30:50] = True
        recs.append(_rec(m, iou, st))
    out = filter_masks(recs, hw, 2)
    assert [r["predicted_iou"] for r in out] == [0.99, 0.95]


# ---------------------------------------------------------------- bank io

def _bank(n=5):
    rng = np.random.default_rng(0)
    frames, hws = [], []
    for _ in range(n):
        h, w = (128, int(rng.integers(1, 129))) if rng.random() < 0.5 else (int(rng.integers(1, 129)), 128)
        f = np.zeros((FRAME, FRAME), bool)
        f[:h, :w] = rng.random((h, w)) < 0.5
        frames.append(f); hws.append((h, w))
    meta = {"n_photos": 3, "places_dir": "/x"}
    return frames, hws, list(range(n)), meta


def test_pack_save_load_roundtrip(tmp_path):
    frames, hws, srcs, meta = _bank()
    bank = pack_bank(frames, hws, srcs, meta)
    assert bank["masks"].dtype == np.uint8 and bank["masks"].shape == (5, 2048)
    p = tmp_path / "b.npz"
    save_bank(p, bank)
    got = load_bank(p)
    for k in ("masks", "hw", "src"):
        assert np.array_equal(got[k], bank[k]) and got[k].dtype == bank[k].dtype
    m = json.loads(str(got["meta"]))
    assert m["version"] == 1 and m["frame"] == 128 and m["model"] == "sam_vit_b"
    assert m["n_photos"] == 3 and "created" in m
    for k, f in enumerate(frames):  # contract unpacking recipe
        un = np.unpackbits(got["masks"][k])[:128 * 128].reshape(128, 128).astype(bool)
        assert np.array_equal(un, f)


def test_pack_empty_bank(tmp_path):
    bank = pack_bank([], [], [], {"n_photos": 0, "places_dir": "/x"})
    save_bank(tmp_path / "e.npz", bank)
    assert load_bank(tmp_path / "e.npz")["masks"].shape == (0, 2048)


@pytest.mark.parametrize("key,bad,msg", [
    ("masks", lambda a: a.astype(np.int16), "masks"),
    ("masks", lambda a: a[:, :100], "masks"),
    ("hw", lambda a: a.astype(np.int64), "hw"),
    ("src", lambda a: a[:-1], "src"),
])
def test_load_bank_rejects_bad_dtype_or_shape(tmp_path, key, bad, msg):
    bank = pack_bank(*_bank())
    bank[key] = bad(bank[key])
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, **bank)
    with pytest.raises(ValueError, match=msg):
        load_bank(p)


def test_load_bank_rejects_missing_key(tmp_path):
    bank = pack_bank(*_bank())
    del bank["hw"]
    np.savez_compressed(tmp_path / "m.npz", **bank)
    with pytest.raises(ValueError, match="hw"):
        load_bank(tmp_path / "m.npz")


# ---------------------------------------------------------------- CLI end to end

def _photos(folder, n=5):
    """Write n small random RGB jpg/png photos into nested folders."""
    rng = np.random.default_rng(1)
    for i in range(n):
        sub = folder / ("a" if i % 2 else "b")
        sub.mkdir(parents=True, exist_ok=True)
        ext = "png" if i % 2 else "jpg"
        Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)).save(sub / f"p{i}.{ext}")
    (folder / "notes.txt").write_text("not a photo")


def _cli(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), "--fake-generator", "--device", "cpu", *args],
                       cwd=REPO_ROOT, capture_output=True, text=True, timeout=CLI_TIMEOUT)
    return r


def test_cli_end_to_end_and_resume(tmp_path):
    places = tmp_path / "places"
    _photos(places)
    full = tmp_path / "full.npz"
    r = _cli("--places", str(places), "--out", str(full), "--n-photos", "3")
    assert r.returncode == 0, r.stderr
    bank = load_bank(full)
    meta = json.loads(str(bank["meta"]))
    assert meta["n_photos"] == 3 and len(bank["masks"]) > 0
    assert set(bank["src"].tolist()) <= set(range(5))
    assert not (tmp_path / "full.partial.npz").exists()

    # interrupted run: stops after 2 photos, leaving a checkpoint; --resume finishes it
    part = tmp_path / "res.npz"
    r = _cli("--places", str(places), "--out", str(part), "--n-photos", "3",
             "--checkpoint-every", "1", "--stop-after", "2")
    assert r.returncode != 0 and not part.exists()
    assert (tmp_path / "res.partial.npz").exists()
    prog = json.loads((tmp_path / "res.progress.json").read_text())
    assert prog["next"] == 2
    r = _cli("--places", str(places), "--out", str(part), "--n-photos", "3", "--resume")
    assert r.returncode == 0, r.stderr
    assert "resum" in r.stdout.lower()
    res = load_bank(part)
    for k in ("masks", "hw", "src"):
        assert np.array_equal(res[k], bank[k])
    assert not (tmp_path / "res.partial.npz").exists()


def test_cli_max_masks_cap(tmp_path):
    places = tmp_path / "places"
    _photos(places)
    out = tmp_path / "cap.npz"
    r = _cli("--places", str(places), "--out", str(out), "--n-photos", "5", "--max-masks", "4")
    assert r.returncode == 0, r.stderr
    assert len(load_bank(out)["masks"]) == 4
