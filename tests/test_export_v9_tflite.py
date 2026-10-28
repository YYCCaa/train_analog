"""Tests for export_v9_tflite.py (ONNX -> TFLite export + ONNX-vs-TFLite verification).

CPU only.  What is covered:

    convert   a random-init tipnet -> model.onnx -> convert(): NHWC shapes
              (1, 256, 256, 1) -> (1, 128, 128, 1), TFLite logits == ONNX logits
              (< 1e-3) on random inputs, float16 side file written and smaller
    CLI       a user runs `python export_v9_tflite.py --onnx ... --data-dir <tiny v9
              set> --check all`: exit 0, a final `TFLITE OK` line, the .tflite and
              the _check.json exist, and the JSON agreement numbers are sane
    args      --check accepts N / all / 0 and rejects junk (no conversion needed)

The p99 of a random model is meaningless; only the ONNX-vs-TFLite agreement is checked.
onnx2tf takes ~15 s, so the conversion runs once per module (fixture `exported`).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]  # Path: repo root (cwd for the CLI)
sys.path.insert(0, str(REPO))

pytest.importorskip("onnx2tf")
pytest.importorskip("onnxruntime")

import export_v9_tflite as ev  # noqa: E402
import train_v9_tip as tv  # noqa: E402

# v9 generator config candidates (list[Path]), first existing one wins:
# $TRAIN_V9_CONFIG, the repo-root copy, the main checkout's copy (same lookup as test_train_v9).
V9_CONFIG_CANDIDATES = [Path(p) for p in (
    os.environ.get("TRAIN_V9_CONFIG", ""),
    str(REPO / "v9_config.yaml"),
    "C:/Users/Administrator/PycharmProjects/train_dial/v9_config.yaml",
) if p]
N_DIALS = 40       # int: tiny dataset size -> int(40 * 0.15) = 6 val images
CLI_TIMEOUT_S = 900  # int: seconds allowed for the CLI subprocess (conversion ~15-60 s)


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def onnx_only(tmp_path_factory):
    """Random-init tipnet exported to ONNX (seed 0) and checked; no TFLite conversion.

    Returns dict: dir Path (scratch folder), onnx Path (model.onnx).
    """
    d = tmp_path_factory.mktemp("export_v9")                       # Path: scratch folder
    torch.manual_seed(0)
    model = tv.build_model("tipnet").eval()                        # random weights, fixed seed
    onnx_path = d / "model.onnx"
    samples = torch.rand(2, 1, tv.IMG_SIZE, tv.IMG_SIZE)           # (2, 1, 256, 256) float32
    ok, diff, _ = tv.export_and_check_onnx(model, str(onnx_path), samples)
    assert ok, diff
    return {"dir": d, "onnx": onnx_path}


@pytest.fixture(scope="module")
def exported(onnx_only):
    """ONNX of `onnx_only` -> convert(float16=True), built once for the module.

    Returns dict: dir Path, onnx Path, tflite Path (float32), conv dict (convert() result).
    """
    d, onnx_path = onnx_only["dir"], onnx_only["onnx"]
    conv = ev.convert(str(onnx_path), str(d / "model.tflite"), float16=True)
    return {"dir": d, "onnx": onnx_path, "tflite": d / "model.tflite", "conv": conv}


# ── convert ──────────────────────────────────────────────────────────────

def test_convert_shapes_and_files(exported):
    conv = exported["conv"]
    assert conv["input_shape"] == [1, 256, 256, 1]
    assert conv["output_shape"] == [1, 128, 128, 1]
    f32, f16 = exported["tflite"], Path(conv["float16_path"])
    assert f32.is_file() and conv["tflite_bytes"] == f32.stat().st_size > 10_000
    assert f16.name == "model_float16.tflite" and f16.is_file()
    assert conv["float16_bytes"] < conv["tflite_bytes"]
    assert ev.io_shapes(str(f32)) == ([1, 256, 256, 1], [1, 128, 128, 1])


def test_tflite_logits_match_onnx(exported):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(exported["onnx"]), providers=["CPUExecutionProvider"])
    interp = ev.make_interpreter(str(exported["tflite"]), threads=2)
    in_idx = interp.get_input_details()[0]["index"]                # int: input tensor id
    out_idx = interp.get_output_details()[0]["index"]              # int: output tensor id
    rng = np.random.default_rng(0)
    for _ in range(4):
        x = rng.random((1, 1, 256, 256), dtype=np.float32)         # (1, 1, 256, 256) NCHW
        want = sess.run(["logits"], {"input": x})[0]               # (1, 1, 128, 128)
        interp.set_tensor(in_idx, np.ascontiguousarray(x.transpose(0, 2, 3, 1)))
        interp.invoke()
        got = interp.get_tensor(out_idx).transpose(0, 3, 1, 2)     # (1, 1, 128, 128)
        assert got.shape == want.shape == (1, 1, 128, 128)
        assert np.abs(got - want).max() < 1e-3


# ── args ─────────────────────────────────────────────────────────────────

def test_check_argument_parsing():
    assert ev.parse_args([]).check == 2000
    assert ev.parse_args(["--check", "all"]).check is None
    assert ev.parse_args(["--check", "0"]).check == 0
    assert ev.parse_args(["--check", "37"]).check == 37
    for bad in ("-1", "many"):
        with pytest.raises(SystemExit):
            ev.parse_args(["--check", bad])
    a = ev.parse_args(["--run-dir", "R"])
    assert a.threads == 4 and a.val_split == 0.15 and not a.float16 and a.onnx is None


# ── CLI smoke, the way a user runs it ────────────────────────────────────

def _v9_config():
    for p in V9_CONFIG_CANDIDATES:
        if p.is_file():
            return p
    return None


def test_cli_smoke_tiny_dataset(onnx_only, tmp_path):
    cfg_path = _v9_config()
    if cfg_path is None:
        pytest.skip("v9_config.yaml not found (set TRAIN_V9_CONFIG)")
    import v7_config
    from generate_synthetic_v7_mp import generate_dataset

    data = tmp_path / "data"             # Path: tiny v9 dataset (dial_*.png + labels.csv)
    res = generate_dataset(v7_config.load_config(str(cfg_path)), str(data), N_DIALS,
                           seed=123, workers=2)
    assert res["completed"]
    out = tmp_path / "m.tflite"          # Path: --out
    proc = subprocess.run(
        [sys.executable, "export_v9_tflite.py", "--onnx", str(onnx_only["onnx"]),
         "--out", str(out), "--data-dir", str(data), "--check", "all", "--threads", "2"],
        cwd=REPO, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    log = proc.stdout[-4000:] + "\n" + proc.stderr[-3000:]
    assert proc.returncode == 0, log
    last = proc.stdout.strip().splitlines()[-1]
    assert last.startswith("TFLITE OK p99="), log
    assert "max_dlogit=" in last and "tip_dmax=" in last and "size_kb=" in last
    assert out.is_file()
    check = tmp_path / "m_check.json"
    assert check.is_file()
    with open(check, encoding="utf-8") as f:
        j = json.load(f)
    n_val = tv.split_sizes(N_DIALS, 0.15)[1]                       # int: 6
    assert j["passed"] is True and j["shapes_ok"] is True
    assert j["n_checked"] == j["n_val"] == n_val and j["n_total"] == N_DIALS
    assert j["max_dlogit"] < ev.MAX_DLOGIT_LIMIT
    assert j["tip_dmax_px"] < ev.TIP_DMAX_LIMIT_PX
    assert j["input_shape"] == [1, 256, 256, 1] and j["output_shape"] == [1, 128, 128, 1]
    assert j["tflite_bytes"] == out.stat().st_size
    assert j["tflite_latency_ms"] > 0 and j["onnx_latency_ms"] > 0
    for backend in ("onnx", "tflite"):
        s = j[backend]
        assert set(s) == {"mean", "median", "p95", "p99", "max", "n_over_7px"}
        assert all(np.isfinite(v) for v in s.values())
        assert 0 <= s["median"] <= s["p99"] <= s["max"]
    # a random model gives the same decoded errors on both backends (agreement, not accuracy)
    assert abs(j["onnx"]["mean"] - j["tflite"]["mean"]) < ev.TIP_DMAX_LIMIT_PX


def test_cli_check_zero_only_converts(onnx_only, tmp_path):
    """--check 0 converts, never touches --data-dir, exits 0 with no TFLITE OK line."""
    out = tmp_path / "only.tflite"
    proc = subprocess.run(
        [sys.executable, "export_v9_tflite.py", "--onnx", str(onnx_only["onnx"]),
         "--out", str(out), "--data-dir", str(tmp_path / "does_not_exist"), "--check", "0"],
        cwd=REPO, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    log = proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]
    assert proc.returncode == 0, log
    assert out.is_file() and not (tmp_path / "only_check.json").exists()
    assert "TFLITE OK" not in proc.stdout and "TFLITE CONVERTED" in proc.stdout


def test_cli_missing_onnx_fails(tmp_path):
    proc = subprocess.run(
        [sys.executable, "export_v9_tflite.py", "--onnx", str(tmp_path / "nope.onnx"),
         "--out", str(tmp_path / "x.tflite"), "--check", "0"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1
    assert proc.stdout.strip().splitlines()[-1].startswith("TFLITE FAIL")
