"""Tests for bench_v9_cpu.py (batch-20 ONNX Runtime CPU latency + val accuracy check of a v9 run).

CPU only, no real data.  Each CLI test builds, in tmp_path, a 40-image synthetic dataset
(random-noise PNGs + labels.csv; int(40 * 0.15) = 6 val images) and a run dir holding a
random-init tipnet best_p99.pth and a train.log with an `args: arch=tipnet width=1.0 ...` line.

    CLI ok        `bench_v9_cpu.py --run-dir R --data-dir D` -> exit 0, last line `CPU OK`, cpu_check.json
                  with every key, n_checked == n_val == 6, model_b20.onnx with input [20, 1, 256, 256]
    no log        no train.log -> note about the tipnet 1.0 fallback, still `CPU OK`
    bad weights   a pth that does not fit the arch -> exit 1, one line naming arch and pth, no traceback
    limit-ms      `--limit-ms 0.001` -> `CPU FAIL` naming latency, JSON latency_ok / passed false
    p99-max       `--p99-max 0.0001` -> `CPU FAIL` naming p99
    check 0       accuracy null, latency reported, `CPU OK`
    decode        numpy decode == tv.decode_logits (< 1e-4 px) for interior / border / corner peaks, win 3/5/9
    padding       --batch 4 with 6 val images -> all 6 scored, errors == the torch model's errors (< 1e-3 px)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]  # Path: repo root (cwd for the CLI)
sys.path.insert(0, str(REPO))

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import train_v9_tip as tv  # noqa: E402

N_DIALS = 40          # int: synthetic dataset size -> 6 val images
CLI_TIMEOUT_S = 240   # int: seconds allowed for one CLI subprocess
FAST = ["--threads", "2", "--runs", "3", "--warmup", "1"]  # list[str]: keep latency loops short
JSON_KEYS = {"run_dir", "pth", "arch", "width", "params", "batch", "threads", "decode_win", "onnx_path",
             "onnx_vs_torch_max_dlogit", "onnx_vs_torch_tip_dmax_px", "latency_ms", "limit_ms",
             "latency_ok", "n_checked", "n_val", "accuracy", "p99_max", "passed"}  # set[str]


# ── fixtures / helpers ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """40 random-noise 256x256 PNGs + labels.csv (filename,tip_x,tip_y).  Returns Path."""
    d = tmp_path_factory.mktemp("bench_ds")
    rng = np.random.default_rng(0)
    rows = ["filename,tip_x,tip_y"]
    for i in range(N_DIALS):
        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8)).save(d / f"dial_{i:03d}.png")
        x, y = rng.uniform(5, 250, 2)
        rows.append(f"dial_{i:03d}.png,{x:.3f},{y:.3f}")
    (d / "labels.csv").write_text("\n".join(rows) + "\n")
    return d


def make_run(path, arch="tipnet", log=True, model_arch=None, seed=0):
    """Run folder with a random-init best_p99.pth (state_dict of model_arch, default arch) and train.log."""
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save(tv.build_model(model_arch or arch).state_dict(), path / "best_p99.pth")
    if log:
        (path / "train.log").write_text(f"start\nargs: data_dir=x arch={arch} width=1.0 epochs=1\nepoch 1\n")
    return path


def run_cli(run, data, *extra):
    """Run bench_v9_cpu.py like a user.  Returns CompletedProcess (text stdout/stderr, never raises)."""
    return subprocess.run([sys.executable, "bench_v9_cpu.py", "--run-dir", str(run), "--data-dir", str(data), *extra],
                          cwd=str(REPO), capture_output=True, text=True, timeout=CLI_TIMEOUT_S)


def last_line(cp):
    return cp.stdout.strip().splitlines()[-1]


@pytest.fixture(scope="module")
def ok_run(dataset, tmp_path_factory):
    """One default CLI run (batch 20, all val images).  Returns dict: run Path, cp, js dict."""
    run = make_run(tmp_path_factory.mktemp("run_ok"))
    cp = run_cli(run, dataset, *FAST)
    js = json.loads((run / "cpu_check.json").read_text()) if (run / "cpu_check.json").is_file() else None
    return {"run": run, "cp": cp, "js": js}


# ── CLI ──────────────────────────────────────────────────────────────────

def test_cli_ok_writes_json_and_onnx(ok_run):
    cp, run, js = ok_run["cp"], ok_run["run"], ok_run["js"]
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert last_line(cp).startswith("CPU OK p99="), last_line(cp)
    assert "batch 20, 2 threads" in last_line(cp)
    assert js is not None and JSON_KEYS <= set(js)
    assert js["n_checked"] == js["n_val"] == 6
    assert js["latency_ms"]["median"] > 0 and js["latency_ms"]["runs"] == 3
    assert js["latency_ok"] is True and js["passed"] is True
    assert js["arch"] == "tipnet" and js["width"] == 1.0 and js["batch"] == 20
    assert set(js["accuracy"]) == {"mean", "median", "p95", "p99", "max", "n_over_3px", "n_over_7px"}
    assert js["onnx_vs_torch_max_dlogit"] < 1e-3 and js["onnx_vs_torch_tip_dmax_px"] < 0.05
    import onnx
    onnx_path = run / "model_b20.onnx"
    assert onnx_path.is_file()
    dims = onnx.load(str(onnx_path)).graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [20, 1, 256, 256]


def test_cli_no_train_log_falls_back_to_tipnet(dataset, tmp_path):
    run = make_run(tmp_path / "run", log=False)
    cp = run_cli(run, dataset, *FAST)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "tipnet" in cp.stdout and "no train.log" in cp.stdout.lower()
    assert last_line(cp).startswith("CPU OK")


def test_cli_weights_not_fitting_arch(dataset, tmp_path):
    run = make_run(tmp_path / "run", arch="tipnet", model_arch="e")
    cp = run_cli(run, dataset, *FAST)
    assert cp.returncode == 1
    assert "Traceback" not in cp.stdout + cp.stderr
    lines = [ln for ln in (cp.stdout + cp.stderr).splitlines() if ln.strip()]
    err = [ln for ln in lines if "tipnet" in ln and "best_p99.pth" in ln]
    assert len(err) == 1, lines
    assert lines[-1] == err[0]


def test_cli_limit_ms_fails_on_latency(dataset, tmp_path):
    run = make_run(tmp_path / "run")
    cp = run_cli(run, dataset, *FAST, "--limit-ms", "0.001")
    assert cp.returncode == 1
    assert last_line(cp).startswith("CPU FAIL") and "latency" in last_line(cp)
    js = json.loads((run / "cpu_check.json").read_text())
    assert js["latency_ok"] is False and js["passed"] is False


def test_cli_p99_max_fails_on_accuracy(dataset, tmp_path):
    run = make_run(tmp_path / "run")
    cp = run_cli(run, dataset, *FAST, "--p99-max", "0.0001")
    assert cp.returncode == 1
    assert last_line(cp).startswith("CPU FAIL") and "p99" in last_line(cp)
    js = json.loads((run / "cpu_check.json").read_text())
    assert js["passed"] is False and js["p99_max"] == 0.0001


def test_cli_check_zero_is_latency_only(dataset, tmp_path):
    run = make_run(tmp_path / "run")
    cp = run_cli(run, dataset, *FAST, "--check", "0")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert last_line(cp).startswith("CPU OK") and "latency median=" in last_line(cp)
    js = json.loads((run / "cpu_check.json").read_text())
    assert js["accuracy"] is None and js["n_checked"] == 0 and js["latency_ms"]["median"] > 0


# ── numpy decode ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("win", [3, 5, 9])
def test_numpy_decode_matches_torch_decode(win):
    import bench_v9_cpu as bb
    rng = np.random.default_rng(1)
    lg = rng.normal(size=(24, 1, 128, 128)).astype(np.float32)
    peaks = [(64, 64), (0, 0), (0, 127), (127, 0), (127, 127), (0, 60), (127, 33), (50, 0), (90, 127), (1, 1),
             (2, 126), (126, 2)]  # (row, col): interior, borders, corners, one off the corner
    for i, (r, c) in enumerate(peaks):
        lg[i, 0, r, c] += 12.0
        lg[i + 12, 0, r, c] += 6.0
        lg[i + 12, 0, r + (1 if r < 127 else -1), c] += 5.5   # near-tie neighbour so the window matters
    got = bb.decode_np(lg, win)
    want = tv.decode_logits(torch.from_numpy(lg), win).numpy()
    assert got.shape == (24, 2)
    assert np.abs(got - want).max() < 1e-4


# ── padding of the last batch ────────────────────────────────────────────

def test_partial_last_batch_scores_every_val_image(dataset, tmp_path):
    run = make_run(tmp_path / "run")
    cp = run_cli(run, dataset, *FAST, "--batch", "4")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    js = json.loads((run / "cpu_check.json").read_text())
    assert js["n_checked"] == js["n_val"] == 6 and js["batch"] == 4
    assert (run / "model_b4.onnx").is_file()
    # torch reference: the same random-init weights, the same val images, one by one
    model = tv.build_model("tipnet")
    model.load_state_dict(torch.load(run / "best_p99.pth", weights_only=True))
    model.eval()
    names, xy = tv.read_labels_csv(str(dataset))
    val = tv.split_indices(len(names), 0.15)[1].tolist()
    errs = []
    with torch.no_grad():
        for i in val:
            x = torch.from_numpy(np.array(Image.open(dataset / names[i]).convert("L"), dtype=np.uint8))
            pred = tv.decode_logits(model(x[None, None].float() / 255.0))[0]
            errs.append(float(torch.linalg.vector_norm(pred - torch.from_numpy(xy[i]))))
    errs = np.asarray(errs)
    acc = js["accuracy"]
    assert abs(acc["mean"] - errs.mean()) < 1e-3
    assert abs(acc["max"] - errs.max()) < 1e-3
    assert abs(acc["median"] - np.median(errs)) < 1e-3
    assert acc["n_over_3px"] == int((errs > 3.0).sum()) and acc["n_over_7px"] == int((errs > 7.0).sum())
