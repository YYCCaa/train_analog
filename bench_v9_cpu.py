"""Deployment-style CPU check of a v9 tip-model run: ONNX Runtime CPU, fixed batch.

Requirement: CPU inference of a batch of 20 images (1 x 256 x 256 gray) must take < 100 ms in total, and
accuracy is the p99 of |tip - label| (image px) on the val split.  This CLI checks a training run folder
(best_p99.pth + train.log) exactly that way:

    run-dir/best_p99.pth --load--> tv.build_model(arch, width) --export--> model_b<B>.onnx  (fixed batch B)
         |                                                                      |
         '-- first val batch: torch vs ORT logits (< 1e-3) and decoded tips (< 0.05 px) <--'
                                                                                |
    latency : warmup + N timed calls of (sess.run + numpy decode of B tips), first val batch as input
    accuracy: val images in batches of B (last batch zero-padded, padded rows ignored), numpy decode
              -> |tip - label| in image px -> mean / median / p95 / p99 / max / n_over_3px / n_over_7px

    python bench_v9_cpu.py --run-dir <run> [--data-dir D] [--batch 20] [--threads 8] [--limit-ms 100]
                           [--check all|N|0] [--p99-max X]

Writes <run-dir>/cpu_check.json and <run-dir>/model_b<B>.onnx; last stdout line is `CPU OK ...` (exit 0) or
`CPU FAIL <reason> ...` (exit 1).  arch / width default to the last `args: ` line of <run-dir>/train.log.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

import train_v9_tip as tv

HM = tv.HM_SIZE   # int: heatmap side (px)
OVER_3PX = 3.0    # float: error threshold counted as n_over_3px (image px)


# ── numpy decode ─────────────────────────────────────────────────────────

def decode_np(logits, win=5):
    """Logits ndarray (B, 1, h, w) -> tips float32 ndarray (B, 2) = (x, y) in image px.

    Same as tv.decode_logits: argmax -> softmax-weighted centroid of the win x win logits around it (cells outside
    the map are ignored, i.e. the window is clipped at the borders) -> x = 2u + 0.5, y = 2v + 0.5.
    """
    lg = np.asarray(logits, dtype=np.float32)[:, 0]          # (B, h, w)
    b, h, w = lg.shape
    r = win // 2                                             # int: window radius (cells)
    flat = lg.reshape(b, -1).argmax(axis=1)                  # (B,) int64: first max, like torch
    iy, ix = flat // w, flat % w                             # (B,) int64 each
    pad = np.pad(lg, ((0, 0), (r, r), (r, r)), constant_values=-np.inf)   # (B, h+2r, w+2r)
    d = np.arange(-r, r + 1, dtype=np.float64)               # (win,) cell offsets
    out = np.empty((b, 2), dtype=np.float32)
    for i in range(b):
        patch = pad[i, iy[i]:iy[i] + win, ix[i]:ix[i] + win].astype(np.float64)   # (win, win)
        e = np.exp(patch - patch.max())                      # -inf outside the map -> weight 0
        wts = e / e.sum()                                    # (win, win) softmax weights
        v = iy[i] + float((wts.sum(axis=1) * d).sum())       # heatmap row
        u = ix[i] + float((wts.sum(axis=0) * d).sum())       # heatmap col
        out[i] = (2.0 * u + 0.5, 2.0 * v + 0.5)
    return out


# ── helpers ──────────────────────────────────────────────────────────────

def log_defaults(run_dir):
    """Read arch / width from the LAST `args: ` line of train.log.  Returns (arch, width, note str or None)."""
    path = os.path.join(run_dir, "train.log")
    if not os.path.isfile(path):
        return "tipnet", 1.0, "no train.log in the run dir: falling back to tipnet width 1.0"
    last = None   # str | None: last line starting with "args: "
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("args: "):
                last = line
    if last is None:
        return "tipnet", 1.0, "no 'args: ' line in train.log: falling back to tipnet width 1.0"
    ma, mw = re.search(r"\barch=(\S+)", last), re.search(r"\bwidth=(\S+)", last)
    arch = ma.group(1) if ma else "tipnet"
    try:
        width = float(mw.group(1)) if mw else 1.0
    except ValueError:
        width = 1.0
    return arch, width, None


def check_arg(s):
    """--check value: 'all' -> None, else a non-negative int."""
    if s == "all":
        return None
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--check must be 'all' or a non-negative integer, got {s!r}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"--check must be >= 0, got {s}")
    return n


def odd_arg(s):
    n = int(s)
    if n < 1 or n % 2 == 0:
        raise argparse.ArgumentTypeError(f"--decode-win must be odd, got {s}")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ORT CPU fixed-batch latency + val accuracy check of a v9 run")
    p.add_argument("--run-dir", required=True, help="folder with best_p99.pth and train.log")
    p.add_argument("--pth", default=None, help="weights (default <run-dir>/best_p99.pth)")
    p.add_argument("--arch", default=None, help="default: from the last 'args: ' line of train.log")
    p.add_argument("--width", type=float, default=None, help="default: from the last 'args: ' line of train.log")
    p.add_argument("--data-dir", default=tv.DEFAULT_DATA_DIR, help="folder with labels.csv + PNGs")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--limit-ms", type=float, default=100.0, help="latency_ok = p90 < this (ms per batch)")
    p.add_argument("--decode-win", type=odd_arg, default=5)
    p.add_argument("--check", type=check_arg, default=None, help="val images to score: all | N | 0 (latency only)")
    p.add_argument("--p99-max", type=float, default=None, help="FAIL when p99 >= this (image px)")
    p.add_argument("--out", default=None, help="default <run-dir>/cpu_check.json")
    p.add_argument("--onnx-out", default=None, help="default <run-dir>/model_b<batch>.onnx")
    a = p.parse_args(argv)
    if a.batch < 1 or a.threads < 1 or a.runs < 1 or a.warmup < 0:
        p.error("--batch, --threads and --runs must be >= 1 and --warmup >= 0")
    return a


def fail(msg):
    """Print a single failure line and return exit code 1."""
    print(msg.replace("\r", " ").replace("\n", " "))
    return 1


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


# ── main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    a = parse_args(argv)
    run_dir = a.run_dir
    pth = a.pth or os.path.join(run_dir, "best_p99.pth")
    arch, width, note = log_defaults(run_dir)
    arch = a.arch or arch
    width = a.width if a.width is not None else width
    if note and (a.arch is None or a.width is None):
        print("note: " + note)
    onnx_path = a.onnx_out or os.path.join(run_dir, f"model_b{a.batch}.onnx")
    out_path = a.out or os.path.join(run_dir, "cpu_check.json")

    # 1. weights -> torch model (CPU fp32 eval)
    print(f"model: arch {arch} width {width}")
    print(f"loading {pth}")
    try:
        model = tv.build_model(arch, width)
        model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    except Exception as e:  # noqa: BLE001 - any load problem is a one-line user error
        msg = " ".join(str(e).split())[:200]
        return fail(f"CPU FAIL cannot load {pth} into arch={arch} width={width}: {type(e).__name__}: {msg}")
    model = model.cpu().float().eval()
    n_params = tv.count_params(model)   # int: trainable parameters

    # data: val images in split order (only the ones needed)
    try:
        names, xy = tv.read_labels_csv(a.data_dir)
    except Exception as e:  # noqa: BLE001
        return fail(f"CPU FAIL cannot read labels.csv in {a.data_dir}: {type(e).__name__}: {e}")
    val_idx = tv.split_indices(len(names), a.val_split)[1].numpy()   # (n_val,) int64 dataset rows
    n_val = len(val_idx)
    n_check = n_val if a.check is None else min(a.check, n_val)       # int: images scored for accuracy
    n_load = min(n_val, max(a.batch, n_check))                        # int: images actually read
    try:
        imgs = np.stack([tv._load_png(a.data_dir, names[int(i)]) for i in val_idx[:n_load]])   # (n_load,256,256) u8
    except Exception as e:  # noqa: BLE001
        return fail(f"CPU FAIL cannot read val images from {a.data_dir}: {type(e).__name__}: {e}")
    n_pad = -(-n_load // a.batch) * a.batch                           # int: n_load rounded up to a batch multiple
    x_all = np.zeros((n_pad, 1, tv.IMG_SIZE, tv.IMG_SIZE), dtype=np.float32)   # zero-padded inputs in [0, 1]
    x_all[:n_load, 0] = imgs.astype(np.float32) / np.float32(255.0)
    print(f"val: {n_val} images, scoring {n_check}, {n_pad // a.batch} batch(es) of {a.batch}")

    # 2. ONNX with a fixed batch dim + parity with torch on the first batch
    import onnx
    import onnxruntime as ort
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    dummy = torch.zeros(a.batch, 1, tv.IMG_SIZE, tv.IMG_SIZE)
    torch.onnx.export(model, dummy, onnx_path, opset_version=17, input_names=["input"],
                      output_names=["logits"], do_constant_folding=True)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"onnx: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f} KB)")
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    first = np.ascontiguousarray(x_all[:a.batch])                     # (B,1,256,256) float32: first val batch
    with torch.no_grad():
        lg_t = model(torch.from_numpy(first)).numpy()                 # (B,1,128,128)
    lg_o = sess.run(["logits"], {"input": first})[0]                  # (B,1,128,128)
    dlogit = float(np.abs(lg_t - lg_o).max())
    dtip = float(np.abs(decode_np(lg_t, a.decode_win) - decode_np(lg_o, a.decode_win)).max())
    parity_ok = dlogit < 1e-3 and dtip < 0.05
    print(f"onnx vs torch: max|dlogit| {dlogit:.2e}, max tip diff {dtip:.4f} px -> {'ok' if parity_ok else 'MISMATCH'}")

    # 3. latency: sess.run + numpy decode of all B tips
    for _ in range(a.warmup):
        decode_np(sess.run(["logits"], {"input": first})[0], a.decode_win)
    times = []   # list[float]: ms per timed call
    for _ in range(a.runs):
        t0 = time.perf_counter()
        decode_np(sess.run(["logits"], {"input": first})[0], a.decode_win)
        times.append((time.perf_counter() - t0) * 1000.0)
    lat = {"median": pct(times, 50), "p90": pct(times, 90), "min": float(min(times)), "max": float(max(times)),
           "runs": len(times)}
    latency_ok = bool(lat["p90"] < a.limit_ms)
    print(f"latency (batch {a.batch}, {a.threads} threads, {a.runs} runs): median {lat['median']:.2f} ms, "
          f"p90 {lat['p90']:.2f} ms, min {lat['min']:.2f} ms, max {lat['max']:.2f} ms "
          f"-> {'ok' if latency_ok else 'OVER'} (limit {a.limit_ms:g} ms)")

    # 4. accuracy over the val images
    acc = None   # dict | None
    if n_check > 0:
        errs = []   # list[ndarray (<=B,)]: per-batch |tip - label| (image px)
        for s in range(0, n_check, a.batch):
            lg = sess.run(["logits"], {"input": np.ascontiguousarray(x_all[s:s + a.batch])})[0]
            tips = decode_np(lg, a.decode_win)
            m = min(a.batch, n_check - s)    # rows that are real, scored images (rest is padding)
            lab = xy[val_idx[s:s + m]]       # (m, 2) float32
            errs.append(np.linalg.norm(tips[:m] - lab, axis=1))
        err = np.concatenate(errs)
        acc = {"mean": float(err.mean()), "median": pct(err, 50), "p95": pct(err, 95), "p99": pct(err, 99),
               "max": float(err.max()), "n_over_3px": int((err > OVER_3PX).sum()),
               "n_over_7px": int((err > tv.OVER_PX).sum())}
        print(f"accuracy on {len(err)} val images: mean {acc['mean']:.3f} median {acc['median']:.3f} "
              f"p95 {acc['p95']:.3f} p99 {acc['p99']:.3f} max {acc['max']:.3f} px, "
              f">3px {acc['n_over_3px']}, >7px {acc['n_over_7px']}")

    # verdict
    reasons = []   # list[str]: why it failed
    if not parity_ok:
        reasons.append(f"onnx-vs-torch mismatch (dlogit {dlogit:.2e}, tip {dtip:.4f}px)")
    if not latency_ok:
        reasons.append(f"latency p90={lat['p90']:.2f}ms >= {a.limit_ms:g}ms")
    if a.p99_max is not None:
        if acc is None:
            reasons.append(f"p99 not measured (--check 0) but --p99-max {a.p99_max:g}")
        elif not acc["p99"] < a.p99_max:
            reasons.append(f"p99={acc['p99']:.3f}px >= {a.p99_max:g}px")
    passed = not reasons
    result = {"run_dir": run_dir, "pth": pth, "arch": arch, "width": width, "params": n_params, "batch": a.batch,
              "threads": a.threads, "decode_win": a.decode_win, "onnx_path": onnx_path,
              "onnx_vs_torch_max_dlogit": dlogit, "onnx_vs_torch_tip_dmax_px": dtip, "latency_ms": lat,
              "limit_ms": a.limit_ms, "latency_ok": latency_ok, "n_checked": n_check, "n_val": n_val,
              "accuracy": acc, "p99_max": a.p99_max, "passed": passed}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    tail = (f"latency median={lat['median']:.2f}ms p90={lat['p90']:.2f}ms "
            f"(batch {a.batch}, {a.threads} threads)")
    if passed:
        p99 = f"{acc['p99']:.3f}px" if acc else "n/a"
        print(f"CPU OK p99={p99} {tail}")
        return 0
    print(f"CPU FAIL {'; '.join(reasons)} {tail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
