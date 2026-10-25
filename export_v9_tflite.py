"""Verified TFLite export for the v9 tip-localisation model (TipNet).

``train_v9_tip.py`` ends a run with ``<run>/model.onnx`` (opset 17).  This script turns
that file into a float32 ``.tflite`` with onnx2tf and then PROVES the TFLite graph is
the same model: it replays the trainer's exact validation split through ONNX Runtime and
the LiteRT interpreter and compares logits, decoded tips and label errors.  The older
helpers (``train_utils_v2.export_tflite`` / ``convert_onnx_to_tflite.py``) only convert.

Data flow (shapes per image; S = 256 image px, H = W = 128 heatmap cells):

    <run>/model.onnx   input "input"  (1, 1, S, S)  NCHW float32, gray / 255 in [0, 1]
          |                 output "logits" (1, 1, H, W)
          |  convert(): copy to <tmp>/model.onnx (any file stem works), onnx2tf
          v
    <tmp>/model_float32.tflite   NHWC  input (1, S, S, 1)  ->  output (1, H, W, 1)
    <tmp>/model_float16.tflite   (only kept with --float16; ~2x slower on CPU and the
          |                       logits drift by ~0.1, so it is never the default)
          |  copied to --out (default <run>/model.tflite), shapes read back by LiteRT
          v
    verify(): the SAME val images the trainer scored
          <data-dir>/labels.csv --read_labels_csv--> names (N,), labels (N, 2) tip x, y
          split_indices(N, val_split)  (seed 42)  --> val_idx (n_val,)  [:n] or all
          |
          |  per image:  PIL "L" -> float32 / 255 -> x (1, 1, S, S)
          +--> ONNX Runtime (NCHW) ----------------------------> logits_o (1, 1, H, W)
          +--> x.transpose(0,2,3,1) -> LiteRT (NHWC) -> out (1, H, W, 1)
          |        .transpose(0,3,1,2) ----------------------> logits_t (1, 1, H, W)
          |
          |  decode_logits() on both (argmax -> 5x5 centroid -> x = 2u + 0.5, y = 2v + 0.5)
          v  tip_o, tip_t (1, 2) image px
          max |logits_t - logits_o|     (max_dlogit)
          |tip_t - tip_o|               (tip_dmax, px)
          |tip - label| for each backend  -> mean / median / p95 / p99 / max / count > 7 px
          median LiteRT latency per image (ms)
          |
          v   PASS = max_dlogit < 1e-3  and  tip_dmax < 0.05 px
    <out stem>_check.json  (all numbers)   last stdout line: "TFLITE OK ..." / "TFLITE FAIL ..."

Exit code: 0 on OK (or when --check 0 only converts), 1 on FAIL or any error.

Usage:
    python export_v9_tflite.py                         # run-dir v9_tipnet_a, 2000 val images
    python export_v9_tflite.py --check all --float16
    python export_v9_tflite.py --onnx m.onnx --out m.tflite --data-dir ds --check 0
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
from PIL import Image

import train_v9_tip as tv

# ── constants ────────────────────────────────────────────────────────────

IN_SHAPE_TFLITE = [1, tv.IMG_SIZE, tv.IMG_SIZE, 1]    # list[int]: expected TFLite input (NHWC)
OUT_SHAPE_TFLITE = [1, tv.HM_SIZE, tv.HM_SIZE, 1]     # list[int]: expected TFLite output (NHWC)
MAX_DLOGIT_LIMIT = 1e-3   # float: pass if max |tflite logit - onnx logit| is below this
TIP_DMAX_LIMIT_PX = 0.05  # float: pass if max |decoded tflite tip - decoded onnx tip| is below this (px)
DEFAULT_RUN_NAME = "v9_tipnet_a"   # str: run folder under train_v9_tip.DEFAULT_RUNS_DIR
WARMUP_RUNS = 3           # int: untimed TFLite invocations before the latency samples


# ── interpreter ──────────────────────────────────────────────────────────

def interpreter_class():
    """The LiteRT Interpreter class: ai_edge_litert preferred, tf.lite as the fallback."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter


def make_interpreter(tflite_path, threads=1):
    """Allocated TFLite interpreter for tflite_path (str) with `threads` (int) CPU threads."""
    interp = interpreter_class()(model_path=str(tflite_path), num_threads=int(threads))
    interp.allocate_tensors()
    return interp


def io_shapes(tflite_path):
    """(input_shape list[int], output_shape list[int]) of a .tflite, read by the interpreter."""
    interp = make_interpreter(tflite_path)
    return ([int(v) for v in interp.get_input_details()[0]["shape"]],
            [int(v) for v in interp.get_output_details()[0]["shape"]])


# ── convert ──────────────────────────────────────────────────────────────

def convert(onnx_path, out_path, float16=False):
    """ONNX -> float32 TFLite (and optionally float16) with onnx2tf.

    onnx_path: str, the trainer's model.onnx.  out_path: str, where the float32 file goes;
    the float16 file goes next to it as <out stem>_float16.tflite.  onnx2tf names its
    outputs after the ONNX file stem, so the ONNX is first copied to <tmp>/model.onnx and
    everything is built in a TemporaryDirectory that vanishes afterwards.

    Returns dict: tflite_path str, tflite_bytes int, float16_path str | None,
    float16_bytes int | None, input_shape list[int] (NHWC), output_shape list[int] (NHWC),
    convert_seconds float.
    """
    import onnx2tf
    out_path = os.path.abspath(out_path)
    f16_path = os.path.splitext(out_path)[0] + "_float16.tflite" if float16 else None  # str | None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "model.onnx")                  # str: stem fixed to "model"
        shutil.copy2(onnx_path, src)
        onnx2tf.convert(input_onnx_file_path=src, output_folder_path=os.path.join(tmp, "out"),
                        non_verbose=True, output_signaturedefs=True)
        shutil.copy2(os.path.join(tmp, "out", "model_float32.tflite"), out_path)
        if float16:
            shutil.copy2(os.path.join(tmp, "out", "model_float16.tflite"), f16_path)
    in_shape, out_shape = io_shapes(out_path)
    return {"tflite_path": out_path, "tflite_bytes": os.path.getsize(out_path),
            "float16_path": f16_path,
            "float16_bytes": os.path.getsize(f16_path) if float16 else None,
            "input_shape": in_shape, "output_shape": out_shape,
            "convert_seconds": time.time() - t0}


# ── verify ───────────────────────────────────────────────────────────────

def load_gray(path):
    """One PNG -> float32 ndarray (1, 1, S, S) in [0, 1] (gray / 255), exactly the trainer's input."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)      # (S, S) uint8
    if arr.shape != (tv.IMG_SIZE, tv.IMG_SIZE):
        raise ValueError(f"{path}: shape {arr.shape}, expected {(tv.IMG_SIZE, tv.IMG_SIZE)}")
    return (arr.astype(np.float32) / np.float32(255.0))[None, None]


def decode_tip(logits):
    """Logits ndarray (1, 1, H, W) -> decoded tip ndarray (2,) float32 (x, y) in image px."""
    return tv.decode_logits(torch.from_numpy(np.ascontiguousarray(logits)))[0].numpy()


def summarise(err):
    """Error array (M,) px -> tv.err_stats dict (mean, median, p95, p99, max, n_over_7px)."""
    return tv.err_stats(np.asarray(err, dtype=np.float64))


def verify(tflite_path, onnx_path, data_dir, n, val_split, threads):
    """Compare TFLite with ONNX on the trainer's validation split.

    n: int | None, how many val images (first n of split_indices order), None = all.
    Returns dict with n_checked int, n_val int, n_total int, max_dlogit float,
    tip_dmax_px float, worst_dlogit_file str, worst_tip_file str, onnx / tflite error
    stats dicts (mean, median, p95, p99, max, n_over_7px), tflite_latency_ms and
    onnx_latency_ms (median per image, floats), passed bool.
    """
    import onnxruntime as ort
    names, labels = tv.read_labels_csv(data_dir)               # (N,) str, (N, 2) float32
    _, val_idx = tv.split_indices(len(names), val_split)       # (n_val,) int64, seed 42
    take = val_idx if n is None else val_idx[:max(0, int(n))]  # (n,) int64: images to check

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name                        # str: "input"
    onnx_out = sess.get_outputs()[0].name                      # str: "logits"
    interp = make_interpreter(tflite_path, threads)
    in_idx = interp.get_input_details()[0]["index"]            # int: TFLite input tensor id
    out_idx = interp.get_output_details()[0]["index"]          # int: TFLite output tensor id

    max_dlogit, worst_dlogit_file = 0.0, ""   # float, str: largest logit difference + its image
    tip_dmax, worst_tip_file = 0.0, ""        # float, str: largest decoded-tip distance + image
    err_o, err_t = [], []                     # list[float]: |tip - label| per image (px)
    lat_t, lat_o = [], []                     # list[float]: per-image latency (ms)
    for k, i in enumerate(take.tolist()):
        name = names[i]
        x = load_gray(os.path.join(data_dir, name))            # (1, 1, S, S) float32 NCHW
        x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))  # (1, S, S, 1) float32 NHWC
        t0 = time.perf_counter()
        lg_o = sess.run([onnx_out], {onnx_in: x})[0]           # (1, 1, H, W) float32
        lat_o.append((time.perf_counter() - t0) * 1e3)
        if k == 0:
            for _ in range(WARMUP_RUNS):                       # warm caches, not timed
                interp.set_tensor(in_idx, x_nhwc)
                interp.invoke()
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, x_nhwc)
        interp.invoke()
        out = interp.get_tensor(out_idx)                       # (1, H, W, 1) float32 NHWC
        lat_t.append((time.perf_counter() - t0) * 1e3)
        lg_t = np.ascontiguousarray(out.transpose(0, 3, 1, 2))  # (1, 1, H, W) float32 NCHW

        d = float(np.abs(lg_t - lg_o).max())
        if k == 0 or d > max_dlogit:                           # k == 0: name is never left blank
            max_dlogit, worst_dlogit_file = d, name
        tip_o, tip_t = decode_tip(lg_o), decode_tip(lg_t)      # (2,) float32 each, image px
        dt = float(np.linalg.norm(tip_t - tip_o))
        if k == 0 or dt > tip_dmax:
            tip_dmax, worst_tip_file = dt, name
        err_o.append(float(np.linalg.norm(tip_o - labels[i])))
        err_t.append(float(np.linalg.norm(tip_t - labels[i])))

    if not err_t:
        raise ValueError("verify() checked 0 images")
    return {"n_checked": len(err_t), "n_val": int(len(val_idx)), "n_total": len(names),
            "max_dlogit": max_dlogit, "worst_dlogit_file": worst_dlogit_file,
            "tip_dmax_px": tip_dmax, "worst_tip_file": worst_tip_file,
            "onnx": summarise(err_o), "tflite": summarise(err_t),
            "onnx_latency_ms": float(np.median(lat_o)),
            "tflite_latency_ms": float(np.median(lat_t)),
            "passed": bool(max_dlogit < MAX_DLOGIT_LIMIT and tip_dmax < TIP_DMAX_LIMIT_PX)}


# ── CLI ──────────────────────────────────────────────────────────────────

def check_arg(text):
    """--check value: 'all' -> None (every val image), else a non-negative int."""
    if text.strip().lower() == "all":
        return None
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or 'all', got {text!r}")
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export the v9 tip model to TFLite and verify it vs ONNX")
    p.add_argument("--run-dir", default=os.path.join(tv.DEFAULT_RUNS_DIR, DEFAULT_RUN_NAME),
                   help="trainer run folder holding model.onnx (default %(default)s)")
    p.add_argument("--onnx", default=None, help="ONNX file (default <run-dir>/model.onnx)")
    p.add_argument("--out", default=None, help="float32 .tflite to write (default <run-dir>/model.tflite)")
    p.add_argument("--float16", action="store_true", help="also write <out stem>_float16.tflite")
    p.add_argument("--data-dir", default=tv.DEFAULT_DATA_DIR,
                   help="folder with labels.csv + PNGs (default %(default)s)")
    p.add_argument("--check", type=check_arg, default=2000, metavar="N|all",
                   help="val-split images to verify on; 0 = convert only (default 2000)")
    p.add_argument("--val-split", type=float, default=0.15, help="same value the trainer used")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime and LiteRT")
    return p.parse_args(argv)


def fmt_stats(tag, s):
    """One printable line: tag + the err_stats numbers (px)."""
    return (f"  {tag:<7} mean={s['mean']:.3f} median={s['median']:.3f} p95={s['p95']:.3f} "
            f"p99={s['p99']:.3f} max={s['max']:.3f} over7px={s['n_over_7px']}")


def run(args):
    """Convert, then verify unless --check 0.  Returns (exit_code int, last_line str)."""
    onnx_path = args.onnx or os.path.join(args.run_dir, "model.onnx")    # str
    out_path = args.out or os.path.join(args.run_dir, "model.tflite")    # str
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    print(f"[export] onnx  {onnx_path}  ({os.path.getsize(onnx_path) / 1024:.1f} KB)")
    conv = convert(onnx_path, out_path, args.float16)                    # dict, see convert()
    size_kb = conv["tflite_bytes"] / 1024                                # float
    print(f"[export] tflite {conv['tflite_path']}  ({size_kb:.1f} KB)  "
          f"in {tv.fmt_secs(conv['convert_seconds'])}")
    if conv["float16_path"]:
        print(f"[export] float16 {conv['float16_path']}  ({conv['float16_bytes'] / 1024:.1f} KB)")
    print(f"[export] shapes: input {conv['input_shape']} -> output {conv['output_shape']}")
    shapes_ok = (conv["input_shape"] == IN_SHAPE_TFLITE                  # bool
                 and conv["output_shape"] == OUT_SHAPE_TFLITE)
    if not shapes_ok:
        print(f"[export] unexpected shapes, wanted {IN_SHAPE_TFLITE} -> {OUT_SHAPE_TFLITE}")
    if args.check == 0:
        return (0 if shapes_ok else 1,
                f"TFLITE CONVERTED (check skipped) size_kb={size_kb:.1f}" if shapes_ok
                else "TFLITE FAIL unexpected input/output shapes")

    res = verify(conv["tflite_path"], onnx_path, args.data_dir, args.check,
                 args.val_split, args.threads)                           # dict, see verify()
    res.update(onnx_path=os.path.abspath(onnx_path), data_dir=os.path.abspath(args.data_dir),
               val_split=args.val_split, threads=args.threads, shapes_ok=shapes_ok,
               max_dlogit_limit=MAX_DLOGIT_LIMIT, tip_dmax_limit_px=TIP_DMAX_LIMIT_PX,
               size_kb=size_kb, **{k: conv[k] for k in (
                   "tflite_path", "tflite_bytes", "float16_path", "float16_bytes",
                   "input_shape", "output_shape", "convert_seconds")})
    res["passed"] = bool(res["passed"] and shapes_ok)
    check_path = os.path.splitext(conv["tflite_path"])[0] + "_check.json"  # str
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"[verify] {res['n_checked']} of {res['n_val']} val images "
          f"(of {res['n_total']}, val_split={args.val_split}), {args.threads} threads")
    print(fmt_stats("onnx", res["onnx"]))
    print(fmt_stats("tflite", res["tflite"]))
    print(f"  max|dlogit|={res['max_dlogit']:.2e} ({res['worst_dlogit_file']})  "
          f"tip_dmax={res['tip_dmax_px']:.4f}px ({res['worst_tip_file']})")
    print(f"  latency median: tflite {res['tflite_latency_ms']:.2f} ms  "
          f"onnx {res['onnx_latency_ms']:.2f} ms")
    print(f"[verify] wrote {check_path}")
    head = (f"p99={res['tflite']['p99']:.3f}px max_dlogit={res['max_dlogit']:.2e} "
            f"tip_dmax={res['tip_dmax_px']:.4f}px size_kb={size_kb:.1f}")  # str
    return (0, f"TFLITE OK {head}") if res["passed"] else (1, f"TFLITE FAIL {head}")


def main(argv=None):
    args = parse_args(argv)
    try:
        code, last = run(args)
    except Exception as exc:  # report as a FAIL line so scripts can grep the last line
        traceback.print_exc()
        msg = (str(exc).splitlines() or [""])[0]   # str: first line only, keeps the last line single
        code, last = 1, f"TFLITE FAIL error={type(exc).__name__}: {msg}"
    print(last, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
