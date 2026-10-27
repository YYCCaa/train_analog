"""Run TipNet v9 once over a confidence-bundle split and keep the raw logit maps on disk (one pass per image).

    bundle <split>_images.npy  uint8 (n, 256, 256) memmap
        |  confbundle.logits(model, batch of <= 512)     CUDA, full float32, TF32 off
        v
    <out-dir>/logits_<split>.npy  float32 (n, 128, 128) .npy (memmap-able; ~1.4 GB per split)

The logits are the only input of every single-pass score (scores.py). The file is a cache, so it lives
outside the repo (default: the session scratchpad). A sanity check recomputes the bundle's `conf`
(5x5 global-softmax mass around the argmax) from the dumped logits and prints the largest deviation.

Usage (from the worktree root):
    python research/confidence/single_pass/dump_logits.py --bundle <bundle dir> --out-dir <cache dir>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="confidence bundle folder (read-only)")
    ap.add_argument("--out-dir", required=True, help="where logits_<split>.npy are written (outside the repo)")
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--batch", type=int, default=512, help="GPU batch, <= 512 (shared GPU)")
    args = ap.parse_args()

    sys.path.insert(0, args.bundle)
    import confbundle as cb

    # out_dir: Path, cache folder for the logit maps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = cb.load_model()
    for split in args.splits:
        meta = cb.load_split(split)
        # imgs: (n, 256, 256) uint8 read-only memmap of the split
        imgs = meta["images"]
        n = len(imgs)
        # out: (n, 128, 128) float32 .npy memmap, filled chunk by chunk
        out = np.lib.format.open_memmap(out_dir / f"logits_{split}.npy", mode="w+",
                                        dtype=np.float32, shape=(n, cb.HM_SIZE, cb.HM_SIZE))
        t0 = time.time()
        for s in range(0, n, args.batch):
            out[s:s + args.batch] = cb.logits(model, imgs[s:s + args.batch], batch=args.batch)
        out.flush()
        # Sanity check: recompute the shipped confidence on the first 2000 maps.
        # z: (k, 16384) float64 logits; p: (k, 128, 128) float64 global softmax
        k = min(2000, n)
        z = out[:k].reshape(k, -1).astype(np.float64)
        p = np.exp(z - z.max(1, keepdims=True))
        p = (p / p.sum(1, keepdims=True)).reshape(k, cb.HM_SIZE, cb.HM_SIZE)
        am = z.argmax(1)
        iy, ix = am // cb.HM_SIZE, am % cb.HM_SIZE
        conf = np.array([p[i, max(0, y - 2):y + 3, max(0, x - 2):x + 3].sum() for i, (y, x) in enumerate(zip(iy, ix))])
        print(f"{split}: {n} maps in {time.time() - t0:.1f} s; max |conf - bundle conf| over {k} = "
              f"{np.abs(conf - meta['conf'][:k]).max():.2e}", flush=True)
        del out


if __name__ == "__main__":
    main()
