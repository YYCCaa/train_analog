"""Step 1: run TipNet once over the bundle splits and cache per-image features, log-prob maps and image crops.

    bundle images (n, 256, 256) uint8 ─TipNet─▶ logits (n, 128, 128) ─features.map_features─▶ cache/<set>_*.npy

Sets (written to CACHE, the session scratchpad; nothing goes into the bundle or the repo):
    train       20000 clean train-sample images          (err from the bundle meta; decode re-checked)
    val         22500 clean val images                   (err from the bundle meta; decode re-checked)
    trainnoisy  K_NOISY x 20000 train images + a training-style noise mask (cb.noise_mask inside the dial);
                err = |noisy decoded tip - label|  -> extra labelled examples (README rule: allowed)

Per set: <set>_feats.npy (n, F) float32, <set>_err.npy (n,) float64, <set>_tip.npy (n, 2) float64,
         <set>_logp.npy (n, 128, 128) float16 global log-probs clipped at LOGP_MIN,
         <set>_crop.npy (n, CROP, CROP) uint8 image crop centred on the decoded tip (zero padded),
         trainnoisy_src.npy (n,) int64 source train index.

Run from the worktree root:  python research/confidence/learned/extract.py   (about 5 min)
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as ft  # noqa: E402

# BUNDLE: Path, the read-only research bundle (data, model, harness)
BUNDLE = Path(r"C:/Users/ADMINI~1/AppData/Local/Temp/claude/C--Users-Administrator-PycharmProjects-train-dial/"
              r"f292a1d6-f81e-4d2c-b353-1ee00d6e9308/scratchpad/confbundle")
# CACHE: Path, scratchpad folder for the large intermediate arrays (not committed)
CACHE = BUNDLE.parent / "learned_cache"
sys.path.insert(0, str(BUNDLE))
import confbundle as cb  # noqa: E402

# BATCH: int, GPU batch (README: <= 512)
BATCH = 512
# K_NOISY: int, noisy copies per train image
K_NOISY = 4
# LOGP_MIN: float, log-probs are clipped here before the float16 store
LOGP_MIN = -40.0
# CROP: int, side (image px) of the image crop around the decoded tip
CROP = 64
# WORKERS: int, CPU processes that build the noisy images (inside_dial is ~24 ms / image)
WORKERS = 8
# FEATS_ONLY: bool, `extract.py feats` recomputes only the feature / tip files (same noisy images: fixed seeds)
FEATS_ONLY = sys.argv[1:] == ["feats"]

_IMGS = None  # worker-global: train images memmap (20000, 256, 256) uint8


def _init_worker():
    global _IMGS
    _IMGS = cb.load_split("train")["images"]


def _noisy_chunk(job):
    """job (src indices (m,) int64, seed int) -> noisy images (m, 256, 256) uint8 (one fresh mask each)."""
    idx, seed = job
    rng = np.random.default_rng(seed)
    out = np.empty((len(idx), 256, 256), np.uint8)
    for j, i in enumerate(idx):
        img = np.asarray(_IMGS[i])
        out[j] = cb.apply_mask(img, cb.noise_mask(rng), cb.inside_dial(img))
    return out


def crops(images, tip):
    """images (m, 256, 256) uint8, tip (m, 2) float64 image px -> (m, CROP, CROP) uint8 centred crops."""
    h = CROP // 2
    pad = np.pad(images, ((0, 0), (h, h), (h, h)))
    cx = np.clip(np.round(tip[:, 0]).astype(int), 0, 255)
    cy = np.clip(np.round(tip[:, 1]).astype(int), 0, 255)
    return np.stack([pad[k, cy[k]:cy[k] + CROP, cx[k]:cx[k] + CROP] for k in range(len(images))])


class Writer:
    """Preallocated .npy memmaps for one set of n images; put() fills rows s:s+m."""

    def __init__(self, name, n):
        f = len(ft.FEATURE_NAMES)
        mk = lambda key, shape, dt: np.lib.format.open_memmap(CACHE / f"{name}_{key}.npy", "w+", dt, shape)
        self.feats = mk("feats", (n, f), np.float32)          # (n, F) float32
        self.tip = mk("tip", (n, 2), np.float64)              # (n, 2) float64 image px
        # FEATS_ONLY: the maps / crops already cached are left untouched (another job may be reading them)
        self.logp = None if FEATS_ONLY else mk("logp", (n, 128, 128), np.float16)   # (n, 128, 128) float16
        self.crop = None if FEATS_ONLY else mk("crop", (n, CROP, CROP), np.uint8)   # (n, CROP, CROP) uint8

    def put(self, s, images, model):
        lg = torch.from_numpy(cb.logits(model, images, batch=BATCH)).cuda()
        f, aux = ft.map_features(lg)
        bad = (~torch.isfinite(f)).sum(0)
        if bad.any():
            raise ValueError("non-finite features: " + str({ft.FEATURE_NAMES[j]: int(bad[j]) for j in torch.flatnonzero(bad)}))
        m = len(images)
        tip = aux["tip"].cpu().numpy()
        self.feats[s:s + m] = f.cpu().numpy()
        self.tip[s:s + m] = tip
        if not FEATS_ONLY:
            self.logp[s:s + m] = aux["logp"].clamp_min(LOGP_MIN).cpu().numpy().astype(np.float16)
            self.crop[s:s + m] = crops(images, tip)
        return tip

    def flush(self):
        for a in (self.feats, self.tip, self.logp, self.crop):
            if a is not None:
                a.flush()


def run_clean(model, split):
    d = cb.load_split(split)
    n = len(d["err"])
    w = Writer(split, n)
    for s in range(0, n, BATCH):
        w.put(s, np.ascontiguousarray(d["images"][s:s + BATCH]), model)
    w.flush()
    tip_err = np.abs(np.asarray(w.tip) - d["tip"]).max()
    conf_col = np.exp(np.asarray(w.feats[:, ft.FEATURE_NAMES.index("log_m5")], np.float64))
    print(f"{split}: n={n} max |tip - meta tip| = {tip_err:.2e} px, max |m5 - meta conf| = "
          f"{np.abs(conf_col - d['conf']).max():.2e}", flush=True)
    np.save(CACHE / f"{split}_err.npy", d["err"])


def run_noisy(model):
    d = cb.load_split("train")
    n0 = len(d["err"])
    src = np.tile(np.arange(n0), K_NOISY)                    # (K*n0,) int64 source train index
    n = len(src)
    w = Writer("trainnoisy", n)
    chunk = 64
    jobs = [(src[s:s + chunk], 1000 + s) for s in range(0, n, chunk)]
    t0 = time.time()
    buf, pos = [], 0
    with Pool(WORKERS, initializer=_init_worker) as pool:
        for k, imgs in enumerate(pool.imap(_noisy_chunk, jobs)):
            buf.append(imgs)
            if sum(len(b) for b in buf) >= BATCH or k == len(jobs) - 1:
                batch = np.concatenate(buf)
                w.put(pos, batch, model)
                pos += len(batch)
                buf = []
                if (pos // BATCH) % 10 == 0:
                    print(f"  noisy {pos}/{n}  {time.time() - t0:.0f}s", flush=True)
    w.flush()
    err = np.linalg.norm(np.asarray(w.tip) - d["label"][src].astype(np.float64), axis=1)
    np.save(CACHE / "trainnoisy_err.npy", err)
    np.save(CACHE / "trainnoisy_src.npy", src)
    print(f"trainnoisy: n={n} err>3: {(err > 3).sum()}  err>7: {(err > 7).sum()}  "
          f"(clean train: {(d['err'] > 3).sum()} / {(d['err'] > 7).sum()})", flush=True)


def main():
    CACHE.mkdir(exist_ok=True)
    model = cb.load_model()
    t0 = time.time()
    for split in ("train", "val"):
        run_clean(model, split)
        print(f"  {time.time() - t0:.0f}s", flush=True)
    run_noisy(model)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
