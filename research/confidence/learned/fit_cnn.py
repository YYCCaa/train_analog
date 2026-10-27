"""Step 3: a tiny CNN on the whole 128x128 log-prob map -> P(err > t px) for the THRESH heads of fit_feats.

    logp (B, 128, 128) float16 cache ─▶ x (B, 2, 128, 128) float32:
        ch0 = (clip(logp, -40) + 20) / 10      the map, background floor included
        ch1 = exp(-d^2 / 32), d = cells from the argmax   (tells the net which peak was decoded)
    x ─conv3x3 s2 ×4 (8,16,16,32 ch, BN, SiLU)─▶ (B, 32, 8, 8) ─[global mean ‖ global max]─▶ (B, 64)
      ─▶ concat argmax-crop branch: logp 17x17 around the argmax ─conv3x3 ×2 (8,16) s2─▶ mean ‖ max (B, 32)
      ─▶ Linear(96, 32) SiLU Linear(32, K)       ~13k params;   score = -logit P(err > 3 px)

Training: BCE on the K ordinal heads, AdamW, dihedral augmentation of the map (the 8 flips / rotations;
an error radius does not care about orientation). Data and protocols exactly as fit_feats.py:
    cv        5-fold grouped CV on the train sample (clean + noisy copies of the fit folds), clean fold-out scores
    final     "trainfit" (train + noisy -> val; train rows fold-out from cv) and
              "xfit" (train + noisy + val fold a -> val fold 1-a; train rows in-sample, mean of the 2 models)
Scores go to CACHE/raw_cnn_<protocol>.npz (val (22500,), train (20000,) float64).

Run from research/confidence/learned:  python fit_cnn.py cv | final | timing
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import eval_conf as ev
from fit_feats import THRESH, HEAD3

DEV = "cuda"
# EPOCHS / BATCH / LR / WD: training schedule (fixed a priori, checked by the train CV only)
EPOCHS, BATCH, LR, WD = 12, 256, 3e-3, 1e-2
# CROP_R: int, radius (cells) of the argmax-centred crop branch (side 2 * CROP_R + 1)
CROP_R = 8


def conv(cin, cout, s):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU())


class MapNet(nn.Module):
    """Tiny two-branch CNN, see the module docstring. forward(x (B,2,128,128), crop (B,1,17,17)) -> (B, K)."""

    def __init__(self, k=len(THRESH)):
        super().__init__()
        self.full = nn.Sequential(conv(2, 8, 2), conv(8, 16, 2), conv(16, 16, 2), conv(16, 32, 2))
        self.crop = nn.Sequential(conv(1, 8, 1), conv(8, 16, 2))
        self.head = nn.Sequential(nn.Linear(96, 32), nn.SiLU(), nn.Linear(32, k))

    def forward(self, x, crop):
        a = self.full(x)
        b = self.crop(crop)
        z = torch.cat([a.mean((2, 3)), a.amax((2, 3)), b.mean((2, 3)), b.amax((2, 3))], 1)
        return self.head(z)


def prep(logp16):
    """logp (B, 128, 128) float16 cuda -> x (B, 2, 128, 128), crop (B, 1, 17, 17) float32 network inputs."""
    lp = logp16.float()
    B = lp.shape[0]
    am = lp.reshape(B, -1).argmax(1)
    iy, ix = am // 128, am % 128
    g = torch.arange(128, device=lp.device)
    d2 = (g[None, :, None] - iy[:, None, None]) ** 2 + (g[None, None, :] - ix[:, None, None]) ** 2
    x = torch.stack([(lp.clamp_min(-40) + 20) / 10, torch.exp(-d2.float() / 32)], 1)
    pad = F.pad(lp, (CROP_R,) * 4, value=-40.0)
    o = torch.arange(-CROP_R, CROP_R + 1, device=lp.device)
    yy = (iy[:, None, None] + o[None, :, None] + CROP_R).expand(B, 2 * CROP_R + 1, 2 * CROP_R + 1)
    xx = (ix[:, None, None] + o[None, None, :] + CROP_R).expand(B, 2 * CROP_R + 1, 2 * CROP_R + 1)
    bb = torch.arange(B, device=lp.device)[:, None, None].expand_as(yy)
    crop = ((pad[bb, yy, xx] + 20) / 10)[:, None]
    return x, crop


def dihedral(t, k):
    """Apply dihedral transform k (0..7) to the last two dims of t."""
    if k & 4:
        t = t.transpose(-1, -2)
    return torch.rot90(t, k & 3, dims=(-2, -1))


def train_net(logp_list, err, seed=0):
    """logp_list: list of (n_i, 128, 128) float16 arrays (memmaps ok); err (n,) -> trained MapNet (cuda, eval)."""
    torch.manual_seed(seed)
    X = torch.cat([torch.from_numpy(np.ascontiguousarray(a)).to(DEV) for a in logp_list])   # (n,128,128) f16
    Y = torch.from_numpy((err[:, None] > THRESH[None]).astype(np.float32)).to(DEV)          # (n, K)
    net = MapNet().to(DEV)
    steps = EPOCHS * ((len(X) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g).to(DEV)
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]
            k = int(torch.randint(8, (1,), generator=g))
            x, crop = prep(dihedral(X[b], k))
            loss = F.binary_cross_entropy_with_logits(net(x, crop), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    del X
    torch.cuda.empty_cache()
    return net.eval()


@torch.no_grad()
def score_net(net, logp, tta=True):
    """logp (n, 128, 128) float16 array -> (n,) float64 score = -logit P(err > 3) (mean over dihedral TTA)."""
    out = np.empty(len(logp))
    for s in range(0, len(logp), 512):
        X = torch.from_numpy(np.ascontiguousarray(logp[s:s + 512])).to(DEV)
        ks = range(8) if tta else [0]
        lg = torch.stack([net(*prep(dihedral(X, k)))[:, HEAD3] for k in ks]).mean(0)
        out[s:s + 512] = -lg.double().cpu().numpy()
    return out


def run_cv(seeds=(0,)):
    tr = C.load_set("train", ("logp", "err"))
    tn = C.load_set("trainnoisy", ("logp", "err"))
    err = tr["err"]
    out = np.zeros(C.N_TRAIN)
    t0 = time.time()
    for f in range(5):
        m_tr, m_tn = C.TRAIN_FOLD != f, C.TRAIN_FOLD[tn["src"]] != f
        lp = [tr["logp"][np.flatnonzero(m_tr)], tn["logp"][np.flatnonzero(m_tn)]]
        e = np.concatenate([err[m_tr], tn["err"][m_tn]])
        for sd in seeds:
            net = train_net(lp, e, seed=sd)
            out[~m_tr] += score_net(net, tr["logp"][np.flatnonzero(~m_tr)]) / len(seeds)
        print(f"  fold {f} {time.time() - t0:.0f}s", flush=True)
    print(f"CNN train CV: AUROC>3 {ev.auroc(out, err > 3):.4f}  AP>3 {ev.average_precision(out, err > 3):.4f}  "
          f"AUROC>2 {ev.auroc(out, err > 2):.4f}", flush=True)
    np.save(C.CACHE / "cnn_train_cv.npy", out)


def run_final(seeds=(0, 1)):
    tr = C.load_set("train", ("logp", "err"))
    tn = C.load_set("trainnoisy", ("logp", "err"))
    va = C.load_set("val", ("logp", "err"))
    t0 = time.time()
    # trainfit: train + noisy -> val
    sv = np.zeros(C.N_VAL)
    for sd in seeds:
        net = train_net([tr["logp"], tn["logp"]], np.concatenate([tr["err"], tn["err"]]), seed=sd)
        sv += score_net(net, va["logp"]) / len(seeds)
    np.savez(C.CACHE / "raw_cnn_trainfit.npz", val=sv, train=np.load(C.CACHE / "cnn_train_cv.npy"))
    print(f"trainfit done {time.time() - t0:.0f}s", flush=True)
    # xfit: train + noisy + val fold a -> val fold 1-a
    sv, st = np.zeros(C.N_VAL), np.zeros(C.N_TRAIN)
    for a in (0, 1):
        ia, ib = np.flatnonzero(C.VAL_FOLD == a), np.flatnonzero(C.VAL_FOLD != a)
        for sd in seeds:
            net = train_net([tr["logp"], tn["logp"], va["logp"][ia]],
                            np.concatenate([tr["err"], tn["err"], va["err"][ia]]), seed=sd)
            sv[ib] += score_net(net, va["logp"][ib]) / len(seeds)
            st += score_net(net, tr["logp"]) / (2 * len(seeds))
        print(f"xfit fold {a} done {time.time() - t0:.0f}s", flush=True)
    np.savez(C.CACHE / "raw_cnn_xfit.npz", val=sv, train=st)
    torch.save(net.state_dict(), C.CACHE / "cnn_last.pth")


def run_timing():
    """CPU ms per image of prep + one forward (no TTA), 1 thread, batch 1."""
    torch.set_num_threads(1)
    global DEV
    net = MapNet().eval()
    print("MapNet params:", sum(p.numel() for p in net.parameters()))
    lp = torch.from_numpy(np.ascontiguousarray(C.load_set("val", ("logp",))["logp"][:200]))
    with torch.no_grad():
        for i in range(20):
            net(*prep(lp[i:i + 1]))
        t0 = time.perf_counter()
        for i in range(200):
            net(*prep(lp[i:i + 1]))
    print(f"CPU {1000 * (time.perf_counter() - t0) / 200:.2f} ms / image (no TTA; x8 with dihedral TTA)")


if __name__ == "__main__":
    {"cv": run_cv, "final": run_final, "timing": run_timing}.get(sys.argv[1] if len(sys.argv) > 1 else "",
                                                                 lambda: print(__doc__))()
