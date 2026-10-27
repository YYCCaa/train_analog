"""Independent re-check of conf-learned: LR3 (4 weights on 3 map features) + isotonic P(err <= 3 px).

Written from scratch in torch from the spec in model_lr3.json (no import of the agent's code):

    logits (128,128) ─▶ f = [ log(m9 / (1 - m9)),  entropy (nats),  log(floor / peak) ]       (3,) float64
         floor = lower median of p over 9x9 local maxima (p >= its 9x9 max in float32, -inf padding)
    z = clip((f - mu) / sd, -8, 8);   score = -(z . w + b)      (higher = more trustworthy)

Checks: max |mine - scores_LR3.npz| on both splits; the frozen harness; paired bootstrap vs conf / m9;
out-of-fold isotonic calibration on val (fold = i % 2) with my own pool-adjacent-violators, ECE over 15 bins.
Run from the repo root; prints only (log via redirect).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# BUNDLE: str, the read-only research bundle (data + meta + the harness in ../harness; not in the repo)
BUNDLE = (r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-PycharmProjects-train-dial"
          r"\f292a1d6-f81e-4d2c-b353-1ee00d6e9308\scratchpad\confbundle")
# HERE: Path, this folder;  AGENT: Path, conf-learned's folder (model_lr3.json, its committed score files)
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent / "learned"
sys.path.insert(0, BUNDLE)
sys.path.insert(0, str(Path.cwd()))
import confbundle as cb  # noqa: E402
import eval_conf as ev  # noqa: E402

DEV = torch.device("cuda")


def lr3_feats(L):
    """L (B,128,128) float64 logits on DEV -> (B,3) float64 [logit m9, entropy, log floor/peak]."""
    B = len(L)
    logp = L - torch.logsumexp(L.reshape(B, -1), 1)[:, None, None]
    p = logp.exp()
    a = L.reshape(B, -1).argmax(1)
    iy, ix = a // 128, a % 128
    yy, xx = torch.arange(128, device=DEV)[None, :, None], torch.arange(128, device=DEV)[None, None, :]
    in9 = ((yy - iy[:, None, None]).abs() <= 4) & ((xx - ix[:, None, None]).abs() <= 4)   # (B,128,128) bool
    inside = (p * in9).sum((1, 2))
    outside = (p * ~in9).sum((1, 2))
    ent = -(p * logp).sum((1, 2))
    pf = p.float()
    is_max = pf >= torch.nn.functional.max_pool2d(pf[:, None], 9, stride=1, padding=4)[:, 0]
    floor = torch.empty(B, dtype=torch.float64, device=DEV)
    for k in range(B):
        h = torch.sort(p[k][is_max[k]]).values
        floor[k] = h[(len(h) - 1) // 2]
    peak = p.reshape(B, -1).gather(1, a[:, None])[:, 0]
    return torch.stack([inside.clamp_min(1e-12).log() - outside.clamp_min(1e-12).log(), ent,
                        (floor + 1e-12).log() - (peak + 1e-12).log()], 1)


def pav(x, y):
    """Isotonic (non-decreasing) fit of y on x by pool-adjacent-violators -> (knot_x, knot_y) for np.interp."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(np.float64)
    blocks = []                                             # list of [sum_y, n, sum_x]
    for xv, yv in zip(xs, ys):
        blocks.append([yv, 1, xv])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s, n, sx = blocks.pop()
            blocks[-1][0] += s; blocks[-1][1] += n; blocks[-1][2] += sx
    return np.array([b[2] / b[1] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


def ece(pred, good, nb=15):
    """Expected calibration error over nb equal-width bins of pred in [0, 1]."""
    idx = np.minimum((pred * nb).astype(int), nb - 1)
    return float(sum(abs(pred[idx == b].mean() - good[idx == b].mean()) * (idx == b).mean()
                     for b in range(nb) if (idx == b).any()))


def main():
    m = json.loads((AGENT / "model_lr3.json").read_text())
    # mu / sd / w: (3,) float64 standardisation and weights of P(err > 3); b: float bias
    mu, sd, w = (torch.tensor(m[k], dtype=torch.float64, device=DEV) for k in ("mu", "sd", "w"))
    b = float(m["b"])
    model = cb.load_model()
    out = {}
    ev.print_header()
    for split in ("train", "val"):
        d = cb.load_split(split)
        s = []
        for st in range(0, len(d["err"]), 512):
            f = lr3_feats(torch.from_numpy(cb.logits(model, d["images"][st:st + 512])).to(DEV).double())
            z = ((f - mu) / sd).clamp(-8, 8)
            s.append((-(z @ w + b)).cpu().numpy())
        out[split] = np.concatenate(s)
        theirs = np.load(AGENT / "scores_LR3.npz")[split]
        print(f"[agree {split}] LR3: max |mine - agent| = {np.abs(out[split] - theirs).max():.2e}")
        ev.evaluate(split, d["conf"], "conf (shipped)")
        ev.evaluate(split, d["m9"], "m9")
        ev.evaluate(split, out[split], "MINE LR3")

    d = cb.load_split("val")
    good, n, rng = d["err"] <= 3.0, len(d["err"]), np.random.default_rng(1)
    for bname in ("conf", "m9"):
        da, dp = [], []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            da.append(ev.auroc(out["val"][i], ~good[i]) - ev.auroc(d[bname][i], ~good[i]))
            dp.append(ev.average_precision(out["val"][i], ~good[i]) - ev.average_precision(d[bname][i], ~good[i]))
        print(f"[boot] LR3 - {bname}: dAUROC3 {np.mean(da):+.4f} [{np.percentile(da, 2.5):+.4f}, "
              f"{np.percentile(da, 97.5):+.4f}]   dAP3 {np.mean(dp):+.4f} [{np.percentile(dp, 2.5):+.4f}, "
              f"{np.percentile(dp, 97.5):+.4f}]")

    # out-of-fold isotonic calibration on val: fit on fold a, predict fold 1-a
    fold = np.arange(n) % 2
    for name, sc in (("conf", d["conf"]), ("m9", d["m9"]), ("LR3", out["val"])):
        pred = np.empty(n)
        for a in (0, 1):
            kx, ky = pav(sc[fold == a], good[fold == a])
            pred[fold != a] = np.interp(sc[fold != a], kx, ky)
        brier = float(((pred - good) ** 2).mean())
        print(f"[calib] iso({name:<4}) OOF: ECE15 {ece(pred, good):.4f}  Brier {brier:.5f}")
    print(f"[calib] raw conf as P:   ECE15 {ece(d['conf'], good):.4f}  Brier {float(((d['conf'] - good) ** 2).mean()):.5f}")
    lo = d["conf"] <= np.quantile(d["conf"], 0.02)
    print(f"[calib] least-confident 2 % by conf: mean conf {d['conf'][lo].mean():.3f}, observed good {good[lo].mean():.3f}")
    np.savez(HERE / "verify_learned.npz", val=out["val"], train=out["train"])


if __name__ == "__main__":
    main()
