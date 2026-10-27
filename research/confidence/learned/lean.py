"""Deployable numpy version of the LR3 confidence + its isotonic calibration (what a webapp would run).

    logits (128, 128) float32 ──lr3_features──▶ f (3,) float64 = [logit(m9), entropy, log(floor / peak)]
        ──model_lr3.json──▶ score = -(z . w + b)  (z = clipped standardised f)   higher = more trustworthy
        ──isotonic_LR3.json──▶ P(err <= 3 px) in [0, 1]                           the number a user reads

    floor = median height of all 9x9 local maxima of the global softmax (the map's background level);
    the same local maxima the webapp already finds for its candidate list (tip_webapp/app/inference.py).

Run from research/confidence/learned:  python lean.py   (checks it against features.py on 2000 val maps, times it)
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent


def lr3_features(logits):
    """logits (128, 128) float32 -> (3,) float64 [logit(9x9 mass), entropy (nats), log(floor / peak)]."""
    L = logits.astype(np.float64)
    mx = L.max()
    lse = mx + np.log(np.exp(L - mx).sum())
    logp = L - lse                                                  # (128, 128) float64 log-probs
    p = np.exp(logp)                                                # (128, 128) float64 probs, sum 1
    iy, ix = divmod(int(np.argmax(L)), L.shape[1])                  # argmax cell (first in row-major order)
    # 1 - m9 is ~1e-5: take logit(m9) as log(inside / outside), never as 1 - m9 (precision; see features.py)
    inside = p[max(0, iy - 4):iy + 5, max(0, ix - 4):ix + 5].sum()  # float 9x9 mass around the argmax
    outside = p.sum() - inside                                      # float mass outside the 9x9 window
    ent = float(-(p * logp).sum())
    pf = p.astype(np.float32)                                       # float32, as features.py compares
    is_max = pf >= maximum_filter(pf, size=9, mode="constant", cval=-np.inf)   # (128, 128) bool local maxima
    h = np.sort(p[is_max])                                          # (n_max,) float64 local-maximum heights
    floor = float(h[(len(h) - 1) // 2])                             # lower median, as torch.nanmedian
    return np.array([np.log(max(inside, 1e-12)) - np.log(max(outside, 1e-12)), ent,
                     np.log(floor + 1e-12) - np.log(p[iy, ix] + 1e-12)])


class LR3:
    """LR3 scorer + isotonic calibration loaded from model_lr3.json / isotonic_LR3.json."""

    def __init__(self, model_json=HERE / "model_lr3.json", iso_json=HERE / "isotonic_LR3.json"):
        m = json.loads(Path(model_json).read_text())
        self.mu, self.sd = np.array(m["mu"]), np.array(m["sd"])     # (3,) float64 standardisation
        self.w, self.b = np.array(m["w"]), float(m["b"])            # (3,) float64 weights, bias of P(err > 3)
        k = json.loads(Path(iso_json).read_text()) if Path(iso_json).exists() else None
        self.kx = None if k is None else np.array(k["x"])           # (n_knots,) float64 score knots
        self.ky = None if k is None else np.array(k["y"])           # (n_knots,) float64 P(err <= 3 px)

    def score(self, logits):
        z = np.clip((lr3_features(logits) - self.mu) / self.sd, -8, 8)
        return float(-(z @ self.w + self.b))

    def prob(self, logits):
        return float(np.interp(self.score(logits), self.kx, self.ky))


if __name__ == "__main__":
    import sys
    import common as C
    import features as ft
    import torch
    lp = C.load_set("val", ("logp", "feats"))
    cols = [ft.FEATURE_NAMES.index(n) for n in ("lgt_m9", "entropy", "log_floor_over_peak")]
    n = 2000
    mine = np.stack([lr3_features(np.asarray(lp["logp"][i], np.float32)) for i in range(n)])
    # cached logp is float16: recompute the torch features from the same float16 maps for a like-for-like check
    ref = ft.map_features(torch.from_numpy(np.asarray(lp["logp"][:n], np.float32)))[0].numpy()[:, cols]
    print("max |numpy - torch| per feature:", np.abs(mine - ref).max(0))
    model = LR3()
    t0 = time.perf_counter()
    for i in range(500):
        model.score(np.asarray(lp["logp"][i], np.float32))
    print(f"LR3 numpy: {1000 * (time.perf_counter() - t0) / 500:.2f} ms / image (features + LR, no iso)")
    # end to end on real model logits: numpy LR3 vs the committed scores_LR3.npz, and the calibrated output
    import confbundle as cb
    lg = cb.logits(cb.load_model(), cb.load_split("val")["images"][:256])      # (256, 128, 128) float32
    s_np = np.array([model.score(lg[i]) for i in range(256)])
    s_ref = np.load(HERE / "scores_LR3.npz")["val"][:256]
    q = np.array([model.prob(lg[i]) for i in range(256)])
    print(f"end-to-end: max |numpy score - scores_LR3.npz| = {np.abs(s_np - s_ref).max():.2e};  "
          f"P(err<=3) of 256 val images: min {q.min():.3f} median {np.median(q):.4f} max {q.max():.5f}")
