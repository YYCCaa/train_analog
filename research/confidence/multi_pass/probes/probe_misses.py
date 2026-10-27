"""Probe: the 13 val misses (> 7 px) under multi-pass consistency. Diagnostic only (no tuning). Run from repo root.

Per miss: clean err, conf, m9, dihedral K=8 max tip spread, number of the 7 dihedral extras within 3 px of the
clean tip, mean-map m5 / m9 at the clean argmax, and the percentile rank (0 = flagged first) of conf, m9 and
the mean-map scores among all 22500 val images; also whether the K=8 mean-map decode fixes the miss.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mp_scores as ms  # noqa: E402

f = ms.load_feat(ms.CACHE / "feat_val.npz")
meta = ms.load_meta("val")
err, lab = meta["err"], meta["label"].astype(np.float64)
miss = np.flatnonzero(err > 7.0)                                       # (13,) int64 val positions


def pct(s):
    """(n,) score -> (n,) float percentile rank, 0 = lowest score (flagged first)."""
    return 100.0 * (rankdata(s) - 1) / (len(s) - 1)


m5_8, m9_8 = ms.score(f, "dih", 8, "avgm5c"), ms.score(f, "dih", 8, "avgm9c")
spread = -ms.score(f, "dih", 8, "spread_max")
agree = (ms.dists(f, "dih", 8) <= 3.0).sum(1)
tta_err = np.linalg.norm(f["dih_avgtip"][:, 2] - lab, axis=1)
p_conf, p_m9, p_m5_8, p_m9_8 = pct(meta["conf"]), pct(meta["m9"]), pct(m5_8), pct(m9_8)
print(f"{'pos':>6} {'err':>6} {'conf':>5} {'m9':>5} {'spread':>6} {'agr3':>4} {'mm5':>5} {'mm9':>5} "
      f"{'%conf':>6} {'%m9':>6} {'%mm5':>6} {'%mm9':>6} {'TTAerr':>6}")
for i in miss[np.argsort(-err[miss])]:
    print(f"{i:6d} {err[i]:6.1f} {meta['conf'][i]:5.2f} {meta['m9'][i]:5.2f} {spread[i]:6.1f} {agree[i]:4d} "
          f"{m5_8[i]:5.2f} {m9_8[i]:5.2f} {p_conf[i]:6.2f} {p_m9[i]:6.2f} {p_m5_8[i]:6.2f} {p_m9_8[i]:6.2f} "
          f"{tta_err[i]:6.1f}")
