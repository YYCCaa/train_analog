"""Probe: the float32 scores_<method>.npz reproduce the float64 harness rows (AUROC>3px / AP>3px). Run from repo root."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mp_scores as ms  # noqa: E402

rows = [json.loads(line) for line in open(ms.RESULTS, encoding="utf-8")]   # list[dict] harness rows
for path in sorted(glob.glob(str(ms.HERE / "scores_*.npz"))):
    name = next(k for k, v in ms.SAVE.items() if path.endswith(f"scores_{v}.npz"))
    for split in ("train", "val"):
        s = np.load(path)[split].astype(np.float64)
        bad = ms.load_meta(split)["err"] > 3.0
        ref = next(r for r in rows if r["name"] == name and r["split"] == split)
        print(f"{name:<22} {split:<5} AUROC3 npz {ms.ev.auroc(s, bad):.5f} vs row {ref['auroc_err3']:.5f}   "
              f"AP3 npz {ms.ev.average_precision(s, bad):.4f} vs row {ref['ap_err3']:.4f}")
