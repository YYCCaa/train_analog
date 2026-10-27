"""Does "bc T1.5 on the dihedral mean map" (chosen after seeing val) also lead on the TRAIN sample?

    train sample (20000 images) ──8 dihedral passes──▶ dm5_K / dm9_K / dbc_K (comparison_compute.dihedral_scores)
    ──▶ frozen harness rows (train) next to conf, m9 and single-pass bc
Run from the repo root; prints only.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison_compute import cb, dihedral_scores, ev  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    tr = cb.load_split("train")
    d = dihedral_scores(cb.load_model(), tr["images"])
    ev.print_header()
    ev.evaluate("train", tr["conf"], "conf, shipped 5x5 mass")
    ev.evaluate("train", tr["m9"], "m9, 9x9 mass")
    for K in (1, 4, 8):
        for f, lab in (("dm5", "dihedral mean-map m5"), ("dbc", "bc T1.5 on dihedral mean map")):
            if not (K == 1 and f == "dm5"):
                ev.evaluate("train", d[f"{f}_{K}"], f"{lab}, K={K}", passes=K)


if __name__ == "__main__":
    main()
