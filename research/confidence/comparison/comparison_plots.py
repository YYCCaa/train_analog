"""Stage 2 of 2: draw the method comparison from comparison_stats.npz (written by comparison_compute.py).

    figC1_methods.png          every method vs the shipped conf on val: ΔAUROC>3 and ΔAP>3 with paired-bootstrap
                               95 % CIs, and % of images flagged to catch all 13 misses
    figC2_cost_calibration.png AUROC>3 vs forward passes (dihedral K); reliability of P(err > 3 px) (raw vs isotonic)

Run from the repo root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# reference palette (dataviz skill, light mode); categorical slots 1-4 in fixed order, validated together
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "axes.axisbelow": True,
})


def save(fig, name):
    """Write fig to HERE/name at 150 dpi and close it."""
    fig.savefig(HERE / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {HERE / name}")
# FAMILY: dict family -> (colour, legend text), in the fixed categorical order
FAMILY = {"baseline": (MUTED, "baseline (shipped today)"), "single": (BLUE, "one pass: map statistics"),
          "learned": (AQUA, "one pass: learned"), "multi": (ORANGE, "K passes: dihedral mean map"),
          "combo": (YELLOW, "K passes: bc on the mean map (new)")}


def ci(a):
    """(N_BOOT,) bootstrap deltas -> (median, lo, hi) floats of the 95 % interval."""
    return tuple(np.percentile(a, [50, 2.5, 97.5]))


def fig_methods(s):
    n = len(s["keys"])
    y = np.arange(n)[::-1].astype(float)                      # (n,) row positions, first method on top
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    a, b, c = axes
    for i in range(n):
        col = FAMILY[str(s["families"][i])][0]
        for ax, arr in ((a, s["d_auc"][i]), (b, s["d_ap"][i])):
            mid, lo, hi = ci(arr)
            if s["keys"][i] == "conf":
                ax.plot([0], [y[i]], "D", ms=8, color=col, zorder=4)
                continue
            ax.plot([lo, hi], [y[i], y[i]], color=col, lw=2, solid_capstyle="round", zorder=3)
            ax.plot([mid], [y[i]], "o", ms=8, color=col, mec="white", mew=1.5, zorder=4)
        c.plot([s["flag_all"][i]], [y[i]], "o", ms=8, color=col, mec="white", mew=1.5, zorder=4)
        c.text(s["flag_all"][i] * 1.12, y[i], f"{s['flag_all'][i]:.2f} %", va="center", fontsize=8.5, color=INK2)
        a.text(0.0435, y[i], f"{s['auroc'][i]:.4f}", va="center", ha="right", fontsize=8.5, color=INK2)
        b.text(0.122, y[i], f"{s['ap'][i]:.3f}", va="center", ha="right", fontsize=8.5, color=INK2)
    labels = [f"{s['labels'][i]}   ·  {s['passes'][i]} pass{'es' if s['passes'][i] > 1 else ''}" for i in range(n)]
    a.set_yticks(y, labels)
    a.tick_params(axis="y", length=0)
    for ax in (a, b):
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    a.set_xlim(-0.012, 0.045)
    b.set_xlim(-0.075, 0.125)
    a.set_xlabel("Δ AUROC (err > 3 px) vs shipped conf   (right: absolute AUROC)")
    b.set_xlabel("Δ average precision (err > 3 px) vs shipped conf   (right: absolute AP)")
    a.set_title("Ranking 3 px errors below good images")
    b.set_title("Precision when flagging the least trusted first")
    c.set_xscale("log")
    c.set_xlim(0.8, 80)
    c.set_xlabel("% flagged to catch all 13 misses (log; lower is better)")
    c.set_title("Catching the 13 misses (hangs on 1-2 images)")
    # family separators
    fam = [str(f) for f in s["families"]]
    for i in range(1, n):
        if fam[i] != fam[i - 1]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color=AXIS, lw=0.8, ls=":", zorder=1)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=col, label=txt) for col, txt in FAMILY.values()]
    fig.legend(handles=handles, loc="upper left", ncol=5, bbox_to_anchor=(0.01, 0.955), fontsize=9.5,
               labelcolor=INK, handletextpad=0.3, columnspacing=1.6)
    fig.suptitle("Every confidence method on val (22,500 images): bars are paired-bootstrap 95 % intervals",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    save(fig, "figC1_methods.png")


def fig_cost_calibration(s):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8))
    K = s["k_curve_K"]
    for row, lab, col, ls in ((2, "bc T1.5 on the mean map", YELLOW, "-"), (0, "mean-map 5x5 mass (conf at K=1)", ORANGE, "-"),
                              (1, "mean-map 9x9 mass (m9 at K=1)", ORANGE, "--")):
        a.plot(K, s["k_curve"][row], ls, color=col, marker="o", ms=8, mec="white", mew=1.5, label=lab)
        a.text(K[-1] * 1.08, s["k_curve"][row][-1], lab, va="center", fontsize=9, color=INK2)
    i = list(s["keys"]).index("lr3")
    a.plot([1], [s["auroc"][i]], "s", ms=9, color=AQUA, mec="white", mew=1.5, label="LR3 (one pass, learned)")
    a.text(1, s["auroc"][i] + 0.0007, "LR3 (learned)", ha="center", va="bottom", fontsize=9, color=INK2)
    a.set_xscale("log", base=2)
    a.set_xticks(K, [f"{k}" for k in K])
    a.set_xlim(0.8, 22)
    a.set_xlabel("forward passes per image (dihedral views averaged)")
    a.set_ylabel("val AUROC (err > 3 px)")
    a.set_title("What each extra forward pass buys")
    a.legend(loc="lower right", fontsize=9, labelcolor=INK, handlelength=3.2)

    for key, lab, col, mk in (("raw_conf", "raw conf read as a probability", MUTED, "D"),
                              ("iso_conf", "isotonic(conf), out-of-fold", BLUE, "o"),
                              ("iso_lr3", "isotonic(LR3), out-of-fold", AQUA, "s")):
        p, o, n = s[f"rel_{key}_pred"], s[f"rel_{key}_obs"], s[f"rel_{key}_n"]
        b.plot(p, np.maximum(o, 1.5e-4), "-", color=col, marker=mk, ms=8, mec="white", mew=1.5, label=lab)
    lim = (1e-4, 1.0)
    b.plot(lim, lim, color=AXIS, lw=1.2, ls="--", zorder=1, label="perfect calibration")
    b.set_xscale("log"), b.set_yscale("log")
    b.set_xlim(*lim), b.set_ylim(*lim)
    b.set_xlabel("predicted P(err > 3 px)   (quantile bins of images, riskiest right)")
    b.set_ylabel("observed share with err > 3 px   (bottom row = none observed)")
    b.set_title("Is the number a real probability?")
    b.legend(loc="upper left", fontsize=9, labelcolor=INK)
    fig.suptitle("Cost vs gain, and calibration (val)", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "figC2_cost_calibration.png")


def main():
    s = dict(np.load(HERE / "comparison_stats.npz"))
    fig_methods(s)
    fig_cost_calibration(s)


if __name__ == "__main__":
    main()
