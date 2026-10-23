"""Look-match statistics: new native-gray v7 (default config) vs old luma-port v7.

The old generator is extracted from commit 69b4769 into <workdir>/old_v7.py and
run as a separate script; the new one runs via its CLI.  Both use the same
base seed.  Usage: python tools/compare_v7_look.py [n] [seed]
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (pathlib.Path)
OLD_COMMIT = "69b4769"  # commit holding the luma-port v7 (str)
IMG = 256  # image side in px (int)
R = IMG // 3  # nominal dial radius in px (int)
C = IMG // 2  # canvas centre coordinate, px (int)
TIMEOUT = 900  # per-generator subprocess timeout, seconds (int)

# YY, XX: per-pixel row / column coordinates, each (256, 256) float64
YY, XX = np.mgrid[0:IMG, 0:IMG].astype(np.float64)
DIST = np.hypot(XX - C, YY - C)  # distance of each pixel from centre, (256, 256) float64
ANG = np.arctan2(YY - C, XX - C)  # polar angle of each pixel, (256, 256) float64 radians


def _run(cmd, label):
    """Run a generator subprocess with a timeout; raise with stderr on failure."""
    r = subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{r.stderr[-2000:]}")


def _image_stats(folder):
    """Per-image metric rows for one generated folder.

    Returns dict[str, np.ndarray], each (n,) float64:
      mean, std, dial (annulus .55-.9R minus pointer wedge +-15deg),
      bezel (1.05-1.2R), pointer (mean along centre->tip at 0.3-0.7 length),
      contrast (|pointer - dial|).
    """
    folder = Path(folder)
    with open(folder / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {k: [] for k in ("mean", "std", "dial", "bezel", "pointer", "contrast")}
    annulus = (DIST >= 0.55 * R) & (DIST <= 0.9 * R)  # (256, 256) bool
    bezel = (DIST >= 1.05 * R) & (DIST <= 1.2 * R)  # (256, 256) bool
    for row in rows:
        im = np.asarray(Image.open(folder / row["filename"]), dtype=np.float64)  # (256,256)
        tx, ty = float(row["tip_x"]), float(row["tip_y"])
        theta = np.arctan2(ty - C, tx - C)  # pointer direction, rad
        dtheta = np.abs((ANG - theta + np.pi) % (2 * np.pi) - np.pi)  # (256,256) rad
        dial = im[annulus & (dtheta > np.deg2rad(15))].mean()
        # pointer interior: 9 samples at 0.3..0.7 of centre->tip, nearest pixel
        ts = np.linspace(0.3, 0.7, 9)  # (9,) float64
        px = np.clip(np.rint(C + ts * (tx - C)).astype(int), 0, IMG - 1)  # (9,) int
        py = np.clip(np.rint(C + ts * (ty - C)).astype(int), 0, IMG - 1)  # (9,) int
        ptr = im[py, px].mean()
        out["mean"].append(im.mean())
        out["std"].append(im.std())
        out["dial"].append(dial)
        out["bezel"].append(im[bezel].mean())
        out["pointer"].append(ptr)
        out["contrast"].append(abs(ptr - dial))
    return {k: np.asarray(v) for k, v in out.items()}


def _summary(s):
    """Aggregate per-image arrays -> dict[str, float] of the reported metrics."""
    return {
        "image_mean": float(s["mean"].mean()),
        "image_std": float(s["std"].mean()),
        "dial_face_mean": float(s["dial"].mean()),
        "bezel_mean": float(s["bezel"].mean()),
        "pointer_mean": float(s["pointer"].mean()),
        "contrast_median": float(np.median(s["contrast"])),
        "low_contrast_frac": float((s["contrast"] < 15).mean()),
    }


def compare(n=500, seed=0, workdir="tmp_smoke/look"):
    """Generate n images with old and new v7 and compare look statistics.

    Returns dict[metric, {"old", "new", "diff"}] (floats, diff = new - old) and
    prints a table.  workdir is resolved against the repo root if relative.
    """
    wd = Path(workdir)
    if not wd.is_absolute():
        wd = REPO_ROOT / wd
    wd.mkdir(parents=True, exist_ok=True)
    old_py = wd / "old_v7.py"  # extracted old generator script
    src = subprocess.run(["git", "show", f"{OLD_COMMIT}:generate_synthetic_v7_mp.py"],
                         cwd=REPO_ROOT, timeout=60, capture_output=True, check=True).stdout
    old_py.write_bytes(src)
    workers = "4"
    _run([sys.executable, str(old_py), "--output-dir", str(wd / "old"),
          "--num-images", str(n), "--seed", str(seed), "--workers", workers], "old v7")
    _run([sys.executable, str(REPO_ROOT / "generate_synthetic_v7_mp.py"),
          "--output-dir", str(wd / "new"), "--num-images", str(n),
          "--seed", str(seed), "--workers", workers, "--overwrite"], "new v7")
    old, new = _summary(_image_stats(wd / "old")), _summary(_image_stats(wd / "new"))
    res = {k: {"old": old[k], "new": new[k], "diff": new[k] - old[k]} for k in old}
    print(f"{'metric':<20}{'old':>10}{'new':>10}{'diff':>10}")
    for k, v in res.items():
        print(f"{k:<20}{v['old']:>10.3f}{v['new']:>10.3f}{v['diff']:>+10.3f}")
    return res


if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 500,
            int(sys.argv[2]) if len(sys.argv) > 2 else 0)
