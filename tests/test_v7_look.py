"""Look-match check: new native-gray v7 vs old luma-port v7 (commit 69b4769).

Marked `slow` (generates 2 x 300 images).  It is NOT deselected by default, so a
plain `pytest` runs it; use `-m "not slow"` to skip or `-m slow` to run only it.

Pointer interior mean and pointer-vs-dial contrast are reported but exempt from
asserts: pointer gray is uniform 0-255 by user decision, which differs from the
old HSV->luma pointer by design.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_v7_look import compare  # noqa: E402


@pytest.mark.slow
def test_look_matches_old_v7(tmp_path):
    res = compare(n=300, workdir=str(tmp_path))
    assert abs(res["image_mean"]["diff"]) <= 8
    assert abs(res["image_std"]["diff"]) <= 6
    assert abs(res["dial_face_mean"]["diff"]) <= 8
    assert abs(res["bezel_mean"]["diff"]) <= 8
    assert abs(res["low_contrast_frac"]["diff"]) <= 0.05
