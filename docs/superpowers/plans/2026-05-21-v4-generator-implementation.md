# v4 Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `generate_synthetic_v4.py` — a tick-and-digit-only synthetic generator on Places365 backgrounds — so we can run the bezel-cue ablation experiment described in `docs/superpowers/plans/2026-05-21-v4-generator-ablation.md`.

**Architecture:** Fork `generate_synthetic_v3.py`, strip every render step except scale-ticks and digits, force photo backgrounds, add one new helper `pick_ink_color` that picks black-or-white per image from the local photo luminance, and wire that color into `draw_scale_ticks` + `draw_scale_digits`. Everything else (geometry sampling, CSV schema, photo loader, post-processing) stays byte-for-byte identical so v4 vs. v3 is a single-variable comparison.

**Tech Stack:** Python 3, PIL/Pillow, NumPy, pytest. Shell is PowerShell on Windows — use PowerShell syntax for any commands.

---

## File Structure

**Create:**
- `generate_synthetic_v4.py` — the new generator (fork of v3)
- `tests/test_v4_generator.py` — unit tests for `pick_ink_color` and `sample_params` v3↔v4 parity

**Modify:** none. No training scripts, no `autocal_utils.py`, no existing test changes.

**Reference (read-only):**
- `generate_synthetic_v3.py` — source of the fork
- `docs/superpowers/plans/2026-05-21-v4-generator-ablation.md` — design spec
- `tests/test_decoupled.py` — example of the project's pytest style

---

### Task 1: Scaffold v4 by forking v3

**Files:**
- Create: `generate_synthetic_v4.py`

- [ ] **Step 1: Copy v3 to v4**

PowerShell:
```powershell
Copy-Item generate_synthetic_v3.py generate_synthetic_v4.py
```

- [ ] **Step 2: Replace the module docstring at the top of generate_synthetic_v4.py**

Open `generate_synthetic_v4.py`. The current top-of-file docstring (lines 1-12) starts with `"""Generate synthetic training images for autocalibration.` Replace that whole docstring with:

```python
"""Generate synthetic training images for the tick-only ablation (v4).

Forked from generate_synthetic_v3.py with every render step except scale
ticks and digits removed, on top of Places365 photo backgrounds.  Ink
color (tick + digit) is picked once per image from the mean luminance of
the photo patch under the implicit dial ellipse.

This is an ablation dataset for measuring how much of v3's center
accuracy came from bezel cues vs. tick-structure learning.  Not a
deployment-target distribution.

See docs/superpowers/plans/2026-05-21-v4-generator-ablation.md for
the design rationale.

Labels per image (identical to v3):
  filename, cx, cy, a, b, theta, phi_0, mark_0_x, mark_0_y, ..., mark_9_x, mark_9_y
"""
```

- [ ] **Step 3: Change OUTPUT_DIR default**

In `generate_synthetic_v4.py`, find:
```python
OUTPUT_DIR = "sv3"
```
Replace with:
```python
OUTPUT_DIR = "sv4"
```

- [ ] **Step 4: Change BG_SOURCE default**

In `generate_synthetic_v4.py`, find:
```python
BG_SOURCE = "procedural"             # "procedural" | "photo"
```
Replace with:
```python
BG_SOURCE = "photo"                  # v4 requires photo; procedural removed
```

- [ ] **Step 5: Verify import works and constants are right**

Run:
```powershell
python -c "import generate_synthetic_v4 as g; print(g.OUTPUT_DIR, g.BG_SOURCE)"
```
Expected output:
```
sv4 photo
```

- [ ] **Step 6: Commit**

```powershell
git add generate_synthetic_v4.py
git commit -m "v4: scaffold ablation generator (fork from v3)"
```

---

### Task 2: Add `pick_ink_color` helper (TDD)

**Files:**
- Create: `tests/test_v4_generator.py`
- Modify: `generate_synthetic_v4.py` (add one new function)

- [ ] **Step 1: Write the failing test**

Create `tests/test_v4_generator.py` with this content:

```python
"""Tests for generate_synthetic_v4.

Covers:
  - pick_ink_color picks black on bright BG, white on dark BG
  - pick_ink_color falls back to a default for an empty patch
  - sample_params matches v3 byte-for-byte under the same seed
"""

import random

import numpy as np
import pytest
from PIL import Image

import generate_synthetic_v4 as v4


def _solid_bg(rgb, size=256):
    return Image.new("RGB", (size, size), rgb)


def test_pick_ink_color_picks_black_on_bright_bg():
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (0, 0, 0)


def test_pick_ink_color_picks_white_on_dark_bg():
    bg = _solid_bg((20, 20, 20))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (255, 255, 255)


def test_pick_ink_color_picks_black_on_threshold():
    # Mean luminance exactly 128 → black (>= threshold).
    bg = _solid_bg((128, 128, 128))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (0, 0, 0)


def test_pick_ink_color_clamps_bbox_to_image():
    # Centre right at the corner so half the bbox is outside the image.
    # Should still return a valid colour (uses whatever pixels are in
    # the clamped bbox).
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=0.0, cy=0.0, a=80.0)
    assert color in [(0, 0, 0), (255, 255, 255)]


def test_pick_ink_color_handles_degenerate_bbox():
    # Centre fully outside the image → empty patch → fallback to black
    # (mean_lum default 128.0 → >= 128 → black).
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=-1000.0, cy=-1000.0, a=80.0)
    assert color == (0, 0, 0)
```

- [ ] **Step 2: Run test to verify it fails (function not defined)**

Run:
```powershell
pytest tests/test_v4_generator.py -v
```
Expected: All four `test_pick_ink_color_*` tests FAIL with `AttributeError: module 'generate_synthetic_v4' has no attribute 'pick_ink_color'`.

- [ ] **Step 3: Add `pick_ink_color` to `generate_synthetic_v4.py`**

Open `generate_synthetic_v4.py`. Find the existing `def _hsv_rgb(h, s, v):` (around line 244). Insert this new function immediately ABOVE `_hsv_rgb`:

```python
def pick_ink_color(bg_img, cx, cy, a, threshold=128.0):
    """Pick a per-image tick/digit ink color based on the photo BG patch.

    Computes the mean luminance of the photo region inside the
    bounding box `[cx - a, cy - a, cx + a, cy + a]` (clamped to image
    bounds) using Rec. 601 weights (0.299/0.587/0.114).  If the patch
    is empty (centre fully outside the image), falls back to a 128.0
    mean luminance.  Returns pure black for mean >= threshold,
    pure white otherwise.

    Called once per image after photo augmentation; the returned
    triple is then used for every tick line and every digit stroke
    drawn on this image (except for the rare TICK_COLORED_PROB
    branch, which keeps its own saturated-hue logic).
    """
    w, h = bg_img.size
    xs = max(0, int(cx - a))
    ys = max(0, int(cy - a))
    xe = min(w, int(cx + a))
    ye = min(h, int(cy + a))
    if xe <= xs or ye <= ys:
        mean_lum = 128.0
    else:
        patch = np.asarray(bg_img.crop((xs, ys, xe, ye)))
        if patch.size == 0:
            mean_lum = 128.0
        else:
            mean_lum = float(
                (0.299 * patch[..., 0]
                 + 0.587 * patch[..., 1]
                 + 0.114 * patch[..., 2]).mean())
    return (0, 0, 0) if mean_lum >= threshold else (255, 255, 255)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
pytest tests/test_v4_generator.py -v
```
Expected: All five tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add generate_synthetic_v4.py tests/test_v4_generator.py
git commit -m "v4: add pick_ink_color helper + unit tests"
```

---

### Task 3: Strip non-tick/digit render steps from `generate_image`

**Files:**
- Modify: `generate_synthetic_v4.py` (rewrite the `generate_image` function body)

This is the largest mechanical change. We're rewriting `generate_image` from 14 steps down to ~5: load photo → sample params → pick ink color → draw ticks + digits → post-process → save.

- [ ] **Step 1: Replace the entire `generate_image` function in `generate_synthetic_v4.py`**

In `generate_synthetic_v4.py`, find the function definition `def generate_image(path, img_size=IMG_SIZE, photo_paths=None,` and the closing `return label` at the end of its body (in v3 these are around line 1638 through line 1793 — they may differ slightly in v4 if you've moved anything, but the function boundaries are the same).

Replace the entire function (from the `def` line through the final `return label`, inclusive) with:

```python
def generate_image(path, img_size=IMG_SIZE, photo_paths=None,
                   bg_prefetcher=None):
    """Generate one v4 synthetic dial image.  Returns dict of label fields.

    Renders ONLY scale ticks and digits on top of a Places365 photo
    background.  No bezel, no dial face, no pointer, no distractor
    text, no intruder dial, no rim shadows.  Per-image tick/digit ink
    color is picked from the mean luminance of the photo patch under
    the implicit dial ellipse (see pick_ink_color).

    `bg_prefetcher` is the same `_BGPrefetcher` interface as v3 — when
    provided, photo decoding has already happened on a worker thread.
    """
    if BG_SOURCE != "photo":
        raise ValueError(
            f"v4 requires BG_SOURCE='photo'; got {BG_SOURCE!r}.  v4 has "
            f"no procedural-background code path.")

    params = sample_params(img_size=img_size)

    # 1. Photo background.
    if bg_prefetcher is not None:
        img = _apply_photo_augmentation(bg_prefetcher.get())
    else:
        if photo_paths is None:
            photo_paths = _load_bg_photo_paths(BG_PHOTO_DIR)
        img = make_photo_background(img_size, photo_paths)

    # 2. Per-image ink color from the photo patch under the dial area.
    #    Computed AFTER photo augmentation so the colour matches what the
    #    model actually sees.
    ink_color = pick_ink_color(img, params["cx"], params["cy"], params["a"])

    # 3. Scale ticks + digits.  Each tick line and glyph is mirrored
    #    into `protect_mask` so the triangle-artifact pass can avoid them.
    protect_mask = Image.new("L", (img_size, img_size), 0)
    protect_draw = ImageDraw.Draw(protect_mask)
    img, tick_inner_rc = draw_scale_ticks(img, params, ink_color,
                                          protect_draw=protect_draw)
    img = draw_scale_digits(img, params, tick_inner_rc, ink_color,
                            protect_draw=protect_draw)

    # 4. Whole-image Gaussian blur (camera-defocus simulation).
    img = apply_global_blur(img)

    # 5. Adversarial triangle artifacts — skip pixels in protect_mask.
    img = add_triangle_artifacts(img, protect_mask=protect_mask)

    # 6. Colour / exposure jitter.
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
    img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # 7. Save.
    img.save(path)

    # Derive marks from the same params we rendered.
    marks = derive_marks_np(params["cx"], params["cy"], params["a"], params["b"],
                            params["theta"], params["phi_0"])

    label = {
        "filename": os.path.basename(path),
        "cx": round(params["cx"], 3),
        "cy": round(params["cy"], 3),
        "a": round(params["a"], 3),
        "b": round(params["b"], 3),
        "theta": round(params["theta"], 6),
        "phi_0": round(params["phi_0"], 6),
    }
    for i in range(N_MARKS):
        label[f"mark_{i}_x"] = round(float(marks[i, 0]), 3)
        label[f"mark_{i}_y"] = round(float(marks[i, 1]), 3)
    return label
```

NOTE: This calls `draw_scale_ticks(img, params, ink_color, protect_draw=...)` and `draw_scale_digits(img, params, tick_inner_rc, ink_color, protect_draw=...)` — both function signatures grow a new `ink_color` positional argument. We add it to the two functions in Tasks 4 and 5. The script won't fully run end-to-end until those tasks land; we commit this step's rewrite first as the structural change.

- [ ] **Step 2: Validate the rewrite doesn't break import**

Run:
```powershell
python -c "import generate_synthetic_v4"
```
Expected: no exception (import succeeds; `generate_image` itself isn't called by import). If you see a `NameError` for one of the dropped functions (e.g. `draw_outer_canvas`), find and delete its call site — every dropped step should already be gone from the new `generate_image`.

- [ ] **Step 3: Commit**

```powershell
git add generate_synthetic_v4.py
git commit -m "v4: strip non-tick/digit render steps from generate_image"
```

---

### Task 4: Wire `ink_color` into `draw_scale_ticks`

**Files:**
- Modify: `generate_synthetic_v4.py` (one function)

- [ ] **Step 1: Modify `draw_scale_ticks` signature**

In `generate_synthetic_v4.py`, find:
```python
def draw_scale_ticks(img, params, protect_draw=None):
```
Replace with:
```python
def draw_scale_ticks(img, params, ink_color, protect_draw=None):
```

- [ ] **Step 2: Replace the per-tick gray path with `ink_color`**

In `generate_synthetic_v4.py`, inside `draw_scale_ticks`, find the per-tick color-decision block (in v3 these were lines 905-913):

```python
            # Per-tick gray offset (luminance jitter only — geometry stays rigid).
            if colored_mode:
                offset = random.randint(*TICK_GREY_PER_TICK_JITTER)
                color_rgb = tuple(
                    max(0, min(255, int(c) + offset)) for c in tick_base_rgb)
            else:
                g = max(0, min(255,
                               base_grey + random.randint(*TICK_GREY_PER_TICK_JITTER)))
                color_rgb = (g, g, g)
            color_rgba = color_rgb + (255,)
```

Replace it with:

```python
            # Per-tick color.  In colored-mode (TICK_COLORED_PROB) keep
            # the v3 per-tick gray offset around the saturated hue;
            # otherwise use the per-image ink_color verbatim (no jitter
            # — auto-color is already a single deterministic choice).
            if colored_mode:
                offset = random.randint(*TICK_GREY_PER_TICK_JITTER)
                color_rgb = tuple(
                    max(0, min(255, int(c) + offset)) for c in tick_base_rgb)
            else:
                color_rgb = ink_color
            color_rgba = color_rgb + (255,)
```

The lines computing `base_grey` (one-per-image, line ~841) can be left in place; they're now unreferenced inside the function but harmless. (They'll be cleaned up if/when Task 8 dead-code pass runs.)

- [ ] **Step 3: Commit**

```powershell
git add generate_synthetic_v4.py
git commit -m "v4: wire ink_color into draw_scale_ticks"
```

---

### Task 5: Wire `ink_color` into `draw_scale_digits`

**Files:**
- Modify: `generate_synthetic_v4.py` (one function)

- [ ] **Step 1: Modify `draw_scale_digits` signature**

In `generate_synthetic_v4.py`, find:
```python
def draw_scale_digits(img, params, tick_inner_rc, protect_draw=None):
```
Replace with:
```python
def draw_scale_digits(img, params, tick_inner_rc, ink_color, protect_draw=None):
```

- [ ] **Step 2: Replace the per-image digit gray pick with `ink_color`**

In `generate_synthetic_v4.py`, inside `draw_scale_digits`, find (in v3 these were lines 992-993):

```python
    grey = random.randint(*DIGIT_GREY_RANGE)
    color = (grey, grey, grey)
```

Replace with:

```python
    color = ink_color
```

(Removing the `random.randint` call drops one RNG draw vs. v3.  That's fine — geometric labels are sampled in `sample_params` before any of this, so they remain identical.)

- [ ] **Step 3: Smoke-run a single image to verify the pipeline works**

Confirm the end-to-end pipeline actually produces an image.  Run:

```powershell
python -c @'
import generate_synthetic_v4 as g
import os, random
random.seed(0)
os.makedirs("sv4_one_image", exist_ok=True)
label = g.generate_image("sv4_one_image/dial_00000.png")
print("OK:", label["filename"], "cx=", label["cx"], "ink-driven render done")
'@
```

Expected: prints `OK: dial_00000.png cx= <float> ink-driven render done` and an image file `sv4_one_image/dial_00000.png` is created. If you get `NameError`/`AttributeError`/`TypeError`, fix it before committing.

If you don't have any photos under `~/places365` yet, this will exit with the "No *.jpg / *.npy files" error from `_load_bg_photo_paths`. In that case, point `BG_PHOTO_DIR` at any directory of JPGs you have, or skip this smoke and rely on Task 7's smoke run after photos are in place.

- [ ] **Step 4: Commit**

```powershell
git add generate_synthetic_v4.py
git commit -m "v4: wire ink_color into draw_scale_digits"
```

---

### Task 6: Sample-params parity test with v3

**Files:**
- Modify: `tests/test_v4_generator.py` (add one test)

This guards against accidental drift in `sample_params` between v3 and v4 over time. With the seeded RNG controlled by the test, both modules' `sample_params` should produce byte-identical geometric labels.

- [ ] **Step 1: Append the parity test to `tests/test_v4_generator.py`**

Open `tests/test_v4_generator.py`. At the bottom of the file, add:

```python
def test_sample_params_geometry_matches_v3():
    """v4's sample_params is verbatim from v3 by design.  This regression
    test fails if anyone touches the geometric sampler in either copy."""
    import generate_synthetic_v3 as v3
    import generate_synthetic_v4 as v4_mod

    SEED = 42
    GEOMETRIC_KEYS = ["cx", "cy", "a", "b", "theta", "phi_0", "outer_scale"]

    random.seed(SEED)
    np.random.seed(SEED)
    v3_params = v3.sample_params()

    random.seed(SEED)
    np.random.seed(SEED)
    v4_params = v4_mod.sample_params()

    for key in GEOMETRIC_KEYS:
        assert v4_params[key] == pytest.approx(v3_params[key]), (
            f"v4 sample_params drifted from v3 on key {key!r}: "
            f"v3={v3_params[key]} v4={v4_params[key]}")
```

- [ ] **Step 2: Run the parity test**

Run:
```powershell
pytest tests/test_v4_generator.py::test_sample_params_geometry_matches_v3 -v
```
Expected: PASS.

If it fails on `outer_scale` (or any other key), `sample_params` has drifted between the two modules — that means a step somewhere unintentionally modified it.  Fix by re-copying that function from v3 verbatim.

- [ ] **Step 3: Run the whole v4 test file to make sure nothing else regressed**

Run:
```powershell
pytest tests/test_v4_generator.py -v
```
Expected: all six tests PASS.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_v4_generator.py
git commit -m "v4: sample_params parity test against v3"
```

---

### Task 7: Smoke-generate 50 images and manually review

**Files:**
- Create (transient): `sv4_smoke/` directory with 50 PNGs + `labels.csv`

This is a **manual checkpoint**, not a unit test.  No code changes here — just run the generator and look at the results.

- [ ] **Step 1: Confirm Places365 photos are available**

```powershell
python -c "import generate_synthetic_v4 as g; paths = g._load_bg_photo_paths(g.BG_PHOTO_DIR); print(f'{len(paths)} photos under {g.BG_PHOTO_DIR}')"
```
Expected: prints a non-zero count.  If it errors with "No *.jpg / *.npy files," populate `BG_PHOTO_DIR` before continuing (e.g. `python download_places365.py --root ~/places365`).

- [ ] **Step 2: Generate 50 smoke images**

```powershell
python generate_synthetic_v4.py --output-dir sv4_smoke --num-images 50
```
Expected: prints progress to ~50 images, no exceptions.

- [ ] **Step 3: Verify the CSV schema matches v3**

```powershell
$v3 = Get-Content sv3/labels.csv -TotalCount 1
$v4 = Get-Content sv4_smoke/labels.csv -TotalCount 1
if ($v3 -eq $v4) { "schema OK" } else { "schema MISMATCH"; "v3: $v3"; "v4: $v4" }
```
Expected: `schema OK`.  If `sv3/labels.csv` doesn't exist, run a 1-image v3 smoke first or compare the header manually against the spec.

- [ ] **Step 4: Manually inspect images**

Open `sv4_smoke/` in a file browser.  Look at 10-20 images and check:
- Are ticks visible against the photo background?
- Are digits readable enough that you (a human) can identify which one is "0"?
- Any leftover dial structure (stray bezel polyline, pointer, hub circle)?  There shouldn't be any.
- Auto-color: bright photos should have black ticks/digits; dark photos should have white.  Spot one bright-BG image and one dark-BG image and confirm.
- The ~10% colored-tick images (`TICK_COLORED_PROB = 0.10`): do any of them have nearly-invisible dark saturated ticks against a dark photo?  (Expected and accepted per the spec; flag for human review of v4 results if many.)

- [ ] **Step 5: Decide go/no-go**

If the smoke looks reasonable, proceed to Task 8.  If you see leftover dial structure or systematic illegibility, stop and report the failure mode — don't run the full 25k.

- [ ] **Step 6: Commit (smoke artifacts NOT in the commit)**

There's nothing to commit at this step — `sv4_smoke/` is local-only and should be in `.gitignore` or just left untracked.  Confirm:
```powershell
git status
```
Expected: no new tracked files from `sv4_smoke/`.

---

### Task 8: Full generation — 25k images

**Files:**
- Create: `sv4/` directory with 25,000 PNGs + `labels.csv`

- [ ] **Step 1: Generate the full dataset**

```powershell
python generate_synthetic_v4.py
```
(Defaults: `--output-dir sv4 --num-images 25000`.)

Expected runtime: similar order to v3 (BG prefetcher is the same; the per-image render is *faster* than v3 since most render steps are dropped).  Roughly 30-60 minutes depending on photo cache.

- [ ] **Step 2: Verify counts**

```powershell
(Get-ChildItem sv4 -Filter "dial_*.png").Count
(Get-Content sv4/labels.csv | Measure-Object -Line).Lines
```
Expected: 25000 PNG files; CSV line count is 25001 (header + 25k rows).

- [ ] **Step 3: Sanity-check the CSV**

```powershell
Get-Content sv4/labels.csv -TotalCount 3
```
Expected: header line followed by two well-formed data rows.  All numeric fields should look sane (no NaN, no obviously-wrong negative `a`).

- [ ] **Step 4: Report ready for training**

`sv4/` is now ready as a drop-in `--data-dir` for any existing trainer:
```
python train_ellipse_j.py --data-dir sv4 --save-onnx ellipse_model_j_v4.onnx
```
That kicks off the ablation experiment — out of scope for this plan.

No commit at this step (data isn't checked in).

---

## Out of Scope For This Plan

- Dead-code cleanup in `generate_synthetic_v4.py` (unused functions like `make_procedural_background`, `draw_outer_canvas`, `draw_distractor_text`, `_sample_intruder_params`, pointer/shadow helpers, and their constants). They're harmless leftovers; trimming improves readability but isn't load-bearing for the ablation. Can be a follow-up commit.
- A `--render-mode` flag on v3 / v3 refactor into composable renderers.
- Training experiments on v4 data (run separately after the dataset is generated).
- Halo or per-tick auto-color mechanisms (deferred per design spec).
- v5 design.
