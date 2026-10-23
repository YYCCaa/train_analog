# v7 Native-Gray Generator + YAML + Studio App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the luma-port `generate_synthetic_v7_mp.py` with a native-grayscale, YAML-configured, per-section-RNG generator, and build a local FastAPI + React studio with a control for every parameter, live in-RAM preview and on-disk dataset generation.

**Architecture:** `v7_config.py` loads, merges, validates and dumps a nested-dict config, using `configs/v7_default.yaml` (values) and `configs/v7_schema.yaml` (types, bounds, help). `generate_synthetic_v7_mp.py` exposes one pure renderer, `render_image(cfg, seed)`, plus a pool driver, `generate_dataset(...)`, which both the CLI and the studio backend call. `studio/backend` (FastAPI) wraps preview and generation jobs. `studio/frontend` (Vite + React + TS) builds its controls from the schema, and the backend serves the built bundle.

**Tech Stack:** Python 3.12, NumPy, Pillow, PyYAML, FastAPI 0.115, uvicorn, pytest, Playwright (python, 1.63), Node 22 / npm 10, Vite, React 18, TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-23-v7-yaml-generator-studio-design.md`

## Global Constraints

- **Native gray only.** No `colorsys`, no `"RGB"`/`"RGBA"` image modes, no HSV, and no `convert("L")` of a colour image anywhere in `generate_synthetic_v7_mp.py`. Every colour is an int gray level in 0–255.
- **Output.** Images are 256×256 uint8 single-channel PNGs named `dial_{idx:04d}.png`. `labels.csv` has the columns `filename, angle_deg, tip_x, tip_y`, with tips rounded to 2 decimals.
- **Per-image seed.** It is `_seed_for_index(seed_base, idx) = (seed_base + idx * 2654435761) & 0xFFFFFFFF` (unchanged). Output is bit-exact regardless of `--workers` and scheduling.
- **RNG streams.** Each RNG section gets its own stream, seeded from `(image_seed, section_name)`. Never use the global `random` / `np.random` state in rendering.
- **Spawn-safe config.** Windows uses `spawn`, so the config dict reaches workers through Pool initializer args, never through module globals mutated in the parent.
- **No unused parameters.** Every key in `v7_default.yaml` has a schema entry and a UI control, and changing it changes the output. Dropped constants: `BLUR_RANGE`, `RIM_SHADOW_SECONDARY_OPACITY`, `VALUE_MIN`, `VALUE_MAX`, `CENTER`, `SATURATION_RANGE`, `DIAL_S_RANGE`, `SHADOW_S_RANGE`, hue draws. The `augment=False` code path is removed: the CLI never used it, and it would be a control that does nothing.
- **Image size.** `canvas.img_size` = 256 is read-only in the UI.
- **Comment rule (user).** At every declaration, comment what it is plus its shape and dtype.
- **Test rule (user).** Generator tests drive the CLI through `subprocess`. Studio tests drive the real page in a browser (Playwright). Test/debug scratch output goes to `tmp_smoke/` (overwrite, never `tmp_smoke_*`), and the folder is deleted before committing.
- **Localhost only.** The studio binds to `127.0.0.1` only. The output folder for app generation must be an absolute path.
- **Python for commands.** `$PY` = `C:/Users/Administrator/PycharmProjects/train_dial/.venv/Scripts/python.exe` (the worktree has no venv). Every pytest call uses a `timeout` (in subprocess calls) or `--timeout`-free short tests. Long runs go in the background with a timer.

## Review Focus

1. **Schema extremes.** With every slider at its schema min, or every slider at its schema max, `render_image` must not crash. For example, a max scale must not push the canvas past `img_size`. Pinned by `test_render_extremes_do_not_crash` (Task 2), and the schema bounds are chosen so the test passes.
2. **Degenerate ranges.** Equal range ends (`[5, 5]`) must render. A YAML with `min > max`, an unknown key (e.g. an old `saturation_range`) or a wrong type must return a clear error that names the key, not a 500 or a traceback in the UI. Pinned by `test_invalid_configs_report_key` (Task 1) and `test_preview_invalid_config_422` (Task 5).
3. **Fast slider dragging.** Preview responses can arrive out of order. The grid must show the result for the *latest* config only. Pinned by the request-sequence check in Task 6 (`test_rapid_drag_shows_latest`).
4. **Output folder edge cases.** A relative path → 400. A missing folder → created. A folder holding old `dial_*.png` without overwrite → 409 with a message. With overwrite → only `dial_*.png`, `labels.csv` and `config.yaml` are removed, and other files survive. Pinned in Task 3 (`test_overwrite_semantics`) and Task 5 (`test_generate_rejects_relative_path`).
5. **Cancel mid-run.** After a cancel, `labels.csv` rows equal exactly the PNGs on disk (orphan PNGs from in-flight workers are deleted), and `config.yaml` has `run.completed: false`. Pinned by `test_cancel_leaves_consistent_folder` (Task 3).

---

## File Structure

| File | Responsibility |
|---|---|
| `configs/v7_default.yaml` | default parameter values only, grouped by section, with derivation comments |
| `configs/v7_schema.yaml` | per key: `type`, `min`, `max`, `step`, `help`, optional `readonly` |
| `v7_config.py` | load / merge / validate / parse / dump configs; `ConfigError` |
| `generate_synthetic_v7_mp.py` | native-gray renderer (`render_image`), section RNGs, pool driver (`generate_dataset`), CLI |
| `tools/compare_v7_look.py` | G7 statistics: new default v7 vs old luma-port v7 (commit `69b4769`) |
| `studio/backend/app.py` | FastAPI app, endpoints, static serving of the built frontend |
| `studio/backend/jobs.py` | single generation job runner (thread + cancel) |
| `studio/frontend/…` | Vite React TS app |
| `studio/run_studio.py` | launcher: builds the frontend if `dist/` is missing, starts uvicorn on 127.0.0.1 |
| `studio/README.md` | how to run (dev and normal) |
| `tests/test_v7_config.py`, `tests/test_v7_render.py`, `tests/test_v7_gray_generator.py` (rewritten), `tests/test_studio_api.py`, `tests/test_studio_e2e.py` | tests |
| `.gitignore` | add `studio/frontend/node_modules/`, `studio/frontend/dist/`, `tmp_smoke/` |

### Config shape (shared by all tasks)

`cfg: dict[str, dict[str, int | float | bool | list[int] | list[float]]]` is a nested dict, `section -> key -> value`. Ranges are 2-element lists `[lo, hi]`. Section names are fixed, and they are also the RNG stream names, plus `angle`:

`canvas, dataset, geometry, outer, ticks, digits, palette, pointer, background, cut_tip, shadows, jitter, hier_noise, tri_artifacts`

Keys that tests reference by name, which must exist exactly:
- `canvas.img_size` = 256 (readonly)
- `dataset.angle_deg_range` = `[0, 359]`
- `ticks.len_range` = `[6, 9]`
- `ticks.per_digit_range` = `[1, 5]`
- `digits.show_prob` = 0.5
- `pointer.gray_range` = `[0, 255]`
- `shadows.prob` = 0.5
- `tri_artifacts.enabled` = true
- `tri_artifacts.count_range` = `[0, 6]`
- `geometry.scale_range` = `[0.9, 1.1]`

All other old constants map 1:1 to `section.snake_case_key` following the spec's mapping table. The implementer names them.

Schema types: `int`, `float`, `bool`, `int_range`, `float_range`.

---

### Task 1: Config layer + default / schema YAML

**Files:**
- Create: `configs/v7_default.yaml`, `configs/v7_schema.yaml`, `v7_config.py`, `tests/test_v7_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces (in `v7_config.py`):
  - `DEFAULT_CONFIG_PATH: str`, `SCHEMA_PATH: str` (absolute, resolved from the module's folder)
  - `class ConfigError(ValueError)` with `.errors: list[str]`; each message starts with `"section.key: "`
  - `load_schema() -> dict` (same nested shape as cfg; leaf = `{type, min, max, step, help, readonly?}`)
  - `default_config() -> dict` (fresh deep copy each call)
  - `merge_config(base: dict, override: dict) -> dict` (deep merge, no mutation of inputs)
  - `validate_config(cfg: dict, schema: dict | None = None) -> None` (raises `ConfigError` collecting all errors: unknown section/key, wrong type, value outside `[min, max]`, range `lo > hi`, int range with non-ints, missing key)
  - `parse_config_text(text: str) -> dict` (YAML text → defaults merged with the text → validated; a YAML syntax error becomes a `ConfigError`)
  - `load_config(path: str | None) -> dict` (`None` → defaults)
  - `dump_config(cfg: dict, run: dict | None = None) -> str` (YAML text, section order preserved; when `run` is given it is appended as a top-level `run:` block)
  - `strip_run(cfg_with_run: dict) -> dict`

- [ ] **Step 1: Derive the gray defaults.** Write a throwaway script in `tmp_smoke/derive_gray.py`. It samples 100k draws of each old colour recipe from commit `69b4769` (`git show 69b4769:generate_synthetic_v7_mp.py`): `_hsv_rgb(h~U, s~DIAL_S_RANGE, v~X_V_RANGE)` for bg/circle/outer/rim, `_sample_shadow_rgb` and `_random_pointer_rgb`, and 3 × `randint(50,200)` for triangles. It converts each to ITU-R 601 luma (`(299R+587G+114B)//1000`). Use the [p1, p99] rounded ints as each `*_gray_range` default, except `pointer.gray_range = [0, 255]` (user decision). Record each derivation as a comment in `v7_default.yaml`, e.g. `# from HSV V 0.67-0.71, S 0-0.15: luma p1-p99`. Delete the script afterwards.
- [ ] **Step 2: Write `configs/v7_default.yaml` and `configs/v7_schema.yaml`.**
  - Include every old constant that is still live, under the section names above, with the shared keys exactly as listed.
  - Add a header comment listing the dropped constants and the reason for each.
  - Choose schema bounds as physical limits that contain the default: probabilities 0–1, gray 0–255, px 0–64 unless the code implies more.
  - The schema bounds must keep `2*(int(img_size//3 * scale) + canvas_margin) <= img_size` (85 × 1.15 → 97; 2 × (97 + 30) = 254). So `geometry.scale_range` max = 1.15 and `geometry.canvas_margin` bounds = [10, 30].
  - Add `.gitignore` entries.
- [ ] **Step 3: Write the failing tests** in `tests/test_v7_config.py`:
  - `test_default_matches_schema` — `validate_config(default_config())` passes, and the set of leaf keys in the default equals the set in the schema.
  - `test_named_keys_exist` — asserts the 10 shared keys and values listed in "Config shape".
  - `test_every_range_default_inside_bounds` — for each range leaf, `schema.min <= lo <= hi <= schema.max`.
  - `test_partial_override_merges` — `parse_config_text("ticks:\n  len_range: [10, 12]\n")` gives `ticks.len_range == [10, 12]`, and every other leaf equals the default.
  - `test_invalid_configs_report_key` — parametrized over these inputs, each asserting `ConfigError` and that the error text contains the given key:
    - `"ticks:\n  len_range: [9, 6]\n"` → `ticks.len_range`
    - `"jitter:\n  saturation_range: [0.7, 1.2]\n"` → `jitter.saturation_range`
    - `"digits:\n  show_prob: yes-please\n"` → `digits.show_prob`
    - `"ticks:\n  len_range: [6.5, 9]\n"` → `ticks.len_range` (int range)
    - `"ticks: [\n"` → `ConfigError` (YAML syntax)
  - `test_equal_range_ends_valid` — `[5, 5]` is accepted.
  - `test_dump_roundtrip` — `parse_config_text(dump_config(cfg)) == cfg`. `dump_config(cfg, run={"num_images": 3})` contains `run:`, and `strip_run(yaml.safe_load(...)) == cfg`.
- [ ] **Step 4: Run** `$PY -m pytest -q tests/test_v7_config.py`. Expected: FAIL (module missing).
- [ ] **Step 5: Implement `v7_config.py`** with the signatures above. Use `yaml.safe_load` / `yaml.safe_dump(sort_keys=False)`. Use `bool` checks before `int` checks, because `bool` is a subclass of `int`.
- [ ] **Step 6: Run** the same command. Expected: all pass.
- [ ] **Step 7: Commit** `configs/ v7_config.py tests/test_v7_config.py .gitignore` with the message `"v7: YAML config layer + default/schema"`.

---

### Task 2: Native-gray renderer with per-section RNG streams

**Files:**
- Modify (rewrite the render core): `generate_synthetic_v7_mp.py`
- Create: `tests/test_v7_render.py`

**Interfaces:**
- Consumes: `v7_config.default_config`, `load_schema`, the config shape.
- Produces (in `generate_synthetic_v7_mp.py`):
  - `SECTIONS: tuple[str, ...]` = the RNG section names above (without `canvas`, `dataset`) plus `"angle"`
  - `section_rngs(image_seed: int) -> dict[str, tuple[random.Random, np.random.Generator]]`, seeded via `np.random.SeedSequence([image_seed, crc32(name)])`
  - `@dataclass RenderResult: image: np.ndarray  # (img_size, img_size) uint8`, `tip_x: float`, `tip_y: float`, `angle_deg: int`
  - `render_image(cfg: dict, image_seed: int) -> RenderResult` — pure, no file I/O, no global RNG

Port note: keep the geometry, the effect pipeline order and the effect semantics of today's `generate_image` (lines 625–1013 at `69b4769`). Change only:
- **Colour model.**
  - Every RGB/HSV colour becomes one gray int drawn from its `*_gray_range`.
  - The hole/edge "darker strip" becomes `gray * U(ptr_darker_v_frac)`.
  - Per-channel noise becomes single-channel noise.
  - The `ImageEnhance.Color` step is removed.
  - Brightness/Contrast/Sharpness apply to the `"L"` image.
- **Layers.** RGBA layers become an `"L"` value image plus an `"L"` alpha mask, composited by a helper `_composite_gray(base: Image L, value: Image L | int, alpha: Image L) -> Image L` that computes `round(base*(255-a)/255 + value*a/255)`.
- **Randomness.** Each draw uses the stream of the section that owns the parameter:
  - `angle`: `angle_deg = rng.randint(*dataset.angle_deg_range)`
  - `geometry`: scale, displacement, hub, chord, tip var, pointer displacement, hole ratio/displacement, stretch
  - `outer`: outer scale
  - `ticks`: tick jitter, grey, count, per-tick len/width, plus the arc gap
  - `digits`: show prob, font, jitter, inside/outside
  - `palette`: bg/circle/outer/rim gray levels
  - `pointer`: gray, darker fracs, gradient, noise, edge
  - `background`: low-freq + pixel noise
  - `cut_tip`: radius
  - `shadows`: has_shadow, light angle, gray levels, offset, opacity, rim shadow
  - `jitter`: brightness/contrast/sharpness
  - `hier_noise`: block + pixel noise
  - `tri_artifacts`: count, placement, gray, noise
- **Cross-section values.** A value one section needs from another (e.g. `dial_radius`, pointer mask) is passed as data, never re-drawn.

- [ ] **Step 1: Write the failing tests** in `tests/test_v7_render.py`:
  - `test_render_shape_and_dtype` — the image is `(256, 256)` uint8, `0 <= tip_x, tip_y < 256`, and `0 <= angle_deg <= 359`.
  - `test_render_deterministic` — the same `(cfg, seed)` twice gives an identical array and tip.
  - `test_no_colour_code_in_v7` — the source text of `generate_synthetic_v7_mp.py` contains none of `colorsys`, `"RGB"`, `"RGBA"`, `hsv`.
  - `test_tick_change_isolated` — for seeds 0..7, render the default and a copy with `ticks.len_range=[12, 14]` and `ticks.per_digit_range=[5, 5]`:
    - `angle_deg` and tips are equal;
    - pixels inside the disc `r < 0.45 * 256/3` around the image centre are equal;
    - at least one image differs somewhere.
  - `test_tri_change_isolated` — the same check with `tri_artifacts.enabled=False`: angle and tips are equal, and pixels on the pointer mask region (approximate with the disc `r < 0.3*256/3`) are equal unless a triangle overlapped. Use seeds where the default has `count >= 1`, and assert the images differ.
  - `test_pointer_gray_change_moves_only_pointer` — set `pointer.gray_range=[255,255]` vs `[0,0]`. Angle and tips are equal, and the mean of pixels in a ring `0.75..0.95 * dial` far from the pointer angle differs by < 1.0.
  - `test_render_extremes_do_not_crash` — build two configs: every numeric leaf at its schema min (ranges `[min, min]`), and every leaf at its schema max (ranges `[max, max]`). Render seeds 0..4 for each, with no exception, and the shape is `(256, 256)`.
- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_v7_render.py`. Expected: FAIL (`render_image` missing).
- [ ] **Step 3: Rewrite the render core** of `generate_synthetic_v7_mp.py` per the port note. Then delete `generate_image`, `_hsv_rgb`, `_random_pointer_rgb`, `_luminance_strip_color` (RGB) and the module constants block. Keep `_seed_for_index`. Update the module docstring: native gray, YAML-driven, the per-section streams, and an ASCII diagram of the pipeline order.
- [ ] **Step 4: Run** the same command. Expected: all pass. View 4 rendered tiles saved to `tmp_smoke/` with the Read tool to confirm they look like the old v7 dials, then delete them.
- [ ] **Step 5: Commit** with the message `"v7: native grayscale renderer with per-section RNG streams"`.

---

### Task 3: Dataset driver + CLI

**Files:**
- Modify: `generate_synthetic_v7_mp.py` (driver + `main`)
- Rewrite: `tests/test_v7_gray_generator.py` (drop the v2_mp parity test, which the spec makes obsolete)

**Interfaces:**
- Consumes: `render_image`, `_seed_for_index`, `v7_config.load_config`, `dump_config`.
- Produces:
  - `class OutputDirNotEmpty(RuntimeError)`
  - `generate_dataset(cfg: dict, output_dir: str, num_images: int, seed: int = 0, workers: int | None = None, chunksize: int = 4, overwrite: bool = False, progress_cb: Callable[[int, int], None] | None = None, cancel_event: threading.Event | None = None) -> dict` returns `{"completed": bool, "done": int, "output_dir": str}`. It creates the folder if missing.
    - If the folder contains any `dial_*.png`, `labels.csv` or `config.yaml` and `overwrite` is False, it raises `OutputDirNotEmpty`. With overwrite, it deletes exactly those files.
    - It writes `config.yaml` = `dump_config(cfg, run={...})`, where the run block has `num_images, seed, workers, chunksize, generator: "generate_synthetic_v7_mp.py", completed, finished_at` (ISO8601).
    - The worker pool is initialized with `(output_dir, seed, cfg)`, and the worker returns the label row.
    - `labels.csv` rows are written in index order (`imap`).
    - It checks `cancel_event` after each result. On cancel: `pool.terminate()`, delete every `dial_*.png` not in the written rows, then write `config.yaml` with `completed: false`.
  - CLI: `--config PATH` (optional; defaults otherwise), `--output-dir` (default `sv7`), `--num-images` (default 30000), `--seed` (0), `--workers` (cpu_count), `--chunksize` (4), `--overwrite`. A `ConfigError` prints its errors and exits 2.

- [ ] **Step 1: Write the failing tests** in `tests/test_v7_gray_generator.py`. All of them run the script through `subprocess.run([PY, "generate_synthetic_v7_mp.py", ...], cwd=repo, timeout=300)`, except the cancel test, which calls `generate_dataset` from a thread.
  - `test_cli_outputs` — 6 images, `--workers 2`, seed 3: rc 0; 6 rows; mode `"L"`, 256×256; tips in range; `config.yaml` exists, and `strip_run` of it equals `default_config()`, with `run.completed is True` and `run.num_images == 6`.
  - `test_workers_bit_exact` — `--workers 1` vs `--workers 4`: identical PNG bytes-as-arrays and identical `labels.csv`.
  - `test_preview_equals_disk` — for idx 0..5, `render_image(default_config(), _seed_for_index(3, idx)).image` equals the array of `dial_{idx:04d}.png`.
  - `test_cli_partial_config` — the YAML file `ticks:\n  len_range: [12, 14]\n` gives rc 0, and `config.yaml` has `ticks.len_range == [12, 14]`.
  - `test_cli_bad_config` — the YAML `ticks:\n  len_range: [9, 6]\n` gives rc 2, and stderr+stdout contains `ticks.len_range`.
  - `test_overwrite_semantics` — the folder holds `dial_0099.png`, `labels.csv`, `notes.txt`:
    - without `--overwrite`: rc ≠ 0, and the message mentions `--overwrite`;
    - with it: rc 0, `dial_0099.png` is gone and `notes.txt` is still there.
  - `test_cancel_leaves_consistent_folder` — run `generate_dataset(default_config(), tmp, 400, workers=4, cancel_event=ev, progress_cb=cb)` where `cb` sets `ev` when `done >= 20`. Then:
    - the result has `completed is False`;
    - the set of PNG names == the set of `labels.csv` filenames;
    - `config.yaml` has `run.completed is False`.
- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_v7_gray_generator.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `generate_dataset`, the worker init/task with cfg, and `main` per the interfaces.
- [ ] **Step 4: Run** it again. Expected: all pass. Also run `$PY -m pytest -q tests/test_v7_config.py tests/test_v7_render.py` (still green).
- [ ] **Step 5: Commit** with the message `"v7: YAML-driven dataset driver, CLI, cancel/overwrite semantics"`.

---

### Task 4: Look-match check against the old v7 (G7)

**Files:**
- Create: `tools/compare_v7_look.py`, `tests/test_v7_look.py`

**Interfaces:**
- Produces: `compare(n: int = 500, seed: int = 0, workdir: str = "tmp_smoke/look") -> dict[str, dict[str, float]]`. It extracts the old v7 via `git show 69b4769:generate_synthetic_v7_mp.py` into `workdir/old_v7.py`, generates `n` images with each generator, and returns one entry per metric with keys `old`, `new` and `diff`. It also prints a table.
  - **Metrics**, using the labels and canvas centre:
    - mean and std of the whole image;
    - mean of the dial face annulus `0.55..0.9 R`, excluding the pointer wedge `±15°`;
    - mean of the bezel ring `1.05..1.2 R`;
    - mean of the pointer interior, sampled along the centre→tip line at 0.3–0.7 of its length;
    - median of `|pointer − dial face|`, plus the fraction of images where it is < 15.
  - `R = 256//3`.

- [ ] **Step 1: Write the test** `test_look_matches_old_v7` (marked `@pytest.mark.slow`; register the marker in `conftest.py` or `pytest.ini` if missing). It calls `compare(n=300)` and asserts:
  - `|diff|` ≤ 8 gray levels for each mean metric;
  - ≤ 6 for the std metric;
  - ≤ 0.05 for the low-contrast fraction.
- [ ] **Step 2: Run** `$PY -m pytest -q -m slow tests/test_v7_look.py` (background, with a 20-minute timeout). If it fails, adjust only the offending `*_gray_range` defaults in `configs/v7_default.yaml`, update their derivation comments, and rerun Tasks 1–3 tests.
- [ ] **Step 3: Save the printed table** into the commit message body for the report to the user.
- [ ] **Step 4: Commit** with the message `"v7: look-match check vs luma-port v7"`.

---

### Task 5: Studio backend (FastAPI)

**Files:**
- Create: `studio/__init__.py`, `studio/backend/__init__.py`, `studio/backend/app.py`, `studio/backend/jobs.py`, `tests/test_studio_api.py`

**Interfaces:**
- Consumes: `v7_config.*`, `render_image`, `_seed_for_index`, `generate_dataset`, `OutputDirNotEmpty`.
- Produces: `create_app(frontend_dist: str | None = None) -> FastAPI`. It mounts `dist` at `/` when given, and the API under `/api`. All JSON:
  - `GET /api/schema` → the schema dict.
  - `GET /api/config/default` → `{"config": cfg}`.
  - `POST /api/config/parse` `{"text": str}` → `{"config": cfg}`, or 422 `{"errors": [str]}`.
  - `POST /api/config/dump` `{"config": cfg}` → `{"text": str}`, or 422.
  - `POST /api/preview` `{"config": cfg, "seeds": [int] (1..16)}` → `{"tiles": [{"seed", "png_b64", "tip_x", "tip_y", "angle_deg"}]}`, or 422 `{"errors": [...]}`.
    - Rendering runs in a module-level `ProcessPoolExecutor(max_workers=min(8, cpu))` created on startup and shut down on shutdown. It is separate from job pools.
    - Each tile is `render_image(cfg, seed)`. A preview seed is used directly as the image seed, and the generate panel's run uses `_seed_for_index`.
  - `POST /api/generate` `{"config", "output_dir", "num_images", "seed", "workers", "overwrite"}` → 202 `{"state": "running"}`.
    - 400 if `output_dir` is not absolute.
    - 409 if a job is running, or on `OutputDirNotEmpty` (message mentions overwrite).
    - 422 for a bad config.
  - `GET /api/jobs/current` → `{"state": "idle"|"running"|"done"|"cancelled"|"error", "done": int, "total": int, "output_dir": str|null, "error": str|null}`.
  - `POST /api/jobs/current/cancel` → current job dict.
- `jobs.py`: `class JobRunner` with `start(cfg, output_dir, num_images, seed, workers, overwrite) -> None`, `status() -> dict` and `cancel() -> dict`. It runs `generate_dataset` in a `threading.Thread`, with `progress_cb` updating counters under a lock. `OutputDirNotEmpty` is checked synchronously before the thread starts, so the HTTP call can return 409.

- [ ] **Step 1: Write the failing tests** in `tests/test_studio_api.py`, using `fastapi.testclient.TestClient(create_app())` as a context manager so startup/shutdown run:
  - `test_schema_and_default` — the two endpoints return dicts with `ticks.len_range` present.
  - `test_preview_tiles` — 3 seeds → 3 tiles. Each `png_b64` decodes to an `"L"` 256×256 PNG equal to `render_image(cfg, seed).image`.
  - `test_preview_invalid_config_422` — `ticks.len_range=[9, 6]` → 422, and some error contains `ticks.len_range`.
  - `test_parse_and_dump_roundtrip`.
  - `test_generate_rejects_relative_path` → 400.
  - `test_generate_job_runs` — absolute `tmp_path/out`, 8 images, workers 2. Poll `/api/jobs/current` until `done` (timeout 120 s): `done == total == 8`, and 8 PNGs exist.
  - `test_generate_conflict_and_cancel` — start 300 images, then:
    - a second start → 409;
    - cancel → the state ends as `"cancelled"`;
    - the PNG set equals the labels set.
- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_studio_api.py`. Expected: FAIL.
- [ ] **Step 3: Implement `app.py` and `jobs.py`** per the interfaces.
- [ ] **Step 4: Run** it again. Expected: all pass.
- [ ] **Step 5: Commit** with the message `"studio: FastAPI backend (preview pool, generation job)"`.

---

### Task 6: Studio frontend + launcher + browser E2E

**Files:**
- Create: `studio/frontend/` (Vite React TS: `package.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api.ts`, `src/components/ParamControl.tsx`, `src/components/PreviewGrid.tsx`, `src/components/GeneratePanel.tsx`, `src/styles.css`), `studio/run_studio.py`, `studio/README.md`, `tests/test_studio_e2e.py`

**Interfaces:**
- Consumes: Task 5's endpoints.
- Produces (DOM contract used by the E2E test):
  - one `<section data-section="{name}">` per config section, with a heading;
  - one control per leaf with `data-param="{section}.{key}"`:
    - scalar → `<input type="range">` plus `<input type="number">`;
    - range → two `<input type="range">` (`data-handle="lo"` / `"hi"`) plus two number inputs;
    - bool → `<input type="checkbox">`;
    - readonly → disabled;
  - each control shows `help` text;
  - preview `<div data-testid="preview-grid">` holding 9 `<figure data-testid="tile" data-seed=…>`, each with `<img>` and, when the overlay is on, an `<svg data-testid="tip-marker">` positioned at `tip_x, tip_y` (scaled to the displayed size);
  - `<input type="checkbox" data-testid="overlay-toggle">`, `<button data-testid="reshuffle">`;
  - `<input type="file" data-testid="load-yaml">`, `<button data-testid="save-yaml">` (downloads `v7_config.yaml` via a Blob);
  - `<div data-testid="config-errors">` showing 422 errors;
  - generate panel: `data-testid` `gen-output-dir`, `gen-num-images`, `gen-seed`, `gen-workers`, `gen-overwrite`, `gen-start`, `gen-cancel`, a `<progress data-testid="gen-progress">` and `<div data-testid="gen-status">` showing the state text and errors.
- Behaviour:
  - The preview is debounced at 150 ms.
  - Each request carries an incrementing sequence number, and a response is applied only if its number is the latest sent (Review Focus 3). The newest number is exposed as `data-applied-seq` on the grid for the test.
  - The job is polled every 500 ms while running.
- `studio/run_studio.py`: arguments `--port` (default 8765) and `--no-browser`.
  - If `studio/frontend/dist/index.html` is missing, it runs `npm ci` and `npm run build` in `studio/frontend` (`shell=True` on Windows).
  - It starts uvicorn with `create_app(frontend_dist=dist)` on `127.0.0.1`, and opens the browser unless `--no-browser`.
  - Dev mode is documented in the README: `npm run dev` with a Vite proxy of `/api` → `127.0.0.1:8765`.

- [ ] **Step 1: Scaffold the Vite React TS app.** Run `npm create vite@5 frontend -- --template react-ts` in `studio/`, then `npm i` and add a range-slider lib if wanted. Commit `package-lock.json`. Confirm Playwright Chromium is installed (`$PY -m playwright install chromium`).
- [ ] **Step 2: Write the failing E2E test** in `tests/test_studio_e2e.py`.
  - A module fixture starts `run_studio.py --port 8799 --no-browser` as a subprocess, waits for `GET /api/schema` (timeout 300 s, to allow the build) and kills it at teardown.
  - The tests use Playwright sync Chromium headless at viewport 1400×900 and drive the page only through clicks, typing, `fill` and `set_input_files`:
    - `test_page_shows_preview` — 9 tile images load, each with naturalWidth 256.
    - `test_every_param_has_control` — for each leaf in `/api/schema`, the `[data-param="…"]` locator count is ≥ 1.
    - `test_tick_slider_changes_only_ticks`:
      1. Record the tiles' `src`.
      2. Set `ticks.len_range` hi to 14 and lo to 12 via the number inputs (`fill` + Enter).
      3. Wait for `data-applied-seq` to increase.
      4. At least one tile's `src` changed.
      5. Every tile's tip-marker position is unchanged with the overlay on.
    - `test_overlay_marker_on_apex` — toggle the overlay. 9 markers are visible. Each marker's centre is within 3 display px of `(tip_x, tip_y) * display_scale`, where the tips come from the `/api/preview` response intercepted with `page.expect_response`.
    - `test_rapid_drag_shows_latest` — type 10 successive values into `ticks.len_range` hi within 1 s. Afterwards the grid's `data-applied-seq` equals the last sent seq, and the hi number input shows the last value.
    - `test_invalid_yaml_shows_error` — `set_input_files` with a YAML containing `ticks:\n  len_range: [9, 6]\n`. `config-errors` contains `ticks.len_range`, and the tiles stay rendered.
    - `test_save_load_roundtrip`:
      1. Change `digits.show_prob` to 1.
      2. Click save (`page.expect_download`).
      3. Reload the page.
      4. Load the downloaded file.
      5. The `digits.show_prob` number input shows `1`.
    - `test_generate_and_cancel`:
      1. Generate 20 images to an absolute `tmp_path/out`; `gen-progress` reaches 20/20, and `gen-status` shows `done`.
      2. The folder has 20 PNGs, a 20-row `labels.csv`, and a `config.yaml` whose `strip_run` equals the config from the saved YAML of the current UI state.
      3. Generate 400 to `tmp_path/out2`, click cancel after progress > 0. The status shows `cancelled`, and `config.yaml` has `completed: false`.
- [ ] **Step 3: Run** `$PY -m pytest -q tests/test_studio_e2e.py` (background, with a 15-minute timer). Expected: FAIL.
- [ ] **Step 4: Implement the components** to the DOM contract, `run_studio.py` and the README (run commands, dev mode, where outputs go, an ASCII diagram of UI → API → renderer).
- [ ] **Step 5: Run** it again. Expected: all pass. Then manually open the studio with the dev-browser / Playwright, take one screenshot to `tmp_smoke/studio.png`, look at it, and delete it.
- [ ] **Step 6: Commit** (no `node_modules` or `dist`) with the message `"studio: React frontend, launcher, browser E2E"`.

---

### Task 7: Final verification + training smoke

- [ ] **Step 1: Run** `$PY -m pytest -q tests/test_v7_config.py tests/test_v7_render.py tests/test_v7_gray_generator.py tests/test_studio_api.py tests/test_studio_e2e.py tests/test_train_gray.py` (background, 20-minute timer). Expected: all pass.
- [ ] **Step 2: Train a few epochs on new data.** Run `$PY generate_synthetic_v7_mp.py --output-dir tmp_smoke/sv7 --num-images 2000 --overwrite`, then `$PY train_e_inv_res_gray.py --data-dir tmp_smoke/sv7 --epochs 10 --lr 1e-3 --model-name tmp_smoke/m --num-workers 4` (background, 30-minute timer). Expected: exit 0 and val mean decreasing. Record the numbers for the report.
- [ ] **Step 3: Clean up and commit.** Delete `tmp_smoke/` and `diag/`, then commit any doc touch-ups.
