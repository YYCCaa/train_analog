# v7 generator: native grayscale + YAML config + tuning studio app — design

Date: 2026-10-23
Status: draft, awaiting approval

## Intent / why

Today `generate_synthetic_v7_mp.py` is a luma port of the RGB generator
`generate_synthetic_v2_mp.py`: it renders in RGB with HSV colour sampling,
then calls `convert("L")` at the very end. Its ~65 tunable constants are
module-level Python globals, so changing the synthetic distribution means
editing code, generating a dataset and eyeballing PNGs.

The user wants three things:

1. **A native grayscale v7.** It renders in gray from the start, with no
   RGB render and no colour-to-gray conversion. Every surface (dial face,
   rim, bezel, pointer, hole, edge, shadows, ticks, digits, artifacts) is
   given a gray level.
2. **Generator + YAML.** Every tunable lives in a YAML file. The generator
   is code only and reads a config.
3. **A full-stack tuning app** with a slider or toggle for every changeable
   parameter. It offers:
   - **Live preview (in RAM).** Dial images are rendered in memory and shown
     as soon as a control moves. Nothing is written to disk.
   - **Generate (to disk).** A full dataset generation is launched from the
     app with the current settings.

Why: the user tunes the synthetic distribution often, wants to *see* the
effect of each knob immediately, and wants the grayscale data to be
grayscale by construction rather than a by-product of an RGB pipeline.

## Decided by the user

- **Native grayscale rendering.** v7 no longer derives from the RGB
  pipeline. Bit-exact parity with v2_mp is dropped.
- **Per-section random streams.** Each feature section draws from its own
  seeded stream, so moving one slider changes only that feature on the
  preview tiles.
- **Default gray levels match today's v7 look.** Each old colour range is
  translated into the gray range it actually produced, so new datasets are
  statistically like the current `sv7`, just generated natively. The
  pointer gray level is uniform over 0–255.
- **Both preview and generation live in the app.** Preview is in RAM.
  Generation writes to disk.
- **Stack:** FastAPI backend plus a React (Vite) frontend.
- **One preview extra: a tip-label overlay toggle.** No RGB view, no
  per-parameter reset or diff badges, and no grid-size control. The user
  was offered these and declined.

## Goals

- **G1. Native gray renderer.** All drawing happens on single-channel
  images: PIL `"L"`, plus `"LA"` or separate `"L"` alpha masks for layers
  that are composited. Every colour parameter is a gray level (int 0–255)
  or a gray range. There is no RGB/HSV code path, and no `convert("L")` of
  an RGB image anywhere in v7. Output stays 256×256 uint8 single-channel
  PNG, with `labels.csv` columns `filename, angle_deg, tip_x, tip_y`.
- **G2. YAML-driven.** The generator reads all tunables from a YAML file
  (`--config path.yaml`). A committed default YAML holds the defaults. A
  partial YAML is merged over the defaults. The config is validated:
  unknown key, wrong type, or min > max all raise a clear error.
- **G3. Per-section RNG streams.** Each section gets its own RNG, seeded
  from `(image seed, section name)`, e.g. via `np.random.SeedSequence`.
  The streams are separate `random.Random` / `np.random.Generator`
  objects. Changing a parameter in section A never changes the random
  draws of section B for the same image seed.
  - The pointer angle is drawn from its own stream (`angle`), so it is
    stable across all slider moves.
  - Sections that must share a value get it passed as data, not re-drawn.
    For example, dial geometry feeds pointer placement.
- **G4. One render path.** A public function such as
  `render_image(cfg, seed) -> (gray uint8 256×256, tip_x, tip_y, angle_deg)`
  does no file I/O. The CLI pool workers and the app (preview and
  generation) all call it. A preview tile with seed S is pixel-identical
  to the on-disk image with seed S under the same config.
- **G5. Determinism.** For a given config, `--seed` and image index, the
  output is bit-exact regardless of `--workers` or scheduling. The
  per-image seed comes from `_seed_for_index(seed, idx)`, as today.
- **G6. App** (FastAPI + React/Vite):
  - **Controls.** Every YAML parameter has a control, grouped by YAML
    section:
    - a numeric scalar gets a slider plus a numeric entry;
    - a `[min, max]` pair gets a two-handle range slider;
    - a bool gets a toggle.
    Each control shows the one-line description from the schema.
  - **Live preview grid** of about 9 tiles. It re-renders on change
    (debounced), with fixed seeds plus a "reshuffle" button.
  - **Tip-label overlay toggle.** A marker is drawn client-side at
    `(tip_x, tip_y)` on each tile. It never alters the image.
  - **YAML files.** Load a YAML file, and save the current settings as YAML.
  - **Generate panel.** Inputs are an absolute output folder, number of
    images, seed, workers and an "overwrite" tick box. A background job
    runs and shows a progress bar. The job can be cancelled.
- **G7. Look matches today's v7.** With the default YAML, per-surface gray
  statistics match the current luma-port v7 within a loose tolerance
  (verification below). Pixel identity is not expected.

## Non-goals

- No change to geometry, pointer shape, the order of the effect pipeline,
  or what each effect does. Only the colour model (RGB/HSV → gray) and the
  RNG plumbing change.
- No training integration, auth, multi-user support or remote deployment.
  This is a local single-user tool bound to 127.0.0.1.
- No YAML port of v2, v3, v4 or v5.

## Anti-goals (a failure even if it "works")

- Preview and disk generation drifting apart, i.e. two render paths.
- A slider that also re-rolls unrelated features on the preview tiles.
- A parameter in the YAML with no UI control, or a control that does
  nothing.
- Preview lagging badly, or blocking while a generation job runs.
- The app writing into the repo tree by surprise.
- Any RGB intermediate sneaking back into v7.

## Constraints

- **Environment.** Windows 11, with a Python venv at repo `.venv`. FastAPI
  0.115, uvicorn and PyYAML are already installed. Node 22 and npm 10 are
  available.
- **Windows multiprocessing uses `spawn`.** The config is passed to
  workers explicitly through initializer args, never through mutated
  globals.
- **User comment rule.** At each declaration, say what it is plus its
  shape and dtype.
- **User testing rule.** Tests imitate a human. The generator is tested
  through CLI runs. The app is tested by driving the real frontend in a
  browser (move controls, see the preview change, run a small generation),
  not only by hitting the API.

## Parameter mapping (old v7 → native gray YAML)

The YAML is grouped by the existing `# ── section ──` comments in today's
v7, and those sections are also the RNG stream boundaries (G3).

| Old (colour) | New (gray) | Default derivation |
|---|---|---|
| `DIAL_S_RANGE`, dial hue | removed | gray has no hue or saturation |
| `BG_V_RANGE`, `CIRCLE_V_RANGE`, `OUTER_V_RANGE`, `RIM_V_RANGE` (HSV V at low S) | `*_gray_range` [int,int] | empirical luma range the old code produced for that surface (≈ V×255 adjusted for S ≤ 0.15) |
| `_random_pointer_rgb` (H,S,V uniform) | `pointer.gray_range` [0,255] | user decision: uniform 0–255 |
| `PTR_DARKER_V_FRAC` (hole/edge darker strip) | same fraction, applied to the gray level | unchanged numbers |
| `SHADOW_S_RANGE`, `SHADOW_V_RANGE` | `shadow.gray_range` | empirical luma of the old shadow colours |
| `TICK_GREY_RANGE`, `DIGIT_GREY_RANGE` | unchanged (already gray) | unchanged |
| `TRI_ARTIFACT_MEAN_RANGE` (per channel) | one gray mean range | empirical luma of 3 independent channel means |
| `SATURATION_RANGE` (ImageEnhance.Color) | removed | has no meaning in gray |
| `BRIGHTNESS_RANGE`, `CONTRAST_RANGE`, `SHARPNESS_RANGE` | unchanged | unchanged |
| `NOISE_INTENSITY`, `BG_VARIATION`, `HIER_NOISE_*` (per channel) | single-channel noise, same amplitudes | unchanged numbers |
| all geometry / ticks / digits / blur / cut-tip / shadow-geometry constants | unchanged | unchanged |

"Empirical luma range" means the builder measures it by sampling the old
colour code, e.g. 100k draws through `colorsys` + ITU-R 601 luma, and
takes the range or percentiles. The builder records how each default was
derived in a YAML comment.

More inventory rules:

- **Dataset section.** It keeps only `angle_deg_range` (default
  `[0, 359]`).
- **`img_size`.** Shown as 256, read-only in the UI, because downstream
  training hard-codes 256. Derived values (`CENTER`, `DIAL_RADIUS_BASE`,
  `DIGIT_DIST`) are computed, not stored.
- **Dropped dead constants.** `BLUR_RANGE`, `RIM_SHADOW_SECONDARY_OPACITY`,
  `VALUE_MIN`, `VALUE_MAX` and `CENTER` are never used by the render, so
  they are dropped. They are listed in a YAML comment.
- **Int vs float.** Integer ranges stay ints, so `randint`-style draws are
  used, and float ranges stay floats. The schema records the type.
- **Schema file.** UI metadata (type, slider min/max/step, one-line
  description) lives in a separate schema file. The user-facing YAML holds
  values only. Slider bounds are sensible physical limits, e.g. 0–1 for
  probabilities and 0–255 for gray levels, and always contain the default.

## Run settings vs config (builder-check calls, made by Claude)

These were settled during design review without the user:

- **Render config vs run settings.** The YAML holds render parameters and
  `angle_deg_range` only. Output folder, image count, seed and workers are
  run settings. They come from CLI flags (`--output-dir`, `--num-images`,
  `--seed`, `--workers`) or from the Generate panel.
- **Output `config.yaml`.** Every generated folder gets one: the full
  merged render config plus a `run:` block with `num_images`, `seed`,
  `workers`, `generator: generate_synthetic_v7_mp.py`, `completed:
  true|false` and a timestamp. One file is enough to reproduce a dataset.
  Verification compares the render section with the YAML the UI saved.
- **Output folder.** The app requires an absolute output path. The CLI
  default stays `sv7`.
- **Overwrite.** When ticked, the job first deletes `dial_*.png`,
  `labels.csv` and `config.yaml` in the target folder, and nothing else.
  Unticked, it refuses a folder that already contains any of those files.
- **Cancel.** Finished PNGs are kept and `labels.csv` holds exactly their
  rows. `config.yaml` is written with `completed: false`, so a partial
  dataset is never mistaken for a full one.
- **Jobs.** One generation job runs at a time. Starting a second one is
  rejected with a clear message.
- **Preview isolation from jobs.** Preview renders run in a pool separate
  from the generation job's pool, so preview stays responsive during
  generation.

## Architecture (builder's call within these lines)

```
  v7_default.yaml (values)               v7_schema.yaml (types, bounds, help)
          │                                        │
          ▼                                        ▼
  generate_synthetic_v7_mp.py  ◄──imports──  FastAPI backend ─────────► React/Vite UI
   load_config(path) -> cfg                   GET  /api/schema, /api/config/default
   render_image(cfg, seed)                    POST /api/preview  {cfg, seeds} -> PNG b64 + tips
     -> (uint8 256x256, tip, angle)           POST /api/config/parse  (YAML text -> cfg / errors)
     per-section RNG streams                  POST /api/generate {cfg, out, n, seed, workers, overwrite}
   CLI: --config --output-dir ...             GET  /api/jobs/current   POST /api/jobs/current/cancel
   pool driver (used by CLI + app job)
```

- **Folder layout.** The app lives in a new top-level `studio/` folder with
  `backend/` (FastAPI) and `frontend/` (Vite React).
- **Running it.** One documented command starts both for development. For
  a normal run, the backend serves the built frontend.
- **Old luma-port v7.** It is replaced in place. It stays in git history,
  and the verification baseline uses it from there.

## Left to the builder

- Exact YAML key naming, file names and casing.
- Section stream seeding details.
- Preview transport, debounce interval and slider component library.
- The launch script.
- Styling.

## Verification

- **Determinism.** CLI with `--workers 1` vs `--workers 4`: identical PNGs
  and `labels.csv`.
- **Preview = disk.** `render_image(cfg, seed_for_index(s, i))` is
  pixel-equal to the CLI's `dial_i.png`.
- **Stream isolation.** Change only a tick parameter. For the same seeds,
  the pointer angle, tip label, and pixels outside the tick annulus/overlay
  stay identical. The same check is repeated for one other section, e.g.
  triangle artifacts.
- **Native gray.** A test or grep confirms no RGB/HSV path in v7
  (`colorsys`, `"RGB"`, `"RGBA"`, `convert("L")` of an RGB image).
- **Look matches old v7 (G7).** Generate about 500 images with the default
  YAML and about 500 with the old v7 from git history. Compare per-region
  gray statistics: mean/std of the dial face, bezel, pointer region and
  whole image, plus the pointer-vs-dial contrast distribution. Agreement
  must be within a stated tolerance, and the numbers are reported to the
  user.
- **Config.** A partial YAML override changes only the intended aspect. An
  invalid YAML gives a clear error.
- **App (human-like, in a browser):**
  1. Open the page; preview tiles appear.
  2. Drag the tick-length slider; only the ticks change on the tiles.
  3. Toggle the tip overlay; markers sit on the apexes.
  4. Save the YAML, then load it back.
  5. Generate 20 images into a temp folder; progress reaches 100%.
  6. The folder holds 20 PNGs, `labels.csv`, and a `config.yaml` whose
     render section equals the saved YAML.
  7. A cancel mid-run leaves `completed: false`.
- **Training smoke.** `train_e_inv_res_gray.py` trains a few epochs on a
  small new-v7 dataset without error.
