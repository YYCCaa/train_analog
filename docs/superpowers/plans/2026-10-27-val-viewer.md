# Validation Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app (`python -m val_viewer`) that shows the best and worst n TipNet v9 validation images, each next to its heatmap, with the true and predicted tip marked on both.

**Architecture:** A new package `val_viewer/` with a FastAPI backend that imports `train_v9_tip.py` for data, split, model and decode, ranks the val split once (cached on disk), and renders dial / heatmap PNGs on request. A plain HTML/CSS/JS page (no build step, no CDN) draws two columns with SVG marker overlays and a detail view.

**Tech Stack:** Python, torch, numpy, Pillow, FastAPI + uvicorn; vanilla JS; pytest + FastAPI TestClient + Playwright Chromium.

**Spec:** `docs/superpowers/specs/2026-10-27-val-viewer-design.md`

## Global Constraints

- Create files only under `val_viewer/`; do not edit any existing file (`train_v9_tip.py` is imported, never edited).
- Never open, list or read anything under `../train_dial_data` or `C:/Users/Administrator/PycharmProjects/train_dial_data` (a guard blocks it); tests use a fake dataset in `tmp_path`.
- Install nothing; use what the venv has: torch, numpy, pillow, fastapi, uvicorn, httpx, pytest, playwright.
- Run everything from the repo root; `train_v9_tip` is imported as a top-level module (`import train_v9_tip as tv`).
- Every variable / attribute / constant declaration gets a comment saying what it holds, its shape and dtype (repo style, see `train_v9_tip.py`).
- Numbers must match the trainer: `tv.split_indices(N, val_split)` (seed 42), `tv.build_model("tipnet")` in eval mode, `tv.to_input`, `tv.decode_logits`, `torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False`, no autocast.
- conf = sum of the global softmax probabilities in the 5 x 5 cells around the argmax cell, window clipped to the map.
- Coordinates: image pixel i is centred at coordinate i; on screen a point (x, y) is drawn at (x + 0.5, y + 0.5) in an SVG `viewBox="0 0 256 256"`. The 128 x 128 heatmap scaled x2 (nearest neighbour) fills the same 256 x 256 box, so both panels use the same marker coordinates.
- Markers: true tip = green crosshair with an open centre + circles r = 3 and r = 7 (image px); predicted tip = orange x; a thin line from predicted to true tip; every marker has a dark outline. The heatmap colour map is one hue, dark (low) to light (high), and contains neither marker hue.
- Heatmap scales: `log` (default) = log10 probability over the 6 decades below the map maximum, values below clipped; `linear` = probability / map maximum.
- Defaults: data dir `tv.DEFAULT_DATA_DIR`, run dir `tv.DEFAULT_RUNS_DIR + "/v9_tipnet_a"`, val split 0.15, cache dir `val_viewer/.cache/` (gitignored by `val_viewer/.gitignore`), device auto (CUDA if available), host 127.0.0.1, port 8020, n default 24, n range 1..1000.
- Weights: `<run>/tip.pth`, else `<run>/best_p99.pth`; neither -> exit code 2.

## Review Focus

1. The first start on 22,500 real images must show progress, and a second start must reuse the cache: the CLI prints `[rank]` progress lines on a cold start and a `[rank] cached` line on a warm one (Task 1, test 10).
2. A page with n = 1000 fires up to 2,000 heatmap requests at once: concurrent requests must all succeed with the model behind a lock (Task 1, test 9).
3. Retraining or swapping weights must not show stale rankings: a changed weights file or labels.csv recomputes the ranking (Task 1, test 7).
4. A tip near the image border: the detail-view zoom window is clamped to the image and the marker still lands on the label (Task 2, test 7).
5. Half-pixel alignment between markers and pixels, on both panels and in the zoom (Task 1, test 6; Task 2, tests 5 and 7).

---

### Task 1: Backend — data, ranking cache, rendering, API, CLI

**Files:**
- Create: `val_viewer/__init__.py`, `val_viewer/__main__.py`, `val_viewer/.gitignore` (`.cache/`)
- Create: `val_viewer/app/__init__.py`, `val_viewer/app/data.py`, `val_viewer/app/render.py`, `val_viewer/app/main.py`
- Create: `val_viewer/static/index.html` (placeholder: `<title>Val viewer</title>` and an empty body; Task 2 replaces it)
- Test: `val_viewer/tests/__init__.py`, `val_viewer/tests/conftest.py`, `val_viewer/tests/test_api.py`

**Interfaces:**
- Produces (`val_viewer/app/data.py`):
  - `resolve_weights(run_dir: str | Path) -> Path` — `tip.pth`, else `best_p99.pth`, else `FileNotFoundError`.
  - `class ValData(data_dir, run_dir, val_split=0.15, cache_dir=None, device="auto", batch=256)` — raises `FileNotFoundError` for a missing data dir / labels.csv / weights. Builds the trainer cache with `tv.build_cache` when `tv.cache_is_valid` is false. Attributes: `n_val: int`, `names: list[str]` (n_val,), `label: (n_val, 2) float64`, `tip: (n_val, 2) float64`, `err: (n_val,) float64`, `conf: (n_val,) float64`, `row: (n_val,) int64` (dataset row of each val position), `cache_hit: bool`, `weights_path: Path`, `run_name: str` (run folder name), `device: str`. Methods: `image(pos: int) -> (256, 256) uint8`, `logits(pos: int) -> (128, 128) float64` (one forward pass, thread-safe), `entries(side: str, n: int) -> list[dict]`, `summary() -> dict`.
  - Entry dict: `{"rank": int (1-based), "pos": int, "name": str, "label": [x, y], "tip": [x, y], "err": float, "conf": float}`. Best = ascending `(err, pos)`; worst = descending err, ties by ascending pos. `n` larger than n_val returns all n_val.
  - Summary dict: `{"run": str, "weights": str (file name), "data_dir": str, "n_val": int, "p99": float, "max_err": float, "n_over_3": int, "n_over_7": int, "device": str}`.
- Produces (`val_viewer/app/render.py`): `dial_png(img: (256,256) uint8) -> bytes` (mode L), `heatmap_png(logits: (128,128) float, scale: str) -> bytes` (128 x 128 RGB), `COLORMAP: (256, 3) uint8`.
- Produces (`val_viewer/app/main.py`): `create_app(data: ValData) -> FastAPI` with routes `GET /` (static/index.html), `/static/*`, `GET /api/health` -> `{"status": "ok"}`, `GET /api/summary`, `GET /api/list?side=best|worst&n=N` (422 for a bad side or n outside 1..1000), `GET /api/image/{pos}.png`, `GET /api/heatmap/{pos}.png?scale=log|linear` (default log; 422 bad scale; 404 pos outside 0..n_val-1).
- `ValData.__init__` prints (stdout, flushed) `[rank] i/n_val` progress lines while ranking (at least one per batch) on a cold start, and `[rank] cached <cache file name>` on a warm one.
- Produces (`val_viewer/__main__.py`): `main(argv: list[str] | None = None)` with flags `--data-dir --run --val-split --cache-dir --device {auto,cuda,cpu} --host --port`; exit code 2 + one-line `error: ...` on `FileNotFoundError`.
- Produces (`val_viewer/tests/conftest.py`): fixture `fake_data(tmp_path) -> SimpleNamespace(data_dir: Path, run_dir: Path, cache_dir: Path, val_idx: (30,) int64, labels: (200, 2) float32, names: list[str])`: 200 random 256 x 256 grayscale PNGs (seeded) + `labels.csv` (`filename,tip_x,tip_y`), a seeded randomly initialised TipNet saved with `torch.save(model.state_dict(), run_dir / "tip.pth")`; the images and labels of val positions 0 and 1 are identical (an exact error tie); val position 2 has the label (2.0, 3.0) (near the corner). Also a fixture `client(fake_data)` = `TestClient(create_app(ValData(..., device="cpu")))`. Task 2 reuses `fake_data`.

- [ ] **Step 1: Write the failing tests in `val_viewer/tests/test_api.py`** (all on the `cpu` device):
  1. `test_lists_sorted`: `best n=5` has 5 entries, ranks 1..5, err non-decreasing, `best[0].err == min(all err)`; `worst n=5` err non-increasing, `worst[0].err == max`.
  2. `test_full_list_and_tie_order`: `best n=1000` returns all 30 positions once; positions 0 and 1 (equal err) appear in the order 0 then 1 in both best and worst.
  3. `test_numbers_match_trainer`: for every entry, `tip` equals `tv.decode_logits` of the model on that image (atol 1e-4), `err == hypot(tip - label)` (atol 1e-6), `label`/`name` equal the csv row `val_idx[pos]`, `conf` equals a numpy 5 x 5 clipped softmax mass of `data.logits(pos)` (atol 1e-6).
  4. `test_bad_requests`: `side=foo`, `n=0`, `n=1001` -> 422; `scale=foo` -> 422; image / heatmap for `pos = 30` -> 404.
  5. `test_pngs`: `/api/image/{pos}.png` decodes to a 256 x 256 mode-L array equal to the dataset PNG; `/api/heatmap/{pos}.png` (no scale) equals `?scale=log`, is 128 x 128 RGB, and differs from `?scale=linear`.
  6. `test_heatmap_brightest_at_argmax`: the highest-luminance pixel of the log heatmap PNG is the argmax cell of `data.logits(pos)` (row, col), for 5 positions.
  7. `test_rank_cache`: a second `ValData` with the same arguments has `cache_hit True` and identical `err`; after saving a different seeded model to `tip.pth` -> `cache_hit False` and different `err`; after editing one `labels.csv` value -> `cache_hit False`.
  8. `test_weights_fallback_and_summary`: a run dir with only `best_p99.pth` loads (`summary()["weights"] == "best_p99.pth"`); with both, `"tip.pth"`; summary `n_val == 30`, `n_over_3`, `n_over_7`, `p99`, `max_err` equal numpy on `data.err`, `run` == run folder name.
  9. `test_concurrent_heatmaps`: 40 heatmap requests from 8 threads all return 200 with identical bytes per pos.
  10. `test_cli`: `python -m val_viewer --data-dir <missing>` -> exit 2, stderr one line starting `error:`; a run dir with no weights -> exit 2; a data dir without labels.csv -> exit 2 (subprocess, timeout 120 s). With `capsys`: the first `ValData(...)` on a fresh cache dir prints a line matching `[rank] 30/30`; a second one prints a line starting `[rank] cached`.

- [ ] **Step 2: Run them and see them fail** — `python -m pytest -q val_viewer/tests/test_api.py` -> errors on missing `val_viewer.app`.
- [ ] **Step 3: Implement `data.py`** — ranking pass: batches of `batch` images from the trainer cache memmap (rows `val_idx`), `torch.no_grad`, logits float64 for the softmax / conf; rank cache file `<cache_dir>/rank_<key>.npz`, key = sha256 of (absolute data dir, labels.csv bytes, cache images file size + mtime, weights file bytes, val_split, a module `CACHE_VERSION` int).
- [ ] **Step 4: Implement `render.py` and `main.py`** — model forwards behind one `threading.Lock`; PNG responses with `Cache-Control: max-age=3600`.
- [ ] **Step 5: Implement `__main__.py`** — construct `ValData` before `uvicorn.run`, print `Val viewer on http://host:port/`.
- [ ] **Step 6: Run the tests and see them pass** — `python -m pytest -q val_viewer/tests/test_api.py` -> all pass.
- [ ] **Step 7: Commit** — `git add val_viewer` ; `git commit -m "val_viewer: backend (ranking cache, PNG rendering, API, CLI) + API tests"`.

### Task 2: Frontend — two-column page, markers, detail view, browser test

**Files:**
- Create / replace: `val_viewer/static/index.html`, `val_viewer/static/app.js`, `val_viewer/static/styles.css`
- Test: `val_viewer/tests/test_e2e_browser.py`

**Interfaces:**
- Consumes: Task 1's API routes, entry and summary dicts, and the `fake_data` fixture (`val_viewer/tests/conftest.py`).
- Produces (DOM contract the test relies on):
  - `#n-input` (number), `input[name=scale][value=log|linear]` (radio), `#markers-toggle` (checkbox), `#summary` (header text).
  - Columns `#col-best`, `#col-worst`; entries `.entry[data-side][data-rank][data-pos]`, each with `img.dial`, `img.heatmap` (src `/api/heatmap/{pos}.png?scale=...`), a caption `.caption`, and one `svg.markers` overlay per panel (`viewBox="0 0 256 256"`).
  - In every `svg.markers`: `.true-tip` group with `data-x`, `data-y` = label + 0.5; `.pred-tip` with `data-x`, `data-y` = tip + 0.5; `line.err-line`; circles `.ring-3` (r = 3), `.ring-7` (r = 7).
  - Detail view `#detail` (hidden when closed) with `img.dial` and `img.heatmap` shown at 512 x 512 CSS px, zoom panels `.zoom[data-x0][data-y0]` (top-left image px of the 32 x 32 window, clamped to 0..224), the same marker overlays, `#detail-prev`, `#detail-next`, `#detail-close`, and `data-side` / `data-rank` / `data-pos` on `#detail`.

- [ ] **Step 1: Write the failing tests in `val_viewer/tests/test_e2e_browser.py`** — server: `python -m val_viewer --data-dir <fake> --run <fake run> --cache-dir <tmp> --device cpu --port <free port>` started with `subprocess.Popen` from the repo root, waited on `/api/health` (40 s), killed in teardown (pattern: `tip_webapp/tests/test_e2e_browser.py`); Chromium headless, viewport 1400 x 1000:
  1. `test_page_shows_both_columns`: the header shows `30` val images and the run name; `#col-best` and `#col-worst` each show 24 entries; the first best caption contains, from `/api/list?side=best&n=1`: `#1`, `val <pos>`, the file name, `err <err:.2f> px`, `conf <conf:.3f>`, `true (<x:.1f>, <y:.1f>)` and `pred (<x:.1f>, <y:.1f>)`.
  2. `test_change_n`: fill `#n-input` with 5 + Enter -> 5 entries per column; fill 0 + Enter -> the field reads 1 and 1 entry per column; fill 5000 + Enter -> the field reads 1000 and 30 entries per column.
  3. `test_scale_switch`: click `linear` -> every `img.heatmap` src contains `scale=linear` and the first heatmap's pixels (canvas readback) differ from the log ones; click `log` -> back.
  4. `test_markers_toggle`: untick `#markers-toggle` -> no `svg.markers` visible; tick -> visible again; the lists are not refetched (no new `/api/list` request).
  5. `test_marker_positions`: for the first 3 worst entries, `.true-tip` data-x/y equal label + 0.5 and `.pred-tip` equal tip + 0.5 (from `/api/list`) on both panels; the rendered `.true-tip` bounding-box centre is at `panel_left + (x + 0.5) * panel_width / 256` (tolerance 1 CSS px).
  6. `test_detail_view`: click worst entry #3 -> `#detail` visible with rank 3; dial and heatmap are 512 x 512 (tolerance 1 px); press ArrowRight -> rank 4; ArrowLeft twice -> rank 2; at rank 1 ArrowLeft stays at 1; click `#detail-next` -> rank 2, `#detail-prev` -> rank 1; Escape closes; reopen and `#detail-close` closes.
  7. `test_zoom_clamped_at_border`: open the detail of val position 2 (label (2.0, 3.0)) -> `.zoom` data-x0 = 0, data-y0 = 0, and its `.true-tip` sits at (2.5, 3.5) in the zoom's own image coordinates.
  8. `test_layout_stacks_when_narrow`: at 1400 px wide `#col-worst` is right of `#col-best` with the same top; at 800 px wide `#col-worst` top is below `#col-best` bottom; at 1100 px both columns still fit side by side with no horizontal page scroll.
  9. `test_lazy_images`: at a 1400 x 700 viewport with n = 30, fewer than 60 heatmap requests are made before scrolling; after scrolling both columns to the bottom, all 60 are made.
- [ ] **Step 2: Run them and see them fail** — `python -m pytest -q val_viewer/tests/test_e2e_browser.py` -> fails (placeholder page has no columns).
- [ ] **Step 3: Implement `index.html` / `styles.css`** — header with summary and controls in one row, two columns (CSS grid, one column below 1100 px), entry = caption + two panels side by side, each panel an `img` with an `svg.markers` overlay; heatmap `image-rendering: pixelated`; images `loading="lazy"`.
- [ ] **Step 4: Implement `app.js`** — fetch summary and both lists, render entries, switches apply without refetching lists, detail view with prev / next / Esc; zoom panels show the 32 x 32 window scaled x8 (CSS `object-view-box` or a clipped, scaled `img`) with markers in window coordinates.
- [ ] **Step 5: Run the tests and see them pass**, then the API tests still pass: `python -m pytest -q val_viewer/tests/test_e2e_browser.py` and `python -m pytest -q val_viewer/tests/test_api.py`.
- [ ] **Step 6: Commit** — `git add val_viewer/static val_viewer/tests/test_e2e_browser.py` ; `git commit -m "val_viewer: two-column page, markers, detail view + browser test"`.

### Task 3 (coordinator): real-data acceptance

Not delegated (agents cannot read the data). After Tasks 1 and 2 are collected: `python -m val_viewer` (timeout), check the spec's pass values (n_val 22,500, p99 3.07 px, max 140.1 px, 13 over 7 px, 235 over 3 px, err / conf equal the research bundle's `val_meta.npz` within 1e-6), then use the page like a person on both columns, both scales, the markers switch and the detail view.
