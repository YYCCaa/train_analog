# Saved per-image parameters + val_viewer over train / val / all — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The studio (and the CLI generator) can save every image's sampled parameters in `params.jsonl`; val_viewer ranks every dataset image, switches between val / train / all, and lists the saved parameters of the image in its detail view.

**Architecture:** The generator writes one strict-JSON record per image in the same step as its `labels.csv` row; the studio passes a `save_params` flag through to it. val_viewer ranks all N rows once (cached), addresses images by dataset row, and reads single records from `params.jsonl` through a line-offset index (`val_viewer/app/params.py`).

**Tech Stack:** Python 3.12, numpy, torch, FastAPI + uvicorn, Pillow; React 18 + TypeScript + Vite 5 (studio frontend, built by `studio/run_studio.py` on start); plain HTML/CSS/JS (val_viewer); pytest + Playwright Chromium.

**Spec:** `docs/superpowers/specs/2026-10-27-saved-params-design.md`

## Global Constraints

- Only packages already installed; never `pip install`. `npm ci` / `npm run build` inside your own worktree's `studio/frontend` (as `run_studio.py` does) is allowed; no new npm packages.
- Never open, list or read anything under `../train_dial_data` or `C:/Users/Administrator/PycharmProjects/train_dial_data`. Tests use synthetic data in temp folders only.
- Run every command from the repo root of your worktree.
- Toggle off or on, PNGs and `labels.csv` are byte-identical to today's; `config.yaml` differs only in its `run:` block, which always gets `save_params: true|false`.
- `params.jsonl` line i = record of `labels.csv` data row i: `{"filename", "seed", "angle_deg", "tip_x", "tip_y", "sampled"}`, one strict-JSON object per line (`\n`), `sampled` = `RenderResult.params` as returned.
- val_viewer never loads `params.jsonl` whole and never shows a record whose `filename` differs from the row's `labels.csv` name.
- val_viewer val-set numbers on the real data stay 22,500 / p99 3.07 / max 140.10 / 235 over 3 px / 13 over 7 px (coordinator checks).
- Comments at declarations state what a variable holds, its type and shape (repo style). Tests act like a person where possible (CLI / HTTP / browser).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Non-finite floats in a record** (NaN / ±inf from some parameter) -> `params.jsonl` must stay strict JSON (written as the strings `"NaN"`, `"Infinity"`, `"-Infinity"`), `/api/params` returns 200, the panel shows the string. Tests: Task 1 case 6, Task 3 case 17.
2. **A torn last line after a hard kill** -> that row says "no parameters saved for this image", every other row works. Test: Task 3 case 17.
3. **`params.jsonl` with CRLF endings or a trailing blank line** (copied / edited on Windows) -> records still found, the blank line is not a record. Test: Task 3 case 17.
4. **Fast double switch of the set** (train then all before the first list arrives) -> the page ends on the last chosen set; a late response for the earlier set is dropped. Test: Task 3 case 23.
5. **n larger than the chosen set** (n = 1000 on a 30-image val set) -> the list holds the whole set, ranks 1..len, header `n` correct. Test: Task 3 case 14.

---

### Task 1: generator writes `params.jsonl`

**Files:**
- Modify: `generate_synthetic_v7_mp.py` (`_worker_task`, `_dataset_files`, `check_output_dir` message, `generate_dataset`, `main`)
- Test: `tests/test_v7_gray_generator.py`

**Interfaces:**
- Produces:
  - `PARAMS_FILE = "params.jsonl"` (module constant)
  - `generate_dataset(cfg, output_dir, num_images, seed=0, workers=None, chunksize=4, overwrite=False, progress_cb=None, cancel_event=None, save_params=False) -> {"completed": bool, "done": int, "output_dir": str}`
  - `params_line(record: dict) -> str`: one strict-JSON line (no trailing `\n`) with non-finite floats anywhere in `record` replaced by `"NaN"` / `"Infinity"` / `"-Infinity"`; `json.loads(line)` works with `parse_constant` rejecting constants
  - CLI flag `--save-params`; `run:` block key `save_params` (bool)

- [ ] **Step 1: Write the failing tests** in `tests/test_v7_gray_generator.py` (reuse `run_cli`, `read_labels`, `read_run`, `SEED`):
  1. `test_save_params_cli`: `run_cli("--output-dir", out, "--num-images", 6, "--workers", 2, "--seed", SEED, "--save-params")` returns 0; `params.jsonl` has 6 lines; for each i: `rec["filename"] == labels[i]["filename"]`, `rec["seed"] == g._seed_for_index(SEED, i)`, `rec["angle_deg"] == int(labels[i]["angle_deg"])`, `rec["tip_x"] == float(labels[i]["tip_x"])`, `rec["sampled"] == json.loads(json.dumps(g.render_image(cfg, rec["seed"]).params))`; `read_run(out)["run"]["save_params"] is True`.
  2. `test_no_save_params_identical_outputs`: same 6-image run without and with `--save-params` into `a` / `b`: `a` has no `params.jsonl`; every PNG byte-equal; `labels.csv` byte-equal; config without `run` equal; `run.save_params` False in `a`.
  3. `test_save_params_workers_bit_exact`: 8 images, `--workers 1` vs `--workers 4`, both `--save-params` -> `params.jsonl` bytes equal.
  4. `test_cancel_and_error_keep_params_aligned`: `generate_dataset(cfg, out, 400, seed=SEED, workers=2, save_params=True, ...)` cancelled at 20, and a run whose `progress_cb` raises at 5: in both, the filenames in `params.jsonl` lines == `labels.csv` rows (same order) == the set of PNGs on disk.
  5. `test_overwrite_and_not_empty_with_params`: a `--save-params` run, then an `--overwrite` run without it -> no `params.jsonl`; a folder holding only `params.jsonl` -> exit 1 without `--overwrite`, stderr mentions `params.jsonl`.
  6. `test_params_non_finite_strict_json`: monkeypatch `g.render_image` (in-process `generate_dataset(..., workers=1, save_params=True)`) to return params with `float("nan")`, `float("inf")`, `-float("inf")` nested in a dict and a list -> every line parses with `json.loads(line, parse_constant=<raise>)`; the values read `"NaN"`, `"Infinity"`, `"-Infinity"`.

- [ ] **Step 2: Run them, see each fail** — `python -m pytest -q tests/test_v7_gray_generator.py` (FAIL: no `--save-params` / no file).

- [ ] **Step 3: Implement.** `_worker_task` returns `(label_row, record_or_None)`; the record is built in the worker (image seed, label values, `res.params`) only when the pool was initialised with `save_params=True` (pass it through `_worker_init`). `accept()` writes the label row and, if saving, `params_line(record) + "\n"` to `params.jsonl` (opened `"w"`, `encoding="utf-8"`, `newline="\n"`) and flushes both files. `_dataset_files` lists `params.jsonl`; the not-empty message names it. `run` gets `"save_params": bool(save_params)`. `main` adds `--save-params` (help: "also write params.jsonl: every image's sampled parameters (~3 KB per image)").

- [ ] **Step 4: Run** `python -m pytest -q tests/test_v7_gray_generator.py` — all pass (new and old). Also `python -m pytest -q tests/test_v7_render.py -k "serialis or params"` stays green.

- [ ] **Step 5: Commit** `git commit -m "v7 generator: --save-params writes params.jsonl aligned with labels.csv"`

### Task 2: studio "save parameters" toggle (same agent as Task 1, after it)

**Files:**
- Modify: `studio/backend/app.py` (`GenerateBody`, `/api/generate` handler, not-empty message), `studio/backend/jobs.py` (`JobRunner.start`, `_run`), `studio/frontend/src/api.ts` (`GenerateRequest`, `startGenerate`), `studio/frontend/src/components/GeneratePanel.tsx`
- Test: `tests/test_studio_api.py`, `tests/test_studio_e2e.py`

**Interfaces:**
- Consumes: Task 1 `generate_dataset(..., save_params=...)`, `PARAMS_FILE`.
- Produces: request field `save_params: StrictBool = False` (pydantic `StrictBool`, so `"yes"` / `1` are 422); `JobRunner.start(cfg, output_dir, num_images, seed, workers, overwrite, save_params=False)` (keyword-optional, so the existing 6-argument test fake keeps working); checkbox `data-testid="gen-save-params"`, label text "save parameters", unticked by default, title "also write params.jsonl: every image's sampled parameters (~3 KB per image)".

- [ ] **Step 1: Write the failing tests.**
  - `tests/test_studio_api.py`:
    7. `test_generate_save_params`: `_gen_body(...)` + `save_params: True`, 8 images, 2 workers -> done; `params.jsonl` 8 lines, filenames equal the `labels.csv` rows; `run.save_params` True.
    8. `test_generate_default_writes_no_params`: unchanged `_gen_body` -> no `params.jsonl`; `run.save_params` False.
    9. `test_generate_save_params_must_be_bool`: `save_params` `"yes"` and `1` -> 422 with an `errors` list; no folder created.
    10. `test_not_empty_counts_params_file`: a folder holding only `params.jsonl` -> 409 whose message contains "overwrite box" and names `params.jsonl`.
  - `tests/test_studio_e2e.py`:
    11. `test_generate_with_save_params`: a person sees the "save parameters" box unticked, generates 20 images into folder A (no `params.jsonl`), ticks the box, generates 20 into folder B -> B's `params.jsonl` has 20 lines; the record of `dial_0003.png` has `sampled == json.loads(json.dumps(render_image(<UI config>, _seed_for_index(<seed>, 3)).params))`.

- [ ] **Step 2: Run, see each fail** — `python -m pytest -q tests/test_studio_api.py` and `python -m pytest -q tests/test_studio_e2e.py -k save_params`.

- [ ] **Step 3: Implement** the field, pass-through, message and checkbox (state `saveParams` with `useState(false)`, sent as `save_params`), beside the overwrite checkbox in `gen-grid`.

- [ ] **Step 4: Run** `python -m pytest -q tests/test_studio_api.py` and `python -m pytest -q tests/test_studio_e2e.py` — all pass.

- [ ] **Step 5: Commit** `git commit -m "studio: 'save parameters' toggle passes save_params to the generator"`

### Task 3: val_viewer over train / val / all, with saved parameters

**Files:**
- Create: `val_viewer/app/params.py`
- Modify: `val_viewer/app/data.py`, `val_viewer/app/main.py`, `val_viewer/static/index.html`, `val_viewer/static/app.js`, `val_viewer/static/styles.css`, `val_viewer/README.md`
- Test: `val_viewer/tests/conftest.py`, `val_viewer/tests/test_api.py`, `val_viewer/tests/test_e2e_browser.py`

**Interfaces:**
- Consumes: the `params.jsonl` format (Global Constraints); `train_v9_tip` as today.
- Produces:
  - `params.py`: `class ParamStore(data_dir, names: list[str], cache_dir: Path)` with `.available: bool` and `.get(row: int) -> dict` raising `NoParamsFile`, `NoRecord`, `RecordMismatch(record_name: str)`; never reads the file whole; a line that is not complete JSON, a blank line, or a missing line -> `NoRecord`; tolerates `\r\n`.
  - `ValData`: ranks all N rows (`CACHE_VERSION = 2`); attributes `n_total`, `n_train`, `n_val`, `names` (N,), `label` / `tip` (N, 2) float64, `err` / `conf` (N,) float64, `split` (N,) str `"train"|"val"`, `split_pos` (N,) int64; `summary(set_name="val") -> dict`, `entries(set_name, side, n) -> list[dict]` (keys `rank, row, split, split_pos, name, label, tip, err, conf`), `image(row)`, `logits(row)`, `params(row) -> dict`.
  - HTTP API and screen exactly as in the spec's Part C; `set` query values `val|train|all` (422 otherwise).
  - conftest helper `write_params_jsonl(fd, rows=None, mutate=None) -> Path`: writes a fake `params.jsonl` (records with the right filenames and a small `sampled` dict holding e.g. `{"pointer": {"gray": 100 + row}, "glares": {"items": [...]}}`), optionally only some rows or with a mutation hook for the bad-file cases.

- [ ] **Step 1: Write the failing tests** (adapt existing tests from val positions to dataset rows without weakening them).
  - `val_viewer/tests/test_api.py`:
    12. `test_summary_per_set`: default set is val (n 30 = today's numbers); `set=train` n 170; `set=all` n 200; `n_over_3(all) == n_over_3(val) + n_over_3(train)` (same for 7 px); `params` `"none"` without a file, `"saved"` with one.
    13. `test_bad_set_422`: `set=test` -> 422 on summary and list.
    14. `test_list_per_set`: train entries all `split == "train"` with `train_idx[split_pos] == row`; `all` holds both splits; best ascending / worst descending err, ties by row ascending; `n=1000` on val returns 30 entries ranked 1..30.
    15. `test_rows_address_images`: `/api/image/{row}.png` of a train row equals the PNG file `names[row]`; heatmap of a train row is 128x128 RGB; row 200 -> 404 on image, heatmap and `/api/params`.
    16. `test_rank_cache_all_rows`: first open prints `[rank] 200/200` and ranks every row; the second open is a cache hit; a v1-style cache file is ignored.
    17. `test_params_outcomes`: no file -> 404 "no parameters saved for this dataset"; full file -> 200 with `sampled` equal to the fixture; file without row r -> 404 "no parameters saved for this image" for r only; a record with the wrong filename -> 409 naming it; last line cut mid-JSON -> 404 for that row, 200 for the others; CRLF endings + trailing blank line -> all records found; a `"NaN"` string value is returned as `"NaN"`.
  - `val_viewer/tests/test_e2e_browser.py` (the module server's fake dataset gets a `params.jsonl`):
    18. `test_set_switch`: a person clicks set train -> header shows `train 170`, every caption has a `train` tag; all -> `all 200 (train 170 · val 30)` and both tags; val -> the original first entries again.
    19. `test_detail_shows_params`: opens worst #1 -> the Parameters column shows section `pointer` with `gray` equal to the fixture value of that row; presses → -> the value of the next entry's row.
    20. `test_detail_no_params_message`: with `/api/params/*` answered 404 "no parameters saved for this dataset" (Playwright route) the column shows "No parameters saved for this dataset" and the studio hint.
    21. `test_params_layout_narrow`: at 800 px with the detail open the column sits below the panels and `scrollWidth <= 800`.
    22. `test_captions_tag_split`: in `all`, a train entry's caption reads `train <split_pos>` and a val entry `val <split_pos>`.
    23. `test_fast_set_switch_keeps_last`: delay the first `/api/list?set=train` responses (route), click train then all -> after both settle every entry's tag set is the one of `all` and the header says `all 200`.

- [ ] **Step 2: Run, see each fail** — `python -m pytest -q val_viewer/tests/test_api.py`, `python -m pytest -q val_viewer/tests/test_e2e_browser.py`.

- [ ] **Step 3: Implement** `params.py`, the all-rows ranking and per-set orders in `data.py`, the routes in `main.py`, the set switch + tags + Parameters column in the page, and README updates (set switch, params column, how to get parameters: tick "save parameters" in the studio).

- [ ] **Step 4: Run** both files — all pass, under their 300 s / 600 s timeouts.

- [ ] **Step 5: Commit** `git commit -m "val_viewer: val/train/all switch over every row; saved parameters in the detail view"`

### Task 4 (coordinator, after Tasks 1-3 are collected)

- [ ] Add `test_generated_params_reach_viewer` to `val_viewer/tests/test_api.py`: `generate_dataset(cfg, tmp, 40, seed=5, workers=1, save_params=True)`, a seeded random TipNet `tip.pth`, open `ValData` -> for rows 0, 17, 39 `/api/params/{row}` `sampled == json.loads(json.dumps(render_image(cfg, _seed_for_index(5, row)).params))`; the detail view of one entry shows its `pointer.gray`.
- [ ] `WT verify` over the touched areas; real-data acceptance per the spec (val numbers; train 127,500 / all 150,000; "No parameters saved" on `sv9_150k`; a 300-image studio run with the toggle opened in val_viewer); `WT land`.
