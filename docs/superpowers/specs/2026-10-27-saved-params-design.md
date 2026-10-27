# Saved per-image parameters, and val_viewer over every image

## Intent

Every synthetic v7 dial is drawn from about 100 random parameters (pointer gray, shadow strength and distance, glare
count, jitter, blur, …). When TipNet gets an image badly wrong, the first question is *what was special about that
image*. Today the answer is lost: a dataset keeps only `labels.csv` (file name, angle, tip) and the shared
`config.yaml` ranges.

This change keeps the answer and shows it:
1. the studio's Generate panel gets a toggle that also saves the parameters each image was drawn with;
2. `val_viewer/` stops being limited to the validation split: the user picks val, train or all, sees the best and
   worst n of that set, and the detail view lists the saved parameters of the image on screen.

It stays a local research tool for one person on the training machine.

## Decisions made by the user

- **Toggle in the studio's Generate panel** to save the parameters of each image.
- **val_viewer looks at all training + val images** through a **val / train / all switch**: best and worst n are
  ranked within the chosen set; every entry is tagged train or val.
- **The parameter list comes only from saved parameter files.** No backfill, no re-rendering in the viewer. A dataset
  generated without the toggle (today's `sv9_150k`) shows "no parameters saved" until it is generated again with the
  toggle on.

## Goals

1. Studio: a "save parameters" checkbox in the Generate panel; with it ticked, the generated folder also holds one
   parameter record per image, aligned with `labels.csv`.
2. The command-line generator can do the same (`--save-params`), so studio and CLI datasets are identical.
3. val_viewer: a val / train / all switch in the toolbar; the header numbers, the best list and the worst list
   follow it; each entry says whether it is a train or a val image.
4. val_viewer detail view: the full parameter list of the image shown, grouped by section, readable without
   scrolling sideways; a clear message when the dataset (or that image) has no saved parameters.
5. Nothing that works today changes: with the toggle off the generator's output is byte-identical to today's;
   with the toggle on the PNGs, `labels.csv` and every config field are still byte-identical (only the extra file and
   one `run:` entry are added); val_viewer's val numbers on `sv9_150k` stay 22,500 / p99 3.07 / max 140.10 /
   235 over 3 px / 13 over 7 px.

## Non-goals

- No backfill of `sv9_150k` (user decision) and no re-rendering of parameters in the viewer.
- No ranking, filtering or searching by parameter; no parameter statistics or plots.
- No browse-in-order mode or jump-to-file (the user chose the switch only).
- No change to training, to the trainer's cache, or to `tip_webapp/`.

## Anti-goals (failure even if it "works")

- **Parameters shown for the wrong image.** Every record carries its file name; the viewer refuses a record whose file
  name differs from the image's `labels.csv` name instead of showing it.
- The toggle changing any image or label (the parameters must be the ones that drew the image on disk).
- A cancelled or failed run leaving parameter records for images that were deleted, or images without records.
- val_viewer loading the whole parameter file into memory (about 3 KB per image, ~480 MB for 150,000 images), or a
  first start that ranks 150,000 images without progress output.

## Part A: generator (`generate_synthetic_v7_mp.py`)

| what | decided |
|---|---|
| API | `generate_dataset(..., save_params=False)` (keyword, default off) |
| CLI | `--save-params` flag, default off; help text says ~3 KB per image |
| file | `params.jsonl` in the output folder, next to `labels.csv` |
| line i | the record of `labels.csv` data row i (same order: `pool.imap` is ordered), one JSON object per line, UTF-8, `\n` endings |
| record | `{"filename": "dial_0003.png", "seed": <int, the seed render_image got = _seed_for_index(seed, idx)>, "angle_deg": <int>, "tip_x": <float>, "tip_y": <float>, "sampled": <RenderResult.params>}` — the same names as the studio's Tile Save `params.json`, without the dataset-wide fields (config, img_size, generator), which live in `config.yaml` once |
| values | `sampled` is exactly the `params` dict `render_image` returned for that image (JSON-encoded as is); `tip_x` / `tip_y` are the same rounded values as in `labels.csv` |
| writing | the record is written in the same step that writes the image's `labels.csv` row, so a cancel, a worker error or Ctrl-C leaves exactly one record per kept image and row |
| owned files | `params.jsonl` joins dial_*.png, labels.csv, config.yaml as a file the generator owns: it counts for the "folder not empty" check and `--overwrite` deletes it (also when the new run does not save parameters, so no stale file survives) |
| config.yaml | the `run:` block gets `save_params: true|false` (`v7_config` ignores the run block on load, so old readers are unaffected) |
| determinism | `params.jsonl` is byte-identical for 1 and N workers, like the PNGs |

## Part B: studio (`studio/`)

- **Generate panel** (`studio/frontend/src/components/GeneratePanel.tsx`): a checkbox labelled **"save parameters"**
  next to "overwrite", same `label.bool` pattern, `data-testid="gen-save-params"`, unticked by default, with a
  one-line hint (title / help text) "also write params.jsonl: every image's sampled parameters (~3 KB per image)".
- **Request**: `GenerateRequest` / `GenerateBody` gain `save_params: bool = False`; the backend passes it through
  `JobRunner.start` to `generate_dataset`. Existing callers that omit it get today's behaviour. A non-boolean value
  is a 422 with the existing `errors` contract. The new `JobRunner.start` parameter is keyword-optional so the test
  fake with the old signature keeps working.
- The "folder not empty" message lists `params.jsonl` among the files it means.
- Tile Save and the preview are unchanged.

## Part C: val_viewer

### Data

- All N dataset rows are ranked once (one batched forward pass; batches ≤ 256; `[rank] i/N` progress lines) and
  cached on disk like today (bump `CACHE_VERSION`; the key rules stay). On `sv9_150k` the first start takes about
  2-3 minutes on the GPU; a warm start stays about 1-2 s.
- The split stays the trainer's (`split_indices(N, val_split)`, seed 42): `train_idx`, `val_idx`. Every row is in
  exactly one of them.
- **Image id = dataset row** (0..N-1, the `labels.csv` data-row index) in every URL and in the page. Captions show the
  split and the position inside it, so `val 6995` keeps meaning what the research tables mean, and `train p` means
  `train_idx[p]`.
- **Sets**: `val` = val rows, `train` = train rows, `all` = every row. Ranking inside a set: best = ascending error,
  worst = descending error, ties by row ascending.
- **Parameters**: `<data-dir>/params.jsonl` if it exists. The viewer reads one record per request by seeking to it
  (an index of line offsets built on first use, cached so later starts do not rescan); it never loads the file whole.
  A record is shown only if its `filename` equals the row's `labels.csv` name.

### HTTP API (changes)

| route | returns |
|---|---|
| `GET /api/summary?set=val\|train\|all` (default val) | as today (run, weights, data dir, device, key) plus `set`, `n` (images in the set), p99 / max error and counts over 3 / 7 px **of that set**, `n_train`, `n_val`, `n_total`, and `params`: `"saved"` or `"none"` |
| `GET /api/list?set=…&side=best\|worst&n=N` | entries: rank, `row`, `split` ("train"/"val"), `split_pos`, name, label, tip, err, conf. 422 for a bad set / side / n outside 1..1000 |
| `GET /api/image/{row}.png`, `GET /api/heatmap/{row}.png?scale=log\|linear` | as today but addressed by dataset row; 404 for a row outside 0..N-1; still carry the `v=<key>` cache token |
| `GET /api/params/{row}` | 200 `{"row", "filename", "seed", "angle_deg", "tip_x", "tip_y", "sampled"}`; 404 `{"detail": "no parameters saved for this dataset"}` when there is no `params.jsonl`; 404 `"no parameters saved for this image"` when the file has no record for that row; 409 `"saved parameters do not match this image (<file name in record>)"` on a file-name mismatch; 404 for a row outside the dataset |

### Screen

```
┌ Val viewer · TipNet v9 (v9_tipnet_a, tip.pth) · val 22,500 · p99 3.07 px · >3 px 235 · >7 px 13 ──────────────────┐
│ set (•) val ( ) train ( ) all     n [ 24 ]     heatmap (•) log ( ) linear     [✓] markers     ◎ true tip  × predicted │
├─────────────────────────── BEST 24 ──────────────────────┬──────────────────────── WORST 24 ─────────────────────────┤
│ #1 [val] val 424 · dial_131592.png                        │ #1 [val] val 6995 · dial_77439.png                        │
│ …                                                         │ …                                                         │
```

- The **set** switch applies at once (lists reload, header numbers follow, n is kept). The header names the set:
  `val 22,500`, `train 127,500` or `all 150,000 (train 127,500 · val 22,500)`.
- Each caption starts with a small **train** / **val** tag (text, not colour alone).

Detail view (wide window): the two 512 px panels and the zooms as today, plus a **Parameters** column on the right
that scrolls on its own:

```
┌ ← prev  next →  WORST #1 [val] val 6995 · dial_77439.png · err 140.10 px · conf 0.788 · …           Close × ┐
│ ┌──────────┐ ┌──────────┐   Parameters · seed <image seed> · angle <deg>° · tip (71.0, 180.9)              │
│ │  dial    │ │ heatmap  │   ▾ angle        angle_deg <deg>                                                 │
│ │  512     │ │  512     │   ▾ geometry     scale 1.0671 · dial_radius 90 · …  (one row per value)         │
│ └──────────┘ └──────────┘   ▾ pointer      gray 182 · hole_gray 59 · forbidden [[74, 144]] · …            │
│ ┌────┐ ┌────┐               ▾ shadows      active false · …                                              │
│ │zoom│ │zoom│               ▾ tri_artifacts  n 1 · items  #1 gray 241 · center_x 66 · …                   │
│ └────┘ └────┘               …  (all 17 sections, expanded; each can be collapsed)                         │
```

- One row per value; nested dictionaries are shown as dotted names (`jitter.capped.brightness`); lists of items
  (`tri_artifacts.items`, reflection / glare items) as numbered sub-rows; short lists of numbers inline. Values are
  shown as stored (no rounding beyond what the file holds).
- Without saved parameters the column shows one line: "No parameters saved for this dataset — generate it with
  'save parameters' ticked in the studio" (or "… for this image" / the mismatch message from the API).
- ← / → and the buttons step through the current list as today; the parameter column follows the entry.
- Narrow window: the parameter column moves below the panels; no horizontal page scroll at 800 px.

## Tests (decided)

Tests never read the real dataset; datasets are synthetic and written to temp folders; ports are free ports except the
studio E2E's existing fixed 8799.
- **Generator** (`tests/test_v7_gray_generator.py`, CLI like a person): `--save-params` writes aligned records equal to
  `render_image`'s params; without it no file and byte-identical PNGs / labels; 1 vs 4 workers byte-identical;
  cancel / error keep records = rows = PNGs; `--overwrite` removes a stale `params.jsonl`; a folder holding only
  `params.jsonl` counts as not empty.
- **Studio API** (`tests/test_studio_api.py`): `save_params: true` writes the file, omitted writes none, non-boolean
  is a 422 with `errors`.
- **Studio browser** (`tests/test_studio_e2e.py`): a person ticks "save parameters", generates 20 images and gets a
  20-line `params.jsonl` whose records match the images; unticked (default) writes none.
- **val_viewer API** (`val_viewer/tests/test_api.py`): per-set summary and lists (val unchanged, train and all counts
  add up, tags right, order and ties), rows addressed in image / heatmap routes, every `/api/params` outcome above,
  bad set → 422, the rank cache built once over all rows.
- **val_viewer browser** (`val_viewer/tests/test_e2e_browser.py`): a person switches val → train → all → val and sees
  the header and tags follow; opens the detail view and reads a known parameter value; steps with → and sees the
  next image's parameters; a dataset without `params.jsonl` shows the message; the 800 px layout has no sideways
  scroll.
- **Generator → viewer together** (added by the coordinator after both parts are collected): a small dataset generated
  with `save_params=True`, opened in val_viewer: `/api/params/{row}.sampled` equals `render_image` for that row and
  the detail view shows it.

## Acceptance on the real data (coordinator, after landing the parts together)

- `python -m val_viewer` on `sv9_150k`: the val set's numbers are unchanged (above); train and all show 127,500 and
  150,000 images; detail view says "No parameters saved for this dataset".
- In the studio, generate a small dataset (e.g. 300 images, seed 0, the `sv9_150k` config) with "save parameters"
  ticked; open it in val_viewer with the `v9_tipnet_a` weights; the detail view lists each image's parameters, and
  for a few images they equal `render_image(config, seed)` re-run by hand.

## Left to the builder

Module layout, the exact markup and styling of the parameter column (within the rules above), how the line-offset
index is stored and keyed, the batch size of the ranking pass, how the set switch is kept in page state.
