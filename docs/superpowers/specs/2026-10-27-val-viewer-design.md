# Validation viewer: best and worst n TipNet v9 predictions with heatmaps

## Intent

TipNet v9 finds the pointer tip of an analog dial (p99 error 3.07 px on the 22,500-image validation split). The
statistics are known; what the failures *look like* is not easy to see. This app lets the user scroll through the
best and the worst n validation images, each shown next to the model's heatmap, with the true tip and the
predicted tip marked on both, so a wrong peak, a smeared peak or a decoy is visible at a glance.

It is a local research tool for one person on the training machine, not a product.

## Decisions made by the user

- **Ranking:** by true tip error only (`err` = distance between predicted and labelled tip, image px).
  Best = smallest error first; worst = largest error first. No confidence-based ranking control.
- **Layout:** two columns side by side, BEST n on the left and WORST n on the right; they stack into one column
  (best above worst) on narrow screens.
- **Heatmap next to the image, with visual cues on the true tip position.**

## Goals

1. Show the best n and the worst n validation images, n chosen by the user.
2. Each entry shows the dial image and its heatmap side by side, at the same scale, with markers on both:
   true tip (crosshair + rings at 3 px and 7 px radius), predicted tip, and a line between them.
3. Each entry is captioned with its numbers: rank, val position, file name, error, confidence, true and
   predicted tip.
4. A detail view for one entry: both panels large, plus a zoom around the true tip; step to the previous / next
   entry of the same list.
5. The numbers match the research exactly (same split, same weights, same decode): on the real data the worst
   error is 140.1 px, 13 errors exceed 7 px, 235 exceed 3 px.

## Non-goals

- No confidence ranking, filtering, search, or editing of labels.
- No upload of new images (that is `tip_webapp/`).
- No other models or splits in the UI (the CLI may point at another run directory with the same architecture).
- No authentication, no multi-user use, no deployment beyond 127.0.0.1 by default.

## Anti-goals (failure even if it "works")

- Markers that are off by half a pixel or a heatmap that is not aligned with the image: the whole point is to
  judge a few-pixel error by eye.
- Numbers that disagree with the research tables (wrong split, TF32 on, a different decode).
- A first start that silently hangs: ranking 22,500 images must show progress in the terminal.
- Touching existing files of the repo (the app is a new folder; `train_v9_tip.py` is imported, not edited).

## Data and model (decided)

All of this reuses `train_v9_tip.py` (imported as a module, run from the repo root):

| what | how |
|---|---|
| labels | `read_labels_csv(data_dir)` -> names (N,) str, labels (N, 2) float32 image px (x, y) |
| images | the trainer's cache `cache_paths(data_dir + "_cache")[0]`: (N, 256, 256) uint8 `.npy`, opened as a memmap. If `cache_is_valid` is false, build it with `build_cache` (as the trainer does) |
| val split | `split_indices(N, val_split)` with `val_split = 0.15`, seed 42 -> `val_idx` (22,500,) int64. **"val position" p** means `val_idx[p]`; the research uses the same numbering |
| model | `build_model("tipnet")`, weights `<run>/tip.pth`, or `<run>/best_p99.pth` when there is no `tip.pth` (the trainer writes only `best_p99.pth`; for `v9_tipnet_a` both files hold identical weights, checked tensor by tensor). `<run>` is a folder path; the header shows the folder name and the weights file used. Eval mode, `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`; input `to_input(uint8 images)`; output logits (B, 1, 128, 128) |
| predicted tip | `decode_logits(logits)` -> (B, 2) image px |
| error | Euclidean distance between predicted tip and label, image px |
| confidence (shown, not ranked) | the shipped `conf`: sum of the global softmax probabilities in the 5 x 5 cells around the argmax cell, window clipped to the map (same as `tip_webapp/app/inference.py::window_mass`) |

Defaults: data dir `train_v9_tip.DEFAULT_DATA_DIR` (`.../train_dial_data/sv9_150k`), run dir
`train_v9_tip.DEFAULT_RUNS_DIR + "/v9_tipnet_a"`. Device: CUDA when available, else CPU.

### Coordinates (must be exact)

- Image coordinates: pixel i is centred at coordinate i (the trainer's convention). On screen, pixel i spans
  [i, i+1) in an SVG `viewBox="0 0 256 256"`, so a point (x, y) is drawn at (x + 0.5, y + 0.5).
- Heatmap cell j covers image pixels 2j and 2j+1, so the 128 x 128 heatmap scaled x2 (nearest neighbour, cells
  visible) occupies exactly the same 256 x 256 box. The same marker coordinates are therefore used on both panels.

### Ranking cache

The first start runs one batched forward pass over the val split (about 10-60 s on the GPU, minutes on CPU) with a
progress line in the terminal, and stores per val position: row, label (x, y), predicted tip (x, y), err, conf.
The result is cached on disk and reused on the next start when the data (labels.csv, cache file) and the weights
file are unchanged; any change recomputes it. Cache location: `--cache-dir`, default `val_viewer/.cache/`
(gitignored).

Ties in error are broken by val position (ascending) so the order is deterministic.

### Heatmaps

Computed per image on request (one forward pass), rendered as a 128 x 128 PNG:
- **log** (default): log10 of the softmax probability, shown over the 6 decades below the map maximum (values
  below are clipped to the bottom colour). The peaks are so sharp that a linear map shows one dot.
- **linear**: probability / map maximum.
- Colour map: one-hue sequential, dark (low) to light (high), chosen so that neither marker colour appears in it.

## Screens

```
┌ Val viewer · TipNet v9 (v9_tipnet_a) · val 22,500 · p99 3.07 px · >3 px 235 · >7 px 13 ──────────────┐
│ n [ 24 ]       heatmap (•) log ( ) linear       [✓] markers                                          │
├──────────────────────── BEST 24 ───────────────┬──────────────────────── WORST 24 ───────────────────┤
│ #1  val 10432 · v9_0123456.png                 │ #1  val 8811 · v9_0098765.png                         │
│     err 0.02 px · conf 0.981                   │     err 140.1 px · conf 0.31                          │
│     true (101.0, 57.2) · pred (101.0, 57.2)    │     true (40.3, 200.9) · pred (150.2, 99.0)           │
│ ┌────────────┐ ┌────────────┐                  │ ┌────────────┐ ┌────────────┐                         │
│ │   dial     │ │  heatmap   │                  │ │  dial   ×  │ │ ▪       ×  │                         │
│ │     ◎      │ │     ◎      │                  │ │  ◎╱        │ │ ◎╱         │                         │
│ └────────────┘ └────────────┘                  │ └────────────┘ └────────────┘                         │
│ #2  …                                          │ #2  …                                                 │
└────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
  ◎ true tip: green crosshair (open centre, so the pixel stays visible) + circles r = 3 px and r = 7 px
  × predicted tip: orange;  ─ thin line from predicted to true tip.  All markers have a dark outline so they
  stay visible on the grey dial and on the heatmap.
```

- **n:** number input, default 24, range 1..1000; applied on Enter or when the field loses focus. Out-of-range
  input is clamped and the field shows the value actually used. If n is larger than half the split the two lists
  overlap; that is allowed.
- **Heatmap scale** and **markers** switches apply to every entry immediately, without reloading the lists.
- Images load lazily (only entries near the viewport fetch their PNGs), so n = 1000 stays usable.
- **Detail view** (click an entry): dial and heatmap at 512 x 512, plus a x8 zoom of the 32 x 32 image-pixel
  window around the true tip for both panels (window clamped to the image), all with the same markers and the
  entry's numbers. Buttons and the ← / → keys step to the previous / next entry of the same list; Esc or the
  close button closes it.

## Architecture (decided)

Same style as `tip_webapp/`: FastAPI + uvicorn backend, plain HTML / CSS / JS frontend with no build step and no
CDN. New package `val_viewer/` at the repo root; run with `python -m val_viewer` from the repo root.

CLI: `python -m val_viewer [--data-dir D] [--run R] [--val-split 0.15] [--cache-dir C] [--device auto|cuda|cpu]
[--host 127.0.0.1] [--port 8020]`. A missing data dir, a missing labels.csv, or a run folder with neither `tip.pth`
nor `best_p99.pth` exits with code 2 and a one-line message before the server starts.

HTTP API (JSON unless noted):

| route | returns |
|---|---|
| `GET /` | the page |
| `GET /api/summary` | run name, data dir, n_val, p99 / max error, counts over 3 px and 7 px, device |
| `GET /api/list?side=best\|worst&n=N` | the first n entries of that ranking: rank, pos, name, label [x, y], tip [x, y], err, conf. 422 for a bad side or n outside 1..1000 |
| `GET /api/image/{pos}.png` | the 256 x 256 grayscale dial (PNG). 404 for pos outside the split |
| `GET /api/heatmap/{pos}.png?scale=log\|linear` | the 128 x 128 heatmap (PNG). 404 / 422 as above |

Dependencies are those already in the venv (torch, numpy, fastapi, uvicorn, pillow); nothing is installed.

## Tests (decided)

Tests never read the real dataset. A fixture writes a small fake dataset in a temp folder (about 200 random
256 x 256 PNGs + `labels.csv` in the trainer's format, so the val split has about 30 images) and a randomly
initialised TipNet `tip.pth`.
- **API tests** (FastAPI TestClient): list order and counts, errors equal the distance between `decode_logits`
  output and the label, PNG sizes and modes, log vs linear differ, 404 / 422 cases, cache reuse and invalidation,
  CLI exit code 2 on missing inputs.
- **Browser test** (Playwright Chromium, server started with `python -m val_viewer` on a free port): a person
  opens the page, sees both columns, changes n, toggles scale and markers, opens the detail view, steps with the
  arrow keys, closes it with Esc, narrows the window and sees the columns stack; the true-tip marker of an entry
  sits at its label + 0.5 in the panel's coordinates.

## Acceptance on the real data (coordinator, after the build)

Build agents cannot read `../train_dial_data`, so Goal 5 is checked by the coordinating session after collecting,
by running `python -m val_viewer` on the real data and using the page like a person:
- The trainer's cache exists and is valid (`sv9_150k_cache_images.npy`, 9.8 GB, built 2026-10-25), so no cache
  build is expected. The GPU is not shared at the moment; the ranking pass uses batches of at most 256.
- Pass values: n_val 22,500; p99 3.07 px; worst error 140.1 px; 13 errors over 7 px; 235 over 3 px; per-image
  err and conf equal the research bundle's `val_meta.npz` (computed with TF32 off) within 1e-6.
- Do **not** compare with the run's `worst_val.csv`: the trainer wrote it with bf16 autocast and TF32 on, so its
  numbers differ slightly by design.

## Left to the builder

Module layout inside `val_viewer/`, the exact colour values (within the rules above), panel sizes in the list
(both columns must fit side by side at a 1100 px wide window), batch size of the ranking pass, cache file format and
key, how lazy loading is done.
