# Val viewer

Look at the **best and worst n images of a TipNet v9 run** (val, train or all rows), each as the 256 x 256 dial
next to its 128 x 128 softmax heatmap, with the true tip (green crosshair) and the predicted tip
(orange x) drawn on both. Click an entry for a 512 px detail view with two x8 zooms of the 32 x 32 px
window around the true tip and, when the dataset has a `params.jsonl`, the parameters the image was drawn with.

- Backend: FastAPI + PyTorch (`app/`), runs the real `train_v9_tip.py` model and decode on the server
- Frontend: plain HTML/CSS/JS, no build step (`static/`)
- Real data (v9 val split, run `v9_tipnet_a`): 22,500 images, p99 3.07 px, 13 over 7 px, 235 over 3 px

## Run

```bash
python -m val_viewer                                  # http://127.0.0.1:8020/
python -m val_viewer --data-dir D --run R             # another dataset / run folder (tip.pth or best_p99.pth)
python -m val_viewer --val-split 0.15 --device cpu    # split fraction, cuda | cpu | auto
python -m val_viewer --cache-dir C --host 0.0.0.0 --port 9000
```

Run it from the repo root (the folder that contains `val_viewer/` and `train_v9_tip.py`). A missing data
dir, `labels.csv`, run dir or weights file exits with code 2 and one `error:` line.

## First start and the rank cache

The first start runs a ranking pass over every dataset row (train and val) and prints `[rank] done/total` lines
(measured: about 30 s on the GPU for 150,000 images). The result (predicted tip, error, confidence, rows) is
cached in `val_viewer/.cache/rank_<key>.npz`; a warm start loads it in about 1 s (`[rank] cached ...`).
The key covers data dir, val split, `labels.csv`, the decoded-image cache, the weights, the device and
`train_v9_tip.py`, so a change of any of them re-ranks by itself. A corrupt or mismatching file is
recomputed. To clear it, delete `val_viewer/.cache/` (or whatever `--cache-dir` points to).
The decoded PNGs also live in the trainer's `<data-dir>_cache_*` files, built once if missing. The trainer's own
validity check (`train_v9_tip.cache_is_valid`, which the viewer uses as is) also treats the cache as stale when
`labels.csv` is newer than `<data-dir>_cache_images.npy`, so a dataset regenerated in place with the same seed and only
a different look is rebuilt
(it prints `[cache] labels.csv is newer than ...: rebuilding`). The rebuilt file has a new mtime, so the rank cache
key changes and the rows are ranked again. A plain restart leaves both caches alone.

## Controls

| Control | Effect |
|---|---|
| set val / train / all | which images are ranked: the val split, the train split or every row; header numbers and both lists follow at once (n is kept) |
| n field (1..1000, Enter) | how many best and worst entries per column (a set smaller than n shows the whole set) |
| heatmap log / linear | colour scale of every heatmap (log = 6 decades below the maximum) |
| markers checkbox | show or hide the true / predicted tip markers |
| click an entry, or Tab to it + Enter / Space | open the detail view |
| Left / Right arrow, prev / next | step through the list in the detail view |
| Esc, Close | close the detail view (focus returns to the entry) |

Images are addressed by **dataset row** (0..N-1, the `labels.csv` data-row index). Each caption carries a
`train` / `val` tag and `<split> <position in split>`, so `val 6995` keeps its research meaning. Inside a set,
best = ascending error, worst = descending error, ties by row ascending.

## Saved parameters

The detail view's **Parameters** column lists the sampled parameters of the image on screen, grouped by
section (collapsible; it moves below the panels in a narrow window). They come from
`<data-dir>/params.jsonl`, written by the generator (`--save-params`) or by the studio: tick **save
parameters** in the Generate panel. A dataset generated without it shows "No parameters saved for this
dataset". The viewer never loads the file whole: it seeks to one line per request through a line-offset index
cached next to the rank cache (`params_<key>.npy`), accepts CRLF endings, treats a torn last line as "no
parameters saved for this image", and refuses (HTTP 409) a record whose `filename` differs from the image's or whose `tip_x` / `tip_y` is more than
0.001 px away from the row's `labels.csv` tip (a `params.jsonl` of another run has the same names). It re-checks the
file's size and mtime on every request: a file edited since the viewer started answers 409 `params.jsonl changed on
disk since the viewer started; restart the viewer`, a deleted one answers 404 `no parameters saved for this dataset`.
`GET /api/params/{row}` returns the record (its `row` is always the requested dataset row; a non-finite number such
as `1e999` comes back as the string `"Infinity"` / `"-Infinity"`, a leading UTF-8 BOM is ignored). Sections the
person collapsed stay collapsed while stepping or opening another entry.

The image URLs carry the rank-cache key (`v=<key>`), so a restart on other weights or data never shows
stale pictures from the browser cache.

## Tests

```bash
python -m pytest -q val_viewer/tests/test_api.py           # backend: ranking, sets, numbers vs the trainer, cache, parameters, CLI
python -m pytest -q val_viewer/tests/test_e2e_browser.py   # real server + headless Chromium (Playwright)
```

Both build a fake 200-image dataset and a random TipNet in a temp folder (CPU, no real data needed).
The browser test is skipped when Playwright is not installed.
