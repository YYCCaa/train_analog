# Val viewer

Look at the **best and worst n validation images of a TipNet v9 run**, each as the 256 x 256 dial
next to its 128 x 128 softmax heatmap, with the true tip (green crosshair) and the predicted tip
(orange x) drawn on both. Click an entry for a 512 px detail view with two x8 zooms of the 32 x 32 px
window around the true tip.

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

The first start runs a ranking pass over the whole val split and prints `[rank] done/total` lines
(about 20 s on the GPU for 22,500 images). The result (predicted tip, error, confidence, val rows) is
cached in `val_viewer/.cache/rank_<key>.npz`; a warm start loads it in about 1 s (`[rank] cached ...`).
The key covers data dir, val split, `labels.csv`, the decoded-image cache, the weights, the device and
`train_v9_tip.py`, so a change of any of them re-ranks by itself. A corrupt or mismatching file is
recomputed. To clear it, delete `val_viewer/.cache/` (or whatever `--cache-dir` points to).
The decoded PNGs also live in the trainer's `<data-dir>_cache_*` files, built once if missing.

## Controls

| Control | Effect |
|---|---|
| n field (1..1000, Enter) | how many best and worst entries per column |
| heatmap log / linear | colour scale of every heatmap (log = 6 decades below the maximum) |
| markers checkbox | show or hide the true / predicted tip markers |
| click an entry, or Tab to it + Enter / Space | open the detail view |
| Left / Right arrow, prev / next | step through the list in the detail view |
| Esc, Close | close the detail view (focus returns to the entry) |

The image URLs carry the rank-cache key (`v=<key>`), so a restart on other weights or data never shows
stale pictures from the browser cache.

## Tests

```bash
python -m pytest -q val_viewer/tests/test_api.py           # backend: ranking, numbers vs the trainer, cache, CLI
python -m pytest -q val_viewer/tests/test_e2e_browser.py   # real server + headless Chromium (Playwright)
```

Both build a fake 200-image dataset and a random TipNet in a temp folder (CPU, no real data needed).
The browser test is skipped when Playwright is not installed.
