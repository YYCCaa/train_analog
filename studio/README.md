# v7 Dial Studio

A local web app for tuning the native-gray v7 dial generator
(`generate_synthetic_v7_mp.py`). Every parameter in `configs/v7_default.yaml`
has a control. A 3x3 preview grid re-renders in RAM as you move them. When the
look is right, you generate a dataset to disk from the same page.

The studio binds to `127.0.0.1` only. Nothing is written to disk except by the
Generate panel, and only into the absolute folder you type there.

## Run it (normal)

From anywhere (the launcher switches to the repo root itself):

```
python studio/run_studio.py               # http://127.0.0.1:8765/, opens a browser tab
python studio/run_studio.py --port 9000 --no-browser
```

On the first run, or after any frontend source changed, the launcher builds the
frontend first: `npm ci` (only if `studio/frontend/node_modules/` is missing),
then `npm run build`. This needs Node 18+ and npm on `PATH`. After that it
serves the API and the built page (`studio/frontend/dist/`) from one uvicorn
process.

## Run it (frontend development)

Use two terminals and hot reload:

```
python studio/run_studio.py --no-browser  # terminal 1: API on 127.0.0.1:8765
cd studio/frontend && npm run dev         # terminal 2: Vite on http://localhost:5173/
```

Vite proxies `/api` to `http://127.0.0.1:8765` (`vite.config.ts`). Open the
Vite URL. Edits to `src/` reload instantly.

## How the pieces talk

```
 browser (React, studio/frontend/src)                      FastAPI (studio/backend/app.py)
 +-------------------------------------+                 +-----------------------------------+
 | controls  (ParamControl x N leaf)   |  GET  /api/schema, /api/config/default              |
 |   slider / number / checkbox        |---------------->| v7_config: schema, defaults,      |
 |        | change                     |                 |            parse / validate / dump|
 |        v  debounce 150 ms, seq += 1 |  POST /api/preview {config, seeds[9]}               |
 | preview grid (PreviewGrid)          |---------------->| ProcessPool --> preview_worker    |
 |   applies a response only if its    |<----------------|   render_image(cfg, seed)         |
 |   seq is the newest one sent        |  tiles: png_b64 + tip_x/tip_y   (same function the  |
 |   tip overlay = client-side <svg>   |                 |    CLI and the job use)           |
 |                                     |  POST /api/config/parse | /api/config/dump          |
 | Load YAML / Save YAML               |<--------------->| YAML text <-> config (422 errors) |
 |                                     |  POST /api/generate, GET /api/jobs/current (500 ms) |
 | generate panel (GeneratePanel)      |---------------->| JobRunner thread -->              |
 |   progress bar, cancel              |  POST /api/jobs/current/cancel   generate_dataset |
 +-------------------------------------+                 +-----------------+-----------------+
                                                                           |
                                                                           v
                                             <output_dir>/dial_0000.png ... labels.csv config.yaml
```

Preview tile seed `S` is pixel-identical to the disk image whose per-image seed
is `S` under the same config, because both paths call `render_image`.

## Using the page

- **Controls (left).** One block per YAML section. Ranges have a two-handle
  slider plus `lo`/`hi` number boxes. Number boxes commit on Enter or when
  they lose focus. `canvas.img_size` is read-only (training expects 256).
- **Preview (right).** 9 fixed seeds. *Reshuffle seeds* picks 9 new ones.
  *tip overlay* draws a marker at each tile's `(tip_x, tip_y)` label. It never
  changes the image.
- **Load YAML / Save YAML.** Save downloads the current settings as
  `v7_config.yaml`. Load accepts that file, a partial YAML (missing keys take
  defaults), or a generated `config.yaml` (its `run:` block is ignored).
  Invalid values are listed in red with the `section.key` they belong to. The
  current settings stay unchanged.
- **Generate dataset.** Fill in an absolute output folder, the image count,
  the base seed, and the workers (blank = generator default). Then press
  Start. A missing folder is created. A folder that already holds
  `dial_*.png` / `labels.csv` / `config.yaml` is refused unless *overwrite* is
  ticked. With overwrite, only those files are removed. Cancel stops the run
  and leaves a consistent folder: the `labels.csv` rows match the PNGs, and
  `config.yaml` has `run.completed: false`.

## Where outputs go

Only to the folder typed into the Generate panel:

| file | content |
|---|---|
| `dial_0000.png` ... | 256x256 single-channel PNGs |
| `labels.csv` | `filename, angle_deg, tip_x, tip_y` |
| `config.yaml` | full render config plus a `run:` block (`num_images`, `seed`, `workers`, `completed`, timestamp) |

To reproduce a dataset from the CLI:
`python generate_synthetic_v7_mp.py --config <folder>/config.yaml ...`.

## Tests

```
python -m pytest -q tests/test_studio_api.py   # backend via TestClient
python -m pytest -q tests/test_studio_e2e.py   # real Chromium against run_studio.py on port 8799
```

The E2E test needs Playwright's Chromium (`python -m playwright install chromium`).
It builds the frontend through the launcher if `dist/` is missing or stale.
