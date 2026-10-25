# Tip finder web app

Find the **pointer tip of an analog dial** in a photo with the shipped TipNet v9 TFLite model.
Open an image (file picker, drag-and-drop, Ctrl+V paste, one camera frame, or a labelled sample).
Select a square around the dial the way MS Paint's select tool works, and the app predicts the tip.
It shows the cues behind the answer: a probability heatmap (log or linear), the tip with its 7-px
error-budget ring, look-alike candidate peaks, the exact 256 × 256 picture the model saw, and a
×5 zoom of the decode window.

- Backend: FastAPI + Pillow + NumPy + LiteRT (`ai_edge_litert`), inference on the server (`app/`)
- Frontend: plain HTML/CSS/JS, no build step, no CDN, works offline (`static/`)
- Model: `model/tipnet_v9.tflite` (float32, 416,408 B) + `model/model_card.json`
- Samples: `samples/v9_val_1..6.png` (256 × 256 synthetic validation dials) + `samples/samples.json` labels
- Nothing is stored: each upload is processed in memory.

The folder imports nothing from outside itself (no torch, no training code), so it can be copied
into its own repository as-is. Only one test imports `train_v9_tip.py`, to cross-check the decode,
and it skips itself when that file or torch is missing.

## Run

```bash
pip install -r tip_webapp/requirements.txt
python -m tip_webapp                                  # http://127.0.0.1:8010/
python -m tip_webapp --host 0.0.0.0 --port 9000       # reachable from the LAN (camera: see below)
python -m tip_webapp --model path/to/other.tflite     # another model with the same I/O
```

Run the commands from the directory that *contains* `tip_webapp/`. The model is loaded at start-up,
so a wrong `--model` fails immediately with exit code 2. A `model_card.json` next to a custom model
is shown in the UI if present; otherwise the shipped card is shown.

## Using it

| Action | How |
|--------|-----|
| Get an image | **Open image…** · drop a file on the stage · **Ctrl+V** · **Camera** → **Capture** · **Samples** |
| Draw a square | press and drag on the image, like Paint. **Pixel-inclusive**: the pixel under the press and the pixel under the pointer are both inside the square, wherever in each pixel the mouse is. With dx, dy = pointer pixel − press pixel, side = max(\|dx\|, \|dy\|) + 1, growing into the drag's quadrant (right/down when dx/dy ≥ 0, else left/up), clamped to the image |
| Move it | drag inside the square (cursor `move`) |
| Resize it | drag one of the 4 corner handles; the opposite corner stays put and it stays square. Handles sit on pixel *edges*, so the pointer snaps to the nearest edge here (unlike drawing) |
| Nudge / size | **← ↑ → ↓** 1 px (**Shift** 10 px) · **+ / −** grow / shrink by 2 px around the centre |
| Predict | **Predict** button or **Enter**; with **Auto-predict on release** (default on) it runs after every drag, and 300 ms after the last key |
| Cancel a drag | **Esc** puts the previous square back |
| Zoom / pan | mouse wheel zooms around the cursor (fit … 16×) · middle-drag, **Space**+drag or two fingers pan · two-finger pinch zooms · **Fit** resets |

After an image loads, the **largest centred square** is preselected and predicted. This automatic
square is *soft*: it usually covers most of the image, so pressing inside it draws a new square
instead of moving it (its corner handles and the keys still work). The status bar marks it
`(auto)`. As soon as you draw, move, resize or nudge, it is your square and dragging inside moves
it, as in Paint.

The Paint-style status bar shows the cursor position in image px, the selection
`left,top · size×size px` and the image size. Everything works with mouse, touch and pen (Pointer
Events). The stage is focusable (`tabindex=0`) for the keyboard.

**Tips for good results.** The model was trained on 256 × 256 synthetic dials that fill the frame.
Select a square that just contains the dial face. A square much larger than the dial shrinks the
pointer below what the model was trained on, and the heatmap then spreads over look-alikes.

## Architecture

```
  Browser (static/index.html + app.js)                             Server (python -m tip_webapp -> uvicorn)
 ┌─────────────────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ file / drop / paste / sample ──┐                 │ GET /api/  │ app/main.py  (FastAPI, HTTP only)            │
 │ camera: getUserMedia ─ Capture ┤ (ONE frame,     │ model      │   /api/health   -> {"status": "ok"}          │
 │   stops every track)           │  camera off)    │◀──────────▶│   /api/model    -> card + file size + I/O    │
 │                                ▼                 │ samples    │   /api/samples  -> samples.json + urls       │
 │ createImageBitmap(imageOrientation: from-image)  │            │   /samples/*    -> sample PNGs               │
 │        ▼                                         │            │   /api/predict  ──┐ 400 / 413 / 422 detail   │
 │ working canvas W × H (opaque, upright)           │            │                   ▼                          │
 │        │ Paint-style square (left, top, size)    │            │ app/inference.py  (pure, no web code)        │
 │        ▼                                         │  POST      │   bytes ─open, guards, exif_transpose─▶ img  │
 │ crop canvas side × side (side = min(size, 1024)) ├──────────▶ │   crop box ─convert("L")─▶ resize 256 Lanczos│
 │   PNG + box "0,0,side"                           │ multipart  │   /255 ─▶ TFLite (1 interpreter, a lock)     │
 │        ▲                                         │            │   logits 128² ─softmax─▶ decode, candidates  │
 │ X = left + (x + 0.5) · size/side − 0.5  ◀── JSON ┤◀────────── │   JSON: tip, heatmap, model input PNG, ...   │
 │ stage / model view / zoom: heatmap + markers     │            └──────────────────────────────────────────────┘
 └─────────────────────────────────────────────────┘
```

**The browser owns the image.** Whatever the source, it is decoded upright into a working canvas.
You select a square in that canvas's natural pixels. On Predict, the browser crops exactly that
square (WYSIWYG) and posts it as a PNG. A square larger than 1024 px is first downsized to 1024
with `imageSmoothingQuality = "high"`. The returned coordinates are mapped back into image px, so
the server never needs EXIF tricks for the UI path. Transparent pixels are composited onto white
in the working canvas, so the stage shows exactly what is uploaded. Images over 40 MP are refused
(same limit as the API). Images over 4096² px are downscaled for the working canvas, with a notice.

## Coordinates and decode

Every coordinate uses the **centre convention**: pixel *i* is centred at coordinate *i*, and its
edges are at *i − 0.5* and *i + 0.5*. Every mapping below scales pixel *edges*. Each grid exactly
tiles its parent, so the mappings are exact and invertible.

```
 uploaded image (W × H)                     crop (size × size)          model input (256 × 256)       heatmap (128 × 128)
┌────────────────────────────────┐          ┌──────────────┐           ┌──────────────┐             ┌──────┐
│        left                    │  crop    │              │  resize   │              │  TipNet     │      │
│   top ┌──────────┐             │ ───────▶ │   X_crop     │ ────────▶ │   x_model    │ ──────────▶ │ u, v │
│       │  square  │ size        │          │              │ Lanczos   │              │  logits     │      │
│       └──────────┘             │          └──────────────┘           └──────────────┘             └──────┘
└────────────────────────────────┘
   x = left + X_crop              X_crop = (x_model + 0.5) · size / 256 − 0.5        x_model = 2u + 0.5
                                  x_model = (X_crop + 0.5) · 256 / size − 0.5        (one cell = 2 × 2 model px)
```

The decode is identical to `train_v9_tip.decode_logits`, ported to NumPy and cross-checked in the tests:

```
 logits (128, 128) ──argmax──▶ cell (iy, ix)
          │                         │  take the 5 × 5 logits around it (cells off the map are ignored)
          │                         ▼
          │                 softmax over those ≤ 25 logits ──▶ weights w
          │                         ▼
          │                 centroid u = Σ w · col, v = Σ w · row   (heatmap px)
          │                         ▼
          │                 tip_model = (2u + 0.5, 2v + 0.5)       (model-input px)
          ▼
 softmax over all 16,384 cells = probability map (sums to 1) ──▶ heatmap, confidence, candidates
```

## Cues

| Cue | What it shows |
|-----|---------------|
| **Heatmap** (stage + model view) | The softmax probability of every heatmap cell, drawn over the predicted square and rendered client-side from the float32 data (toggling never re-requests). **Log** (default) maps log₁₀ p clipped to [−6, 0], so secondary hypotheses such as shadows, decoy triangles and reflections stay visible. **Linear** maps p ÷ peak, which is honest but usually a single dot. Colormap: inferno, with alpha rising with the value so the photo stays visible. Opacity slider 0–100 %. |
| **Tip marker** | A cyan crosshair at the decoded tip, drawn at the pixel centre (screen = (X + 0.5) · scale + offset). The ring is the 7-px error budget at model scale, i.e. radius 7 × size / 256 image px. The label is the 5 × 5 mass. |
| **Candidates** | White hollow circles with their mass %, one per look-alike peak, so a strong competitor is visible on the image. |
| **Verdict chip** | **⚠ Ambiguous** when the runner-up mass is **> 0.2 × the winner's** mass, or when the confidence is **< 0.5**. Otherwise **✓ Clear peak**. |
| **Model view** | The 256 × 256 gray image the model actually saw (`model_input_png`), with the heatmap, markers and a dashed outline of the zoom window. |
| **Tip zoom ×5** | 48 × 48 model px around the tip, nearest-neighbour, with the 5 × 5-cell decode window (10 × 10 model px) dashed and its cell grid. |
| **Readout** | The tip in image px and in model px, confidence %, peak probability, runner-up % (and its ratio to the winner), the square, and the timings (preprocess, inference, server total, browser round trip). Samples also show the ground truth (a green diamond on every view) and the error in px, against the 7-px budget. |
| **Header badge** | Model name · file KB · validation p99, from `/api/model`. **About the model** lists the whole card plus the interpreter's real input/output shapes. |

The readout numbers are mirrored on `#readout` as `data-*` attributes for tests: `data-tip-x/y`,
`data-tip-model-x/y`, `data-confidence`, `data-peak-prob`, `data-runner-up`, `data-ambiguous`,
`data-box-left/top/size`, `data-seq`, plus `data-gt-x/y` and `data-error-px` for samples. The stage
mirrors its view transform and selection: `data-scale`, `data-offset-x/y`, `data-natural-w/h`,
`data-sel-left/top/size`, `data-sel-auto`, `data-heatmap`, `data-heatmap-scale` and
`data-heatmap-opacity`. Image edge *e* sits at screen `e · scale + offset`.

## Camera

**Camera** opens a live preview inline (`getUserMedia`, `facingMode: "environment"`, ideally
1920 × 1080). A camera picker appears when there is more than one camera. **Capture** copies ONE
frame into the working canvas and then **stops every track**, so the camera light goes off.
**Cancel**, switching to another image and hiding the tab also stop the tracks.

Browsers expose cameras only in a **secure context**: `https://`, `http://localhost` or
`http://127.0.0.1`. Plain `http://` on a LAN IP (e.g. `--host 0.0.0.0` opened from a phone as
`http://192.168.1.20:8010`) is *not* a secure context, and the app says so. Use localhost on the
same machine, or put the server behind https. On a phone, **Open image…** can open the camera app
directly instead. Permission-denied, no-camera and camera-busy errors each get their own message.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/` | the web page |
| GET  | `/api/health` | `{"status": "ok"}` |
| GET  | `/api/model` | the model card + `served_file`, `file_bytes` and `interpreter` (`runtime`, `num_threads`, `input_shape`, `input_dtype`, `output_shape`, `output_dtype`) |
| GET  | `/api/samples` | `{"coords", "samples": [{"id", "file", "url", "label": {"x", "y"}, "note", "source"}]}` |
| GET  | `/samples/<file>` | a sample PNG |
| POST | `/api/predict` | multipart form: `file` (required), `box` (optional) |

`POST /api/predict` fields:

- `file`: the image. Anything Pillow reads (PNG, JPEG, WebP, GIF/APNG first frame, BMP, TIFF,
  16-bit PNG …), at most **20 MB** and **40 MP**.
- `box`: `"left,top,size"`, three integers in the uploaded image's px *after*
  `ImageOps.exif_transpose`, fully inside the image, with size ≥ 16. If it is omitted or blank,
  the server uses the largest centred square, `size = min(W, H)`, `left = (W − size) // 2`,
  `top = (H − size) // 2`.

Server pipeline: exif_transpose → crop box → `convert("L")` (16-bit images are scaled to 8 bit
first instead of clipping) → resize 256 × 256 Lanczos (a 256-px box is not resampled at all) →
/255 → TFLite → logits (128, 128) → softmax → decode.

### Response (200)

| Field | Type | Meaning |
|-------|------|---------|
| `image` | `{width, height}` int | the uploaded image's size after EXIF rotation |
| `box` | `{left, top, size}` int | the square actually used (yours or the default) |
| `tip` | `{x, y}` float | the decoded tip in **uploaded-image px**: `x = left + (x_model + 0.5) · size / 256 − 0.5` |
| `tip_model` | `{x, y}` float | the decoded tip in **256-px model-input px** |
| `peak_cell` | `{x, y}` int | the argmax heatmap cell (column, row) the decode window is centred on |
| `confidence` | float 0–1 | the softmax probability mass inside the 5 × 5 decode window |
| `peak_prob` | float 0–1 | the maximum probability of any cell |
| `candidates` | list (≤ 5) | the local maxima of the probability map (the largest value in their 9 × 9 cell neighbourhood) whose 5 × 5 `mass` is ≥ 0.001. Each has `x`, `y` (image px), `x_model`, `y_model` (model px, decoded like the tip), `mass` (5 × 5 sum), `peak` (cell probability) and `cell` `{x, y}`. The first is the winner, the argmax peak, and its coordinates are exactly `tip`. The rest are sorted by mass, largest first. Tied plateaus are reported once. They expose look-alikes such as shadows, decoys and reflections. |
| `heatmap` | object | `{width: 128, height: 128, dtype: "float32", encoding: "base64-le", data}`: the softmax probabilities, row-major little-endian float32, base64. Cell (row v, col u) covers model px 2u … 2u + 1. |
| `model_input_png` | string | a `data:image/png;base64,…` URL of the 256 × 256 gray image the model saw |
| `timing_ms` | `{preprocess, inference, total}` float | decode + crop + gray + resize, the TFLite call, and the whole request on the server |

### Errors

Errors are JSON `{"detail": "<human readable message>"}`:

| Status | When |
|--------|------|
| 400 | no file, an empty file, not an image, a corrupt or truncated image, a malformed form |
| 413 | the file is larger than 20 MB or the image is larger than 40 megapixels |
| 422 | the box is malformed (not three integers), lies outside the image, or is smaller than 16 px (this includes images under 16 px with no box) |

Example:

```bash
curl -F file=@dial.jpg -F box=120,40,600 http://127.0.0.1:8010/api/predict
```

## Model

From `model/model_card.json`: TipNet v9, 100,001 parameters, trained on 150,000 synthetic 256 × 256
gray dials (`sv9_150k`), exported to float32 TFLite (it matches the PyTorch model to 7e-5).
Validation on 22,500 images: mean 0.58 px, **median 0.42 px**, p95 1.55 px, **p99 3.07 px**,
13 images over 7 px, max 140 px. Sample 6 is that worst image. The model picks a look-alike 140 px
away (the opposite end of the needle-shaped triangle), and the true tip appears as the runner-up
candidate (19 %), which the ambiguous chip flags. It is kept as an honest failure demo.

## Tests

```bash
pip install pytest httpx playwright
python -m playwright install chromium
python -m pytest -q tip_webapp/tests      # from the directory containing tip_webapp/
```

- `test_inference.py`: box parsing and validation, the default box, byte/pixel limits, EXIF,
  gray + Lanczos preprocessing (bit-exact for 256-px boxes), the exact model↔image mapping, the
  NumPy decode against `train_v9_tip.decode_logits` on 62 maps incl. every border and corner
  (< 1e-4 px; skipped without torch), the candidate finder on synthetic 2-peak and plateau maps, and
  the real model on the 6 samples (1–5 within 7 px, 6 a known > 7 px failure; errors are printed).
- `test_api.py`: the HTTP contract via `TestClient`. It covers health, model, samples and static
  files, and a predict on every sample (the full response schema). A sample pasted at (137, 61)
  into a 640 × 480 canvas with `box=137,61,256` gives the raw tip + offset within 1e-3 px. A
  2×-upscaled composite with a 512 box is within 1.5 px of the 2×-mapped label. It also covers the
  default box on a non-square image, EXIF-upright boxes, every 400/413/422 case, and the
  `--model` guard.
- `test_e2e_browser.py`: starts the real `python -m tip_webapp` in a subprocess (killed in teardown)
  and drives headless Chromium like a person. It opens a 900 × 700 noisy composite through the
  file chooser, mouse-drags the dial's square from the centre of its first pixel to the centre of its
  last (Paint's pixel-inclusive rule), and checks the auto-predicted tip within 3 px. A separate test
  zooms to >= 4x and draws down-right (press at 0.8 of a pixel, release at 0.2 of another) and up-left,
  checking the exact pixel-inclusive squares. It
  presses ArrowRight ×3, drags inside to move, drags corner handles (stays square, anchored,
  clamped), and uses −/Enter. It wheel-zooms around the cursor, pans with Space+drag and
  middle-drag, resets with Fit, and cancels a drag with Esc. It toggles the heatmap and changes
  opacity and log/linear while checking canvas pixels. It picks sample 1 (GT marker + error < 7 px) and sample 6 (ambiguous
  chip + large error), drops a file, uploads a non-image (alert), and checks the insecure-context
  camera message and a 375 px phone layout. The **camera** test launches Chromium with
  `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=<dial>.y4m`. The y4m is a
  sample dial padded into 640 × 480 I420 frames, written by the test. The test clicks Camera,
  checks Cancel, then Capture, asserts every track is `ended` and the image is 640 × 480, then
  selects the dial: the tip lands within 3 px of label + pad. Skipped automatically if Playwright
  is not installed.
