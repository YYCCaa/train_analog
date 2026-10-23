# Grayscale web app

A small, self-contained web app that converts any image to grayscale. Pick, drag-and-drop or paste
an image, choose a conversion method and output format, compare original and result side by side,
and download the result.

- Backend: FastAPI + Pillow + NumPy (`app/`)
- Frontend: plain HTML/CSS/JS, no build step, no CDN, works offline (`static/`)
- Nothing is stored: each upload is converted in memory and streamed back.

The folder imports nothing from outside itself, so it can be copied into its own repository as-is.

## Run

```bash
pip install -r grayscale_webapp/requirements.txt
python -m grayscale_webapp                 # http://127.0.0.1:8000/
python -m grayscale_webapp --host 0.0.0.0 --port 9000
```

Run the command from the directory that *contains* `grayscale_webapp/`.

## Architecture

```
  Browser (static/index.html + app.js)                    Server (python -m grayscale_webapp → uvicorn)
 ┌──────────────────────────────────────┐               ┌──────────────────────────────────────────────┐
 │ drop zone / Choose image / Ctrl+V    │               │ app/main.py  (FastAPI, HTTP only)            │
 │        │  client pre-check:          │  GET /api/    │   /                → static/index.html       │
 │        │  image/*, ≤ 20 MB           │  methods      │   /static/*        → css, js                 │
 │        ▼                             │◀─────────────▶│   /api/health      → {"status":"ok"}         │
 │ FormData{file, method, format} ──────┼── POST ──────▶│   /api/methods     → [{id,label,description}]│
 │                                      │  /api/        │   /api/grayscale   ──┐                       │
 │ result <img> ◀── blob + headers ─────┼◀─ grayscale ──│                      │ maps exceptions:      │
 │  X-Image-Width/Height, X-Gray-Method │               │                      │ 400 / 413 / 422       │
 │  Content-Disposition → Download name │               │                      ▼                       │
 │ error box (role=alert) ◀── {"detail"}│               │ app/converter.py  (pure, no web code)        │
 └──────────────────────────────────────┘               │   bytes ─open+guards─▶ first frame           │
                                                        │         ─EXIF upright─▶ RGB(A) uint8         │
                                                        │         ─method─▶ gray (H,W) uint8 [+alpha]  │
                                                        │         ─encode PNG/JPEG/WebP─▶ bytes        │
                                                        └──────────────────────────────────────────────┘
```

## Conversion methods

All methods take 8-bit RGB and produce 8-bit gray. Non-Pillow methods round half up (`floor(x + 0.5)`).

| id          | formula                                   | notes                                         |
|-------------|-------------------------------------------|-----------------------------------------------|
| `luma601`   | 0.299 R + 0.587 G + 0.114 B               | default; bit-identical to Pillow `convert("L")` |
| `luma709`   | 0.2126 R + 0.7152 G + 0.0722 B            | HDTV / sRGB weights                           |
| `average`   | (R + G + B) / 3                           |                                               |
| `lightness` | (max(R,G,B) + min(R,G,B)) / 2             |                                               |

## Image handling

- EXIF orientation is applied, so phone photos come out upright (EXIF is not copied to the output).
- **Animated images (GIF, WebP, APNG) are converted from their first frame only.**
- Palette, CMYK, 16-bit and other modes are normalised to 8-bit RGB(A) first.
- Transparency is kept (`LA`) when the output is PNG or WebP; JPEG output is flattened onto white.
- Output format `same` keeps PNG, JPEG and WebP; any other input (GIF, BMP, TIFF, ...) becomes PNG.
- JPEG is written at quality 95; WebP is written lossless so gray values are exact.
- Limits: 20 MB per file and 50 megapixels per image (decompression-bomb guard).

## API

| Method | Path              | Description |
|--------|-------------------|-------------|
| GET    | `/`               | the web page |
| GET    | `/api/health`     | `{"status": "ok"}` |
| GET    | `/api/methods`    | `[{"id", "label", "description"}, ...]` |
| POST   | `/api/grayscale`  | multipart form: `file` (required), `method` (default `luma601`), `format` (`same` \| `png` \| `jpeg` \| `webp`, default `same`) |

Successful `POST /api/grayscale` returns `200` with the image bytes and headers:

```
Content-Type: image/png | image/jpeg | image/webp
Content-Disposition: attachment; filename="<stem>_gray.<ext>"; filename*=UTF-8''<stem>_gray.<ext>
X-Image-Width: <int>    X-Image-Height: <int>    X-Gray-Method: <method id>
```

Errors are JSON `{"detail": "<human readable message>"}`:

| Status | When |
|--------|------|
| 400 | no file, empty file, not an image, corrupt/truncated, unsupported pixel format |
| 413 | file larger than 20 MB or image larger than 50 megapixels |
| 422 | unknown `method` or `format` |

Example:

```bash
curl -F file=@photo.jpg -F method=luma709 -F format=png -OJ http://127.0.0.1:8000/api/grayscale
```

## Tests

```bash
pip install pytest httpx playwright
python -m playwright install chromium
python -m pytest -q grayscale_webapp/tests      # from the directory containing grayscale_webapp/
```

- `test_converter.py` – numeric correctness of every method, alpha, JPEG flattening, EXIF, format fallback, bad input.
- `test_api.py` – HTTP contract via FastAPI's `TestClient` (status codes, headers, filenames).
- `test_e2e_browser.py` – starts the real server in a subprocess and drives the page in headless
  Chromium like a person: choose a file, drag-and-drop, change method/format, download and verify
  the pixels, error messages, keyboard access, reset, and a 375 px mobile layout.
  Skipped automatically if Playwright is not installed.
