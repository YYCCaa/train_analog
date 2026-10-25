// Tip finder frontend: get an image (file / drop / paste / camera / sample) into a working canvas,
// let the user select a square like MS Paint's select tool, crop exactly that square, POST it to
// /api/predict, and draw the cues (heatmap, tip, 7-px ring, candidates, ground truth) on top.
//
//   source (File | camera frame | sample)                       server
//        │ createImageBitmap(imageOrientation: from-image)        ▲
//        ▼                                                        │ PNG crop (side ≤ 1024) + box "0,0,side"
//   work.canvas  W × H, opaque, upright  ──crop sel──▶ crop canvas ┘
//        │                                                        │ JSON: tip / candidates in crop px,
//        │ drawn with the view transform                          │ tip_model in 256 px, heatmap 128²
//        ▼                                                        ▼
//   stage canvas (CSS px):  screenX = (X + 0.5) * scale + offsetX   ◀── X = left + (x + 0.5) * size/side − 0.5
//
// Coordinates: "image px" use the centre convention (pixel i is centred at i). The selection and the
// view work in pixel EDGES (pixel i spans [i, i + 1]), so edge e sits at screen e * scale + offsetX.
// Drawing a NEW square is pixel-inclusive like Paint (the pixels under the press and the pointer are both
// inside); resizing by a corner handle snaps to the nearest pixel edge.
"use strict";

(() => {
  // ---- constants ---------------------------------------------------------------------------------

  // MODEL_SIZE: number, side of the model input in model px (256)
  const MODEL_SIZE = 256;
  // HEAT_SIZE: number, side of the returned heatmap in cells (128); one cell = 2 × 2 model px
  const HEAT_SIZE = 128;
  // MIN_BOX: number, smallest selection side in image px (the server refuses smaller boxes)
  const MIN_BOX = 16;
  // MAX_UPLOAD_SIDE: number, crops larger than this are downsized before upload (px)
  const MAX_UPLOAD_SIDE = 1024;
  // MAX_PIXELS: number, largest image the app accepts (40 MP, same as the API)
  const MAX_PIXELS = 40_000_000;
  // WORK_CAP_PIXELS: number, working-canvas pixel cap (4096², safe on phone browsers); bigger images are downscaled
  const WORK_CAP_PIXELS = 4096 * 4096;
  // MAX_ZOOM: number, largest view scale (screen px per image px)
  const MAX_ZOOM = 16;
  // FIT_PAD: number, CSS px of margin kept around the image when fitting it
  const FIT_PAD = 12;
  // BUDGET_MODEL_PX: number, radius of the error-budget ring at model scale (model px)
  const BUDGET_MODEL_PX = 7;
  // AMBIGUOUS_RUNNER_RATIO: number, runner-up mass above this fraction of the winner's marks "ambiguous"
  const AMBIGUOUS_RUNNER_RATIO = 0.2;
  // AMBIGUOUS_MIN_CONFIDENCE: number, a 5×5 confidence below this marks "ambiguous"
  const AMBIGUOUS_MIN_CONFIDENCE = 0.5;
  // LOG_FLOOR: number, log10 p is clipped to [LOG_FLOOR, 0] for the log heatmap
  const LOG_FLOOR = -6;
  // KEY_PREDICT_DELAY_MS: number, auto-predict waits this long after the last keyboard nudge
  const KEY_PREDICT_DELAY_MS = 300;
  // DRAG_CLICK_PX: number, a draw gesture shorter than this (CSS px) is a click and keeps the old selection
  const DRAG_CLICK_PX = 4;
  // ZOOM_WIN: number, side of the tip-zoom window in model px; ZOOM_K: number, its magnification
  const ZOOM_WIN = 48;
  const ZOOM_K = 5;
  // INFERNO_HEX: string, 64 inferno colormap stops as rrggbb hex (matplotlib), interpolated to 256 below
  const INFERNO_HEX =
    "00000402010a04031207051b0b072410092d150b371b0c41210c4a280b532f0a5b3609613d0965440a684a0c6b510e6c" +
    "59106e5f136e65156e6c186e721a6e781c6d7f1e6c85216b8c23699225689827669f2a63a52c60ab2f5eb1325ab73557" +
    "bf3952c43c4eca404acf4446d44842d94d3dde5238e25734e65d2fea632aed6925f06f20f3761bf57d15f78410f98b0b" +
    "fa9407fb9b06fca309fcaa0ffcb216fbba1ffac228f9c932f7d13df5d949f4e156f2e865f1ef75f3f586f6fa96fcffa4";

  // LUT: Uint8ClampedArray (256 * 3), RGB of the colormap at t = i / 255
  const LUT = (() => {
    const stops = [];
    for (let i = 0; i < INFERNO_HEX.length; i += 6) {
      stops.push([0, 2, 4].map((o) => parseInt(INFERNO_HEX.slice(i + o, i + o + 2), 16)));
    }
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i += 1) {
      const f = (i / 255) * (stops.length - 1);
      const a = Math.floor(f);
      const b = Math.min(stops.length - 1, a + 1);
      const w = f - a;
      for (let c = 0; c < 3; c += 1) out[i * 3 + c] = stops[a][c] * (1 - w) + stops[b][c] * w;
    }
    return out;
  })();

  // ---- element refs ------------------------------------------------------------------------------

  // $: (id: string) => HTMLElement, shorthand lookup by element id
  const $ = (id) => document.getElementById(id);

  // el: object of HTMLElement refs used throughout the UI
  const el = {
    app: document.querySelector("[data-testid=app]"),
    badge: $("model-badge"),
    openButton: $("open-button"),
    fileInput: $("file-input"),
    cameraButton: $("camera-button"),
    samplesButton: $("samples-button"),
    samplesMenu: $("samples-menu"),
    samplesGrid: $("samples-grid"),
    fitButton: $("fit-button"),
    zoomLabel: $("zoom-label"),
    autoPredict: $("auto-predict"),
    predictButton: $("predict-button"),
    stage: $("stage"),
    canvas: $("stage-canvas"),
    stageEmpty: $("stage-empty"),
    cameraPanel: $("camera-panel"),
    video: $("camera-video"),
    cameraSelect: $("camera-select"),
    captureButton: $("capture-button"),
    cameraCancel: $("camera-cancel"),
    statusCursor: $("status-cursor"),
    statusSelection: $("status-selection"),
    statusImage: $("status-image"),
    errorBox: $("error-box"),
    noticeBox: $("notice-box"),
    readout: $("readout"),
    verdict: $("verdict-chip"),
    status: $("status"),
    outTip: $("out-tip"),
    outTipModel: $("out-tip-model"),
    outConfidence: $("out-confidence"),
    outPeak: $("out-peak"),
    outRunner: $("out-runner"),
    outBox: $("out-box"),
    outGt: $("out-gt"),
    outError: $("out-error"),
    outTiming: $("out-timing"),
    gtRows: document.querySelectorAll(".gt-row"),
    candidates: $("candidates"),
    heatToggle: $("heatmap-toggle"),
    heatOpacity: $("heatmap-opacity"),
    heatOpacityValue: $("heatmap-opacity-value"),
    scaleRadios: document.querySelectorAll('input[name="heatmap-scale"]'),
    scaleHelp: $("scale-help"),
    legend: $("legend-canvas"),
    legendTicks: $("legend-ticks"),
    modelCanvas: $("model-canvas"),
    zoomCanvas: $("zoom-canvas"),
    aboutList: $("about-list"),
  };
  // ctx: CanvasRenderingContext2D of the stage; modelCtx / zoomCtx: of the model view and tip zoom
  const ctx = el.canvas.getContext("2d");
  const modelCtx = el.modelCanvas.getContext("2d");
  const zoomCtx = el.zoomCanvas.getContext("2d");
  // reduceMotion: MediaQueryList, true when the user asked for less animation (no marching ants)
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  // coarsePointer: MediaQueryList, true on touch-first devices (bigger handles, phone hints)
  const coarsePointer = window.matchMedia("(pointer: coarse)");

  // ---- state -------------------------------------------------------------------------------------

  // state: mutable app state
  //   work:       null | {canvas: HTMLCanvasElement, w: number, h: number, name: string,
  //                       source: "file"|"drop"|"paste"|"camera"|"sample", sample: null | {id, label: {x, y}, note}}
  //   view:       {scale: number screen px per image px, ox: number, oy: number CSS px offset, fit: boolean}
  //   sel:        null | {left: number, top: number, size: number} integer image px (edges: left .. left + size)
  //   selAuto:    boolean, sel is the automatic preselection (largest centred square). It is "soft":
  //               pressing inside it draws a new square instead of moving it, because it usually covers
  //               almost the whole image. Any user edit (draw / move / resize / keys) makes it a normal one.
  //   drag:      null | {mode: "draw"|"move"|"resize"|"pan", pointerId, startSel,
  //                       anchor?: {x, y} (draw: integer pixel INDICES; resize: integer EDGE coords),
  //                       startImg?: {x, y}, startScreen: {x, y}, startView?: {...}}
  //   pointers:   Map<pointerId, {x, y}> CSS px positions of active pointers (for two-finger gestures)
  //   pinch:      null | {d0: number, mid0: {x, y}, scale0: number, ox0: number, oy0: number}
  //   spaceDown:  boolean, Space is held (Space + drag pans)
  //   hover:      null | {x, y} CSS px of the mouse over the stage
  //   result:     null | prediction mapped to image px (see handleResult)
  //   heat:       {on: boolean, opacity: number 0..1, mode: "log"|"linear"}
  //   heatCanvas: HTMLCanvasElement 128 × 128 RGBA, cached heatmap image; heatKey: string cache key
  //   controller: AbortController | null, aborts an in-flight prediction when a new one starts
  //   seq:        number, prediction counter (mirrors onto #readout[data-seq])
  //   keyTimer:   number | null, pending keyboard auto-predict timeout
  //   ants:       number, marching-ants dash phase
  //   dpr:        number, devicePixelRatio the stage canvas was sized for
  //   camera:     {stream: MediaStream | null, deviceId: string | null}
  //   samples:    Array<{id, file, url, label: {x, y}, note}>
  const state = {
    work: null,
    view: { scale: 1, ox: 0, oy: 0, fit: true },
    sel: null,
    selAuto: false,
    drag: null,
    pointers: new Map(),
    pinch: null,
    spaceDown: false,
    hover: null,
    result: null,
    heat: { on: true, opacity: 0.7, mode: "log" },
    heatCanvas: document.createElement("canvas"),
    heatKey: "",
    controller: null,
    seq: 0,
    keyTimer: null,
    ants: 0,
    dpr: window.devicePixelRatio || 1,
    camera: { stream: null, deviceId: null },
    samples: [],
  };
  state.heatCanvas.width = HEAT_SIZE;
  state.heatCanvas.height = HEAT_SIZE;

  // ---- small helpers -----------------------------------------------------------------------------

  // clamp: (v: number, lo: number, hi: number) => number
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  // f1 / f2 / f3: (v: number) => string with 1 / 2 / 3 decimals
  const f1 = (v) => v.toFixed(1);
  const f2 = (v) => v.toFixed(2);
  const f3 = (v) => v.toFixed(3);
  // pct: (v: number 0..1, digits?: number) => string such as "97.1 %"
  const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)} %`;

  function setAppState(name) {
    el.app.dataset.state = name; // "idle" | "loading" | "done" | "error"
  }

  function showError(message) {
    el.errorBox.textContent = message;
    el.errorBox.hidden = false;
    setAppState("error");
  }

  function clearError() {
    el.errorBox.textContent = "";
    el.errorBox.hidden = true;
  }

  function showNotice(message) {
    el.noticeBox.textContent = message || "";
    el.noticeBox.hidden = !message;
  }

  function announce(message) {
    el.status.textContent = message;
  }

  // ---- view transform ----------------------------------------------------------------------------

  // stageSize: () => {w: number, h: number} CSS px of the stage
  function stageSize() {
    return { w: el.stage.clientWidth, h: el.stage.clientHeight };
  }

  // fitScale: () => number, the scale that shows the whole working image inside the stage
  function fitScale() {
    if (!state.work) return 1;
    const { w, h } = stageSize();
    return Math.max(1e-3, Math.min((w - 2 * FIT_PAD) / state.work.w, (h - 2 * FIT_PAD) / state.work.h));
  }

  // maxScale: () => number, upper zoom limit (16×, or the fit scale for tiny images)
  const maxScale = () => Math.max(MAX_ZOOM, fitScale());

  // clampView: () => void, centres the image along an axis where it is smaller than the stage,
  // otherwise keeps the stage covered (no empty gutter can be panned into view)
  function clampView() {
    if (!state.work) return;
    const { w, h } = stageSize();
    const v = state.view;
    const iw = state.work.w * v.scale;
    const ih = state.work.h * v.scale;
    v.ox = iw <= w ? (w - iw) / 2 : clamp(v.ox, w - iw, 0);
    v.oy = ih <= h ? (h - ih) / 2 : clamp(v.oy, h - ih, 0);
  }

  function fitView() {
    state.view.scale = fitScale();
    state.view.fit = true;
    clampView();
    syncView();
  }

  // zoomAt: (sx: number, sy: number CSS px, scale: number) => void, keeps the image point under (sx, sy) fixed
  function zoomAt(sx, sy, scale) {
    const v = state.view;
    const next = clamp(scale, fitScale(), maxScale());
    const ix = (sx - v.ox) / v.scale; // edge coords of the anchor point
    const iy = (sy - v.oy) / v.scale;
    v.scale = next;
    v.ox = sx - ix * next;
    v.oy = sy - iy * next;
    v.fit = Math.abs(next - fitScale()) < 1e-9;
    clampView();
    syncView();
  }

  // toImage: (sx, sy CSS px) => {x, y} continuous image EDGE coordinates (pixel i spans [i, i + 1))
  function toImage(sx, sy) {
    const v = state.view;
    return { x: (sx - v.ox) / v.scale, y: (sy - v.oy) / v.scale };
  }

  // edgeToScreen: (e: number edge coord, axis "x"|"y") => number CSS px
  function edgeToScreen(e, axis) {
    const v = state.view;
    return e * v.scale + (axis === "x" ? v.ox : v.oy);
  }

  // centreToScreen: (X, Y centre-convention image px) => {x, y} CSS px, X sits at (X + 0.5) * scale + offsetX
  function centreToScreen(X, Y) {
    const v = state.view;
    return { x: (X + 0.5) * v.scale + v.ox, y: (Y + 0.5) * v.scale + v.oy };
  }

  // eventPoint: (e: PointerEvent | WheelEvent) => {x, y} CSS px relative to the stage's top-left
  function eventPoint(e) {
    const rect = el.stage.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  // syncView: () => void, mirrors the view onto data-* attributes and the zoom label, then redraws
  function syncView() {
    const v = state.view;
    el.stage.dataset.scale = String(v.scale);
    el.stage.dataset.offsetX = String(v.ox);
    el.stage.dataset.offsetY = String(v.oy);
    el.zoomLabel.textContent = state.work ? `${Math.round(v.scale * 100)}%` : "–";
    scheduleDraw();
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = stageSize();
    state.dpr = dpr;
    el.canvas.width = Math.max(1, Math.round(w * dpr));
    el.canvas.height = Math.max(1, Math.round(h * dpr));
    if (state.work) {
      if (state.view.fit) fitView();
      else {
        state.view.scale = clamp(state.view.scale, fitScale(), maxScale());
        clampView();
        syncView();
      }
    }
    drawStage();
  }

  // ---- selection ---------------------------------------------------------------------------------

  // normalizeSel: (s: {left, top, size}) => {left, top, size} integer square fully inside the image, side ≥ 16
  function normalizeSel(s) {
    const { w, h } = state.work;
    const maxSide = Math.min(w, h);
    const size = clamp(Math.round(s.size), Math.min(MIN_BOX, maxSide), maxSide);
    return {
      left: clamp(Math.round(s.left), 0, w - size),
      top: clamp(Math.round(s.top), 0, h - size),
      size,
    };
  }

  // squareFromPixels: (ax, ay: integer anchor pixel indices in 0..w-1 / 0..h-1, qx, qy: number continuous
  // pointer EDGE coords) => {left, top, size}. The DRAW rule, pixel-inclusive like MS Paint: the pixel
  // under the press and the pixel under the pointer are both inside the square, wherever in the pixel
  // the pointer is. side = max(|dx|, |dy|) + 1 (dx, dy = pointer pixel − anchor pixel), growing from the
  // anchor pixel into the drag's quadrant (right when dx >= 0, else left; down when dy >= 0, else up),
  // clamped to the room the image leaves in that quadrant.
  //
  //   dx >= 0 (right):  left = ax            room = w − ax        dx < 0 (left):  left = ax + 1 − side   room = ax + 1
  //   pixels:           [ax] . . . [cx]                                           [cx] . . . [ax]
  function squareFromPixels(ax, ay, qx, qy) {
    const { w, h } = state.work;
    const cx = clamp(Math.floor(qx), 0, w - 1);
    const cy = clamp(Math.floor(qy), 0, h - 1);
    const dx = cx - ax;
    const dy = cy - ay;
    const right = dx >= 0;
    const down = dy >= 0;
    const room = Math.min(right ? w - ax : ax + 1, down ? h - ay : ay + 1);
    const side = Math.min(Math.max(Math.abs(dx), Math.abs(dy)) + 1, room);
    return { left: right ? ax : ax + 1 - side, top: down ? ay : ay + 1 - side, size: side };
  }

  // squareFromAnchor: (ax, ay integer edge coords, px, py edge coords, minSide: number) => {left, top, size}
  // The RESIZE rule (corner handles sit on pixel EDGES, so the pointer snaps to the nearest edge):
  // side = max(|dx|, |dy|), growing from the anchor corner into the drag's quadrant, clamped to the image.
  function squareFromAnchor(ax, ay, px, py, minSide) {
    const { w, h } = state.work;
    const dx = Math.round(px) - ax;
    const dy = Math.round(py) - ay;
    const sx = dx < 0 ? -1 : 1;
    const sy = dy < 0 ? -1 : 1;
    const room = Math.min(sx > 0 ? w - ax : ax, sy > 0 ? h - ay : ay);
    const side = clamp(Math.max(Math.abs(dx), Math.abs(dy), minSide), 0, room);
    return { left: sx > 0 ? ax : ax - side, top: sy > 0 ? ay : ay - side, size: side };
  }

  // defaultSel: () => {left, top, size}, the largest centred square (same rule as the API's default box)
  function defaultSel() {
    const { w, h } = state.work;
    const size = Math.min(w, h);
    return { left: Math.floor((w - size) / 2), top: Math.floor((h - size) / 2), size };
  }

  const sameSel = (a, b) => !!a && !!b && a.left === b.left && a.top === b.top && a.size === b.size;

  // setSel: (s: {left, top, size} | null, auto?: boolean) => void, stores the square (auto = the soft
  // preselection), mirrors it onto data-* and the status bar, redraws
  function setSel(s, auto = false) {
    state.sel = s ? { left: s.left, top: s.top, size: s.size } : null;
    state.selAuto = !!s && auto;
    el.stage.dataset.selAuto = String(state.selAuto);
    if (state.sel) {
      el.stage.dataset.selLeft = String(state.sel.left);
      el.stage.dataset.selTop = String(state.sel.top);
      el.stage.dataset.selSize = String(state.sel.size);
      el.statusSelection.textContent = `${state.sel.left},${state.sel.top} · ${state.sel.size}×${state.sel.size} px` +
        (state.selAuto ? " (auto)" : "");
    } else {
      delete el.stage.dataset.selLeft;
      delete el.stage.dataset.selTop;
      delete el.stage.dataset.selSize;
      el.statusSelection.textContent = "–";
    }
    markStale();
    scheduleDraw();
  }

  // selScreenRect: () => {x0, y0, x1, y1} CSS px of the selection's outer edges
  function selScreenRect() {
    const s = state.sel;
    return {
      x0: edgeToScreen(s.left, "x"),
      y0: edgeToScreen(s.top, "y"),
      x1: edgeToScreen(s.left + s.size, "x"),
      y1: edgeToScreen(s.top + s.size, "y"),
    };
  }

  // handleRadius: (pointerType: string) => number CSS px within which a corner handle is grabbed
  const handleRadius = (pointerType) => (pointerType === "touch" || coarsePointer.matches ? 20 : 9);

  // hitTest: (p: {x, y} CSS px, pointerType: string) => "nw"|"ne"|"sw"|"se"|"inside"|"outside"
  // Inside the soft automatic preselection counts as "outside", so a press there draws a new square.
  function hitTest(p, pointerType) {
    if (!state.sel) return "outside";
    const r = selScreenRect();
    const rad = handleRadius(pointerType);
    const corners = { nw: [r.x0, r.y0], ne: [r.x1, r.y0], sw: [r.x0, r.y1], se: [r.x1, r.y1] };
    for (const [name, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(p.x - cx) <= rad && Math.abs(p.y - cy) <= rad) return name;
    }
    if (!state.selAuto && p.x > r.x0 && p.x < r.x1 && p.y > r.y0 && p.y < r.y1) return "inside";
    return "outside";
  }

  // CURSOR_FOR: Record<hit, string>, CSS cursor for each hit-test result
  const CURSOR_FOR = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", inside: "move", outside: "crosshair" };

  function updateCursor(p, pointerType) {
    if (!state.work) {
      el.stage.style.cursor = "default";
      return;
    }
    if (state.drag) {
      el.stage.style.cursor = state.drag.mode === "pan" ? "grabbing"
        : state.drag.mode === "move" ? "move"
          : state.drag.mode === "resize" ? CURSOR_FOR[state.drag.corner] : "crosshair";
      return;
    }
    if (state.spaceDown) {
      el.stage.style.cursor = "grab";
      return;
    }
    el.stage.style.cursor = p ? CURSOR_FOR[hitTest(p, pointerType)] : "crosshair";
  }

  function updateCursorReadout(p) {
    if (!state.work || !p) {
      el.statusCursor.textContent = "–";
      return;
    }
    const q = toImage(p.x, p.y);
    const ix = Math.floor(q.x);
    const iy = Math.floor(q.y);
    el.statusCursor.textContent =
      ix >= 0 && iy >= 0 && ix < state.work.w && iy < state.work.h ? `${ix}, ${iy} px` : "–";
  }

  // ---- pointer input (mouse, touch, pen) ----------------------------------------------------------

  function onPointerDown(e) {
    if (!state.work || !el.cameraPanel.hidden) return;
    el.stage.focus({ preventScroll: true });
    const p = eventPoint(e);
    state.pointers.set(e.pointerId, p);

    if (e.pointerType === "touch" && state.pointers.size === 2) {
      // A second finger turns the gesture into pinch-zoom / two-finger pan; undo any half-drawn selection.
      if (state.drag && state.drag.mode !== "pan") setSel(state.drag.startSel, state.drag.startAuto);
      state.drag = null;
      const [a, b] = [...state.pointers.values()];
      state.pinch = {
        d0: Math.hypot(a.x - b.x, a.y - b.y) || 1,
        mid0: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        scale0: state.view.scale,
        ox0: state.view.ox,
        oy0: state.view.oy,
      };
      return;
    }
    if (state.pointers.size > 1) return;

    const isPan = e.button === 1 || (e.button === 0 && state.spaceDown);
    if (!isPan && e.button !== 0) return;
    e.preventDefault();
    try { el.canvas.setPointerCapture(e.pointerId); } catch (_) { /* pointer already gone */ }

    const base = { pointerId: e.pointerId, startSel: state.sel ? { ...state.sel } : null, startAuto: state.selAuto, startScreen: p };
    if (isPan) {
      state.drag = { ...base, mode: "pan", startView: { ox: state.view.ox, oy: state.view.oy } };
    } else {
      const hit = hitTest(p, e.pointerType);
      if (hit === "inside") {
        state.drag = { ...base, mode: "move", startImg: toImage(p.x, p.y) };
      } else if (hit !== "outside") {
        // Resize: the corner opposite the grabbed one stays put and becomes the anchor.
        const s = state.sel;
        const anchor = {
          x: hit.includes("w") ? s.left + s.size : s.left,
          y: hit.includes("n") ? s.top + s.size : s.top,
        };
        state.drag = { ...base, mode: "resize", corner: hit, anchor };
      } else {
        const q = toImage(p.x, p.y);
        // anchor: {x, y} integer pixel INDICES (the pixel under the press, clamped into the image), not edges
        const anchor = { x: clamp(Math.floor(q.x), 0, state.work.w - 1), y: clamp(Math.floor(q.y), 0, state.work.h - 1) };
        state.drag = { ...base, mode: "draw", anchor };
      }
    }
    updateCursor(p, e.pointerType);
  }

  function onPointerMove(e) {
    const p = eventPoint(e);
    if (state.pointers.has(e.pointerId)) state.pointers.set(e.pointerId, p);
    if (e.pointerType === "mouse" || e.pointerType === "pen") state.hover = p;
    updateCursorReadout(p);

    if (state.pinch && state.pointers.size >= 2) {
      const [a, b] = [...state.pointers.values()];
      const pz = state.pinch;
      const d = Math.hypot(a.x - b.x, a.y - b.y) || 1;
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const next = clamp(pz.scale0 * (d / pz.d0), fitScale(), maxScale());
      const ix = (pz.mid0.x - pz.ox0) / pz.scale0;
      const iy = (pz.mid0.y - pz.oy0) / pz.scale0;
      state.view.scale = next;
      state.view.ox = mid.x - ix * next;
      state.view.oy = mid.y - iy * next;
      state.view.fit = Math.abs(next - fitScale()) < 1e-9;
      clampView();
      syncView();
      return;
    }

    const drag = state.drag;
    if (!drag || drag.pointerId !== e.pointerId) {
      updateCursor(p, e.pointerType);
      return;
    }
    if (drag.mode === "pan") {
      state.view.ox = drag.startView.ox + (p.x - drag.startScreen.x);
      state.view.oy = drag.startView.oy + (p.y - drag.startScreen.y);
      state.view.fit = false;
      clampView();
      syncView();
      return;
    }
    const q = toImage(p.x, p.y);
    if (drag.mode === "move") {
      const s = drag.startSel;
      const dx = Math.round(q.x - drag.startImg.x);
      const dy = Math.round(q.y - drag.startImg.y);
      setSel({
        left: clamp(s.left + dx, 0, state.work.w - s.size),
        top: clamp(s.top + dy, 0, state.work.h - s.size),
        size: s.size,
      });
    } else if (drag.mode === "resize") {
      setSel(squareFromAnchor(drag.anchor.x, drag.anchor.y, q.x, q.y, MIN_BOX));
    } else if (drag.mode === "draw") {
      const moved = Math.hypot(p.x - drag.startScreen.x, p.y - drag.startScreen.y);
      if (moved >= DRAG_CLICK_PX || drag.drawing) {
        drag.drawing = true;
        setSel(squareFromPixels(drag.anchor.x, drag.anchor.y, q.x, q.y));
      }
    }
  }

  // finishDrag: (cancelled: boolean) => void, ends a draw / move / resize / pan; auto-predicts if the square changed
  function finishDrag(cancelled) {
    const drag = state.drag;
    state.drag = null;
    if (!drag) return;
    try { el.canvas.releasePointerCapture(drag.pointerId); } catch (_) { /* already released */ }
    if (drag.mode === "pan") return;
    if (cancelled || (drag.mode === "draw" && !drag.drawing)) {
      setSel(drag.startSel, drag.startAuto); // a click, Esc or a cancelled pointer keeps the previous square
    } else {
      setSel(normalizeSel(state.sel));
    }
    if (!cancelled && !sameSel(state.sel, drag.startSel) && el.autoPredict.checked) predict();
  }

  function onPointerUp(e) {
    state.pointers.delete(e.pointerId);
    if (state.pinch) {
      if (state.pointers.size < 2) state.pinch = null;
      return;
    }
    if (state.drag && state.drag.pointerId === e.pointerId) finishDrag(e.type === "pointercancel");
    updateCursor(eventPoint(e), e.pointerType);
  }

  function onWheel(e) {
    if (!state.work || !el.cameraPanel.hidden) return;
    e.preventDefault();
    const p = eventPoint(e);
    const unit = e.deltaMode === 1 ? 0.05 : e.deltaMode === 2 ? 1 : 0.0015; // lines / pages / pixels
    zoomAt(p.x, p.y, state.view.scale * Math.exp(-e.deltaY * unit));
    updateCursorReadout(p);
  }

  // ---- keyboard ----------------------------------------------------------------------------------

  // ARROWS: Record<key, [dx, dy]>, nudge direction per arrow key
  const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

  function schedulePredictFromKeys() {
    if (!el.autoPredict.checked) return;
    clearTimeout(state.keyTimer);
    state.keyTimer = setTimeout(() => {
      state.keyTimer = null;
      predict();
    }, KEY_PREDICT_DELAY_MS);
  }

  function onStageKeyDown(e) {
    if (e.key === " ") {
      e.preventDefault();
      if (!state.spaceDown) {
        state.spaceDown = true;
        updateCursor(state.hover, "mouse");
      }
      return;
    }
    if (e.key === "Escape") {
      if (state.drag) {
        e.preventDefault();
        finishDrag(true);
        updateCursor(state.hover, "mouse");
      }
      return;
    }
    if (!state.work || !state.sel) return;
    if (e.key === "Enter") {
      e.preventDefault();
      predict();
      return;
    }
    const s = state.sel;
    let next = null;
    if (ARROWS[e.key]) {
      const step = e.shiftKey ? 10 : 1;
      const [dx, dy] = ARROWS[e.key];
      next = normalizeSel({ left: s.left + dx * step, top: s.top + dy * step, size: s.size });
    } else if (e.key === "+" || e.key === "=" || e.code === "NumpadAdd") {
      next = normalizeSel({ left: s.left - 1, top: s.top - 1, size: s.size + 2 });
    } else if (e.key === "-" || e.key === "_" || e.code === "NumpadSubtract") {
      next = normalizeSel({ left: s.left + 1, top: s.top + 1, size: s.size - 2 });
    } else {
      return;
    }
    e.preventDefault();
    if (state.drag) return; // keys do not fight a pointer drag
    if (!sameSel(next, s)) {
      setSel(next);
      schedulePredictFromKeys();
    }
  }

  function onStageKeyUp(e) {
    if (e.key === " ") {
      state.spaceDown = false;
      updateCursor(state.hover, "mouse");
    }
  }

  // ---- image intake ------------------------------------------------------------------------------

  // decodeBlob: (blob: Blob) => Promise<{src: CanvasImageSource, w: number, h: number, close: () => void}>
  // Upright decode: createImageBitmap honours EXIF orientation with imageOrientation "from-image";
  // browsers without that option fall back to an <img>, which applies EXIF orientation by default.
  async function decodeBlob(blob) {
    if (window.createImageBitmap) {
      try {
        const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
        return { src: bmp, w: bmp.width, h: bmp.height, close: () => bmp.close() };
      } catch (_) { /* fall back to <img> */ }
    }
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.decoding = "async";
      img.src = url;
      await img.decode();
      return { src: img, w: img.naturalWidth, h: img.naturalHeight, close: () => {} };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // loadBlob: (blob: Blob, meta: {name: string, source: string, sample?: object}) => Promise<void>
  async function loadBlob(blob, meta) {
    clearError();
    showNotice("");
    if (blob.type && !blob.type.startsWith("image/")) {
      showError(`“${meta.name}” is not an image (${blob.type}). Choose a PNG, JPEG, WebP, GIF, BMP or other image file.`);
      return;
    }
    if (blob.size === 0) {
      showError(`“${meta.name}” is empty.`);
      return;
    }
    let decoded;
    try {
      decoded = await decodeBlob(blob);
    } catch (_) {
      showError(`“${meta.name}” is not an image the browser can decode.`);
      return;
    }
    if (decoded.w * decoded.h > MAX_PIXELS) {
      decoded.close();
      showError(`“${meta.name}” is ${decoded.w}×${decoded.h} (${(decoded.w * decoded.h / 1e6).toFixed(1)} MP); the limit is 40 MP.`);
      return;
    }
    setWorkingImage(decoded.src, decoded.w, decoded.h, meta);
    decoded.close();
  }

  // setWorkingImage: (src: CanvasImageSource, w, h: number natural px, meta) => void
  // Copies the source into an opaque working canvas (transparent pixels over white), preselects the
  // largest centred square, fits the view and, with auto-predict on, predicts right away.
  function setWorkingImage(src, w, h, meta) {
    let cw = w;
    let ch = h;
    if (w * h > WORK_CAP_PIXELS) {
      const k = Math.sqrt(WORK_CAP_PIXELS / (w * h));
      cw = Math.max(1, Math.floor(w * k));
      ch = Math.max(1, Math.floor(h * k));
    }
    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const c = canvas.getContext("2d");
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, cw, ch);
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = "high";
    c.drawImage(src, 0, 0, cw, ch);

    if (state.controller) state.controller.abort();
    state.controller = null;
    clearTimeout(state.keyTimer);
    state.work = { canvas, w: cw, h: ch, name: meta.name, source: meta.source, sample: meta.sample || null };
    state.result = null;
    state.heatKey = "";
    el.stage.classList.add("has-image");
    el.stageEmpty.hidden = true;
    el.stage.dataset.naturalW = String(cw);
    el.stage.dataset.naturalH = String(ch);
    el.stage.dataset.source = meta.source;
    el.statusImage.textContent = `${cw}×${ch} px`;
    el.predictButton.disabled = false;
    if (cw !== w || ch !== h) {
      showNotice(`The ${w}×${h} image was downscaled to ${cw}×${ch} for the working canvas; coordinates refer to the working copy.`);
    }
    if (Math.min(cw, ch) < MIN_BOX) {
      setSel(null);
      fitView();
      resetReadout();
      showError(`The image is only ${cw}×${ch} px; the square must be at least ${MIN_BOX} px.`);
      return;
    }
    fitView();
    setSel(defaultSel(), true);
    resetReadout();
    announce(`Opened ${meta.name} (${cw}×${ch} px). The largest centred square is selected.`);
    setAppState("idle");
    if (el.autoPredict.checked) predict();
  }

  function onFileChosen(file, source) {
    if (!file) return;
    const name = file.name || "pasted image";
    loadBlob(file, { name, source });
  }

  // ---- prediction --------------------------------------------------------------------------------

  // decodeHeatmap: (heatmap: {width, height, dtype, encoding, data: base64}) => Float32Array (width * height)
  function decodeHeatmap(heatmap) {
    const bin = atob(heatmap.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    const view = new DataView(bytes.buffer);
    const n = heatmap.width * heatmap.height;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = view.getFloat32(i * 4, true); // base64-le: little-endian float32
    return out;
  }

  // loadImageUrl: (url: string) => Promise<HTMLImageElement>, resolves once decoded
  async function loadImageUrl(url) {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  }

  function setBusy(busy) {
    el.predictButton.classList.toggle("is-busy", busy);
    el.predictButton.querySelector(".btn-label").textContent = busy ? "Predicting…" : "Predict";
    el.readout.setAttribute("aria-busy", String(busy));
  }

  // predict: () => Promise<void>, crops the current square (WYSIWYG), uploads it, shows the result
  async function predict() {
    if (!state.work || !state.sel || !el.cameraPanel.hidden) return;
    clearTimeout(state.keyTimer);
    state.keyTimer = null;
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    // s: {left, top, size} image px, the square this request is about (frozen now)
    const s = { ...state.sel };
    const seq = ++state.seq;
    clearError();
    setBusy(true);
    setAppState("loading");
    announce(`Predicting in ${s.left},${s.top} · ${s.size}×${s.size} px…`);

    try {
      // side: number, uploaded crop side; > 1024 px squares are downsized (high-quality smoothing)
      const side = Math.min(s.size, MAX_UPLOAD_SIDE);
      const crop = document.createElement("canvas");
      crop.width = side;
      crop.height = side;
      const c = crop.getContext("2d");
      c.imageSmoothingEnabled = true;
      c.imageSmoothingQuality = "high";
      c.drawImage(state.work.canvas, s.left, s.top, s.size, s.size, 0, 0, side, side);
      const blob = await new Promise((resolve, reject) => {
        crop.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the selection as PNG."))), "image/png");
      });

      const body = new FormData();
      body.append("file", blob, "selection.png");
      body.append("box", `0,0,${side}`);
      const t0 = performance.now();
      const res = await fetch("/api/predict", { method: "POST", body, signal: controller.signal });
      if (!res.ok) {
        let detail = `The server could not run the model on this square (HTTP ${res.status}).`;
        try {
          const json = await res.json();
          if (typeof json.detail === "string") detail = json.detail;
        } catch (_) { /* non-JSON error body */ }
        throw new Error(detail);
      }
      const json = await res.json();
      const roundTrip = performance.now() - t0;
      const modelImg = await loadImageUrl(json.model_input_png);
      if (controller !== state.controller) return; // superseded by a newer request
      handleResult(json, s, side, seq, modelImg, roundTrip);
    } catch (err) {
      if (err.name === "AbortError" || controller !== state.controller) return;
      announce("Prediction failed.");
      showError(err.message || "Network error: could not reach the server.");
    } finally {
      if (controller === state.controller) {
        state.controller = null;
        setBusy(false);
      }
    }
  }

  // handleResult: maps the server's crop-px answer back to image px and stores it as state.result:
  //   {seq: number, box: {left, top, size} image px, side: number uploaded px,
  //    tip: {x, y} image px, tipModel: {x, y} model px, peakCell: {x, y} heatmap cell,
  //    confidence: number, peak: number, candidates: Array<{x, y image px, xModel, yModel, mass}>,
  //    runnerRatio: number | null, ambiguous: boolean, probs: Float32Array (128²), modelImg: HTMLImageElement,
  //    timing: {preprocess, inference, total} ms, roundTrip: number ms}
  function handleResult(json, s, side, seq, modelImg, roundTrip) {
    const k = s.size / side; // image px per uploaded px
    const toImg = (x, y) => ({ x: s.left + (x + 0.5) * k - 0.5, y: s.top + (y + 0.5) * k - 0.5 });
    const candidates = json.candidates.map((c) => ({ ...toImg(c.x, c.y), xModel: c.x_model, yModel: c.y_model, mass: c.mass }));
    const winner = candidates[0];
    const runner = candidates[1] || null;
    const runnerRatio = runner && winner && winner.mass > 0 ? runner.mass / winner.mass : null;
    const ambiguous = (runnerRatio !== null && runnerRatio > AMBIGUOUS_RUNNER_RATIO) || json.confidence < AMBIGUOUS_MIN_CONFIDENCE;
    state.result = {
      seq,
      box: s,
      side,
      tip: toImg(json.tip.x, json.tip.y),
      tipModel: json.tip_model,
      peakCell: json.peak_cell,
      confidence: json.confidence,
      peak: json.peak_prob,
      candidates,
      runnerRatio,
      ambiguous,
      probs: decodeHeatmap(json.heatmap),
      modelImg,
      timing: json.timing_ms,
      roundTrip,
    };
    state.heatKey = "";
    renderReadout();
    drawCues();
    scheduleDraw();
    setAppState("done");
  }

  // ---- readout -----------------------------------------------------------------------------------

  // gtInfo: () => null | {gt: {x, y} image px, error: number px, budget: number px}
  function gtInfo() {
    const r = state.result;
    const sample = state.work && state.work.sample;
    if (!r || !sample) return null;
    const gt = sample.label;
    return { gt, error: Math.hypot(r.tip.x - gt.x, r.tip.y - gt.y), budget: (BUDGET_MODEL_PX * r.box.size) / MODEL_SIZE };
  }

  // READOUT_KEYS: string[], the #readout data-* keys (camelCase) a prediction writes; cleared on reset
  const READOUT_KEYS = ["seq", "tipX", "tipY", "tipModelX", "tipModelY", "confidence", "peakProb", "runnerUp",
    "ambiguous", "boxLeft", "boxTop", "boxSize", "candidates", "gtX", "gtY", "errorPx"];

  function resetReadout() {
    const d = el.readout.dataset;
    READOUT_KEYS.forEach((key) => delete d[key]);
    d.state = "empty";
    for (const node of [el.outTip, el.outTipModel, el.outConfidence, el.outPeak, el.outRunner, el.outBox, el.outGt, el.outError, el.outTiming]) {
      node.textContent = "–";
      node.className = "";
    }
    el.gtRows.forEach((row) => { row.hidden = !(state.work && state.work.sample); });
    if (state.work && state.work.sample) el.outGt.textContent = `${f1(state.work.sample.label.x)}, ${f1(state.work.sample.label.y)}`;
    el.candidates.replaceChildren();
    el.verdict.hidden = true;
    drawCues();
  }

  function renderReadout() {
    const r = state.result;
    if (!r) return resetReadout();
    const d = el.readout.dataset;
    d.state = "done";
    d.seq = String(r.seq);
    d.tipX = String(r.tip.x);
    d.tipY = String(r.tip.y);
    d.tipModelX = String(r.tipModel.x);
    d.tipModelY = String(r.tipModel.y);
    d.confidence = String(r.confidence);
    d.peakProb = String(r.peak);
    d.runnerUp = String(r.candidates[1] ? r.candidates[1].mass : 0);
    d.ambiguous = String(r.ambiguous);
    d.boxLeft = String(r.box.left);
    d.boxTop = String(r.box.top);
    d.boxSize = String(r.box.size);
    d.candidates = String(r.candidates.length);

    el.outTip.textContent = `${f1(r.tip.x)}, ${f1(r.tip.y)}`;
    el.outTipModel.textContent = `${f1(r.tipModel.x)}, ${f1(r.tipModel.y)}`;
    el.outConfidence.textContent = pct(r.confidence);
    el.outConfidence.className = r.confidence < AMBIGUOUS_MIN_CONFIDENCE ? "bad" : "";
    el.outPeak.textContent = f3(r.peak);
    el.outRunner.textContent = r.candidates[1]
      ? `${pct(r.candidates[1].mass)} (${f2(r.runnerRatio)}× winner)`
      : "none";
    el.outRunner.className = r.runnerRatio !== null && r.runnerRatio > AMBIGUOUS_RUNNER_RATIO ? "bad" : "";
    el.outBox.textContent = `${r.box.left},${r.box.top} · ${r.box.size}`;
    const t = r.timing;
    el.outTiming.textContent =
      `preprocess ${f1(t.preprocess)} · inference ${f1(t.inference)} · server ${f1(t.total)} · round trip ${f1(r.roundTrip)} ms`;

    const g = gtInfo();
    el.gtRows.forEach((row) => { row.hidden = !g; });
    if (g) {
      d.gtX = String(g.gt.x);
      d.gtY = String(g.gt.y);
      d.errorPx = String(g.error);
      el.outGt.textContent = `${f1(g.gt.x)}, ${f1(g.gt.y)}`;
      const within = g.error <= g.budget;
      el.outError.textContent = `${f2(g.error)} px ${within ? "✓ within" : "✗ over"} ${f1(g.budget)} px`;
      el.outError.className = within ? "good" : "bad";
    } else {
      delete d.gtX;
      delete d.gtY;
      delete d.errorPx;
    }

    // Verdict chip: "ambiguous" when a look-alike is strong or the decode window holds < 50 % of the mass.
    el.verdict.hidden = false;
    if (r.ambiguous) {
      el.verdict.className = "chip chip-warn";
      el.verdict.textContent = r.confidence < AMBIGUOUS_MIN_CONFIDENCE
        ? `⚠ Ambiguous · confidence ${pct(r.confidence, 0)}`
        : `⚠ Ambiguous · look-alike at ${pct(r.runnerRatio, 0)} of winner`;
    } else {
      el.verdict.className = "chip chip-ok";
      el.verdict.textContent = "✓ Clear peak";
    }
    el.verdict.dataset.ambiguous = String(r.ambiguous);

    // Candidate list: one bar per peak, the winner highlighted.
    const maxMass = Math.max(...r.candidates.map((c) => c.mass), 1e-9);
    el.candidates.replaceChildren(...r.candidates.map((c, i) => {
      const li = document.createElement("li");
      li.dataset.mass = String(c.mass);
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = i === 0 ? "tip" : `#${i + 1}`;
      const bar = document.createElement("span");
      bar.className = "bar";
      bar.title = `(${f1(c.x)}, ${f1(c.y)}) image px · (${f1(c.xModel)}, ${f1(c.yModel)}) model px`;
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(2, (c.mass / maxMass) * 100)}%`;
      bar.append(fill);
      const mass = document.createElement("span");
      mass.className = "mass";
      mass.textContent = pct(c.mass);
      li.append(rank, bar, mass);
      return li;
    }));

    const errText = g ? ` · error ${f2(g.error)} px vs ground truth` : "";
    announce(`Tip at ${f1(r.tip.x)}, ${f1(r.tip.y)} px · confidence ${pct(r.confidence)}${r.ambiguous ? " · ambiguous" : ""}${errText}.`);
    markStale();
  }

  // markStale: () => void, flags the readout when the square no longer matches the one predicted
  function markStale() {
    const r = state.result;
    el.readout.dataset.stale = String(!!r && !sameSel(r.box, state.sel));
  }

  // ---- heatmap image -----------------------------------------------------------------------------

  // heatValue: (p: number probability, peak: number) => number t in [0, 1] for the current scale
  function heatValue(p, peak) {
    if (state.heat.mode === "log") return clamp((Math.log10(Math.max(p, 1e-30)) - LOG_FLOOR) / -LOG_FLOOR, 0, 1);
    return clamp(p / (peak || 1), 0, 1);
  }

  // heatAlpha: (t: number 0..1) => number 0..1, opacity rising with value so the photo stays visible
  // (log: nearly linear in log10 p, so 1e-5 .. 1e-3 tails show; linear: a gentle lift of the faint shoulder)
  function heatAlpha(t) {
    return state.heat.opacity * Math.pow(t, state.heat.mode === "log" ? 0.85 : 0.6);
  }

  // heatCanvas: () => HTMLCanvasElement 128 × 128 RGBA of the current result, rebuilt only when inputs change
  function heatCanvas() {
    const r = state.result;
    const key = `${r.seq}|${state.heat.mode}|${state.heat.opacity}`;
    if (key === state.heatKey) return state.heatCanvas;
    const c = state.heatCanvas.getContext("2d");
    const img = c.createImageData(HEAT_SIZE, HEAT_SIZE);
    for (let i = 0; i < r.probs.length; i += 1) {
      const t = heatValue(r.probs[i], r.peak);
      const li = Math.round(t * 255) * 3;
      img.data[i * 4] = LUT[li];
      img.data[i * 4 + 1] = LUT[li + 1];
      img.data[i * 4 + 2] = LUT[li + 2];
      img.data[i * 4 + 3] = Math.round(heatAlpha(t) * 255);
    }
    c.putImageData(img, 0, 0);
    state.heatKey = key;
    return state.heatCanvas;
  }

  function drawLegend() {
    const c = el.legend.getContext("2d");
    const w = el.legend.width;
    const h = el.legend.height;
    c.clearRect(0, 0, w, h);
    for (let x = 0; x < w; x += 1) {
      const t = x / (w - 1);
      const li = Math.round(t * 255) * 3;
      c.fillStyle = `rgba(${LUT[li]},${LUT[li + 1]},${LUT[li + 2]},${Math.max(heatAlpha(t), 0)})`;
      c.fillRect(x, 0, 1, h);
    }
    const peak = state.result ? f3(state.result.peak) : "peak";
    const ticks = state.heat.mode === "log" ? ["10⁻⁶", "10⁻⁴", "10⁻²", "1"] : ["0", "½ peak", `peak ${peak}`];
    el.legendTicks.replaceChildren(...ticks.map((t) => Object.assign(document.createElement("span"), { textContent: t })));
    el.scaleHelp.textContent = state.heat.mode === "log"
      ? "Log: log₁₀ p clipped to [−6, 0]. Secondary hypotheses (shadows, decoys, reflections) stay visible."
      : "Linear: p ÷ peak. Honest but usually a single dot, because the model is very sure.";
  }

  function syncHeatData() {
    el.stage.dataset.heatmap = state.heat.on ? "on" : "off";
    el.stage.dataset.heatmapScale = state.heat.mode;
    el.stage.dataset.heatmapOpacity = String(state.heat.opacity);
  }

  // ---- drawing: shared markers -------------------------------------------------------------------

  // haloText: (c: ctx, text: string, x, y: number, colour: string, size: number px) => void, text with a dark halo
  function haloText(c, text, x, y, colour, size) {
    c.font = `600 ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    c.textBaseline = "middle";
    c.lineJoin = "round";
    c.lineWidth = Math.max(3, size / 3.5);
    c.strokeStyle = "rgba(0,0,0,0.85)";
    c.strokeText(text, x, y);
    c.fillStyle = colour;
    c.fillText(text, x, y);
  }

  // strokeTwice: (c, draw: () => void, colour: string, width: number) => void, a dark outline under a coloured stroke
  function strokeTwice(c, draw, colour, width) {
    c.beginPath();
    draw();
    c.lineWidth = width + 2.5;
    c.strokeStyle = "rgba(0,0,0,0.8)";
    c.stroke();
    c.lineWidth = width;
    c.strokeStyle = colour;
    c.stroke();
  }

  // drawTipMarker: crosshair with a gap at the tip, plus the error-budget ring (radius in canvas px)
  function drawTipMarker(c, x, y, ringR, unit) {
    const colour = cssVar("--tip");
    const gap = Math.max(3 * unit, Math.min(ringR * 0.45, 8 * unit));
    const arm = ringR + 10 * unit;
    strokeTwice(c, () => {
      c.moveTo(x - arm, y); c.lineTo(x - gap, y);
      c.moveTo(x + gap, y); c.lineTo(x + arm, y);
      c.moveTo(x, y - arm); c.lineTo(x, y - gap);
      c.moveTo(x, y + gap); c.lineTo(x, y + arm);
    }, colour, 1.6 * unit);
    strokeTwice(c, () => c.arc(x, y, ringR, 0, Math.PI * 2), colour, 1.4 * unit);
    c.beginPath();
    c.arc(x, y, 1.3 * unit, 0, Math.PI * 2);
    c.fillStyle = colour;
    c.fill();
  }

  function drawCandidateMarker(c, x, y, label, unit) {
    const r = 8 * unit;
    strokeTwice(c, () => c.arc(x, y, r, 0, Math.PI * 2), cssVar("--cand"), 1.6 * unit);
    haloText(c, label, x + r + 4 * unit, y - r * 0.6, cssVar("--cand"), 12 * unit);
  }

  function drawGtMarker(c, x, y, unit) {
    const r = 7 * unit;
    strokeTwice(c, () => {
      c.moveTo(x, y - r); c.lineTo(x + r, y); c.lineTo(x, y + r); c.lineTo(x - r, y); c.closePath();
    }, cssVar("--gt"), 1.8 * unit);
    haloText(c, "GT", x + r + 3 * unit, y + r * 0.9, cssVar("--gt"), 11 * unit);
  }

  // cssVarCache: Map<string, string>, resolved custom-property colours (re-read when the theme changes)
  const cssVarCache = new Map();
  function cssVar(name) {
    if (!cssVarCache.has(name)) cssVarCache.set(name, getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#fff");
    return cssVarCache.get(name);
  }

  // drawResultMarkers: (c, map: (X, Y) => {x, y} canvas px, pxPerImagePx: number, unit: number) => void
  // Candidates (hollow, with mass %), ground truth, then the tip on top.
  function drawResultMarkers(c, map, pxPerImagePx, unit) {
    const r = state.result;
    r.candidates.slice(1).forEach((cand) => {
      const p = map(cand.x, cand.y);
      drawCandidateMarker(c, p.x, p.y, pct(cand.mass, cand.mass < 0.1 ? 1 : 0), unit);
    });
    const g = gtInfo();
    if (g) {
      const p = map(g.gt.x, g.gt.y);
      drawGtMarker(c, p.x, p.y, unit);
    }
    const tip = map(r.tip.x, r.tip.y);
    const ringR = Math.max(4 * unit, BUDGET_MODEL_PX * (r.box.size / MODEL_SIZE) * pxPerImagePx);
    drawTipMarker(c, tip.x, tip.y, ringR, unit);
    haloText(c, pct(r.candidates[0].mass, 0), tip.x + ringR + 5 * unit, tip.y - ringR * 0.7 - 6 * unit, cssVar("--tip"), 12 * unit);
  }

  // ---- drawing: stage ----------------------------------------------------------------------------

  // drawPending: boolean, a requestAnimationFrame redraw is queued
  let drawPending = false;
  function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => {
      drawPending = false;
      drawStage();
    });
  }

  function drawStage() {
    const dpr = state.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    const work = state.work;
    if (!work) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const v = state.view;
    const iw = work.w * v.scale;
    const ih = work.h * v.scale;

    // 1. the working image (pixelated when zoomed in, like Paint)
    ctx.imageSmoothingEnabled = v.scale < 2;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(work.canvas, v.ox, v.oy, iw, ih);

    // 2. heatmap over the square that was predicted (128 cells tile it exactly)
    const r = state.result;
    if (r && state.heat.on) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(heatCanvas(), edgeToScreen(r.box.left, "x"), edgeToScreen(r.box.top, "y"), r.box.size * v.scale, r.box.size * v.scale);
    }

    // 3. dim everything outside the selection
    const s = state.sel;
    let sr = null;
    if (s) {
      sr = selScreenRect();
      ctx.beginPath();
      ctx.rect(v.ox, v.oy, iw, ih);
      ctx.rect(sr.x0, sr.y0, sr.x1 - sr.x0, sr.y1 - sr.y0);
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fill("evenodd");
    }

    // 4. cues
    if (r) drawResultMarkers(ctx, (X, Y) => centreToScreen(X, Y), v.scale, 1);

    // 5. marching-ants marquee and 4 corner handles
    if (sr) {
      const px = 0.5 / dpr; // half a device pixel keeps 1-px lines crisp
      const x = Math.round(sr.x0 * dpr) / dpr + px;
      const y = Math.round(sr.y0 * dpr) / dpr + px;
      const w = Math.round((sr.x1 - sr.x0) * dpr) / dpr - 2 * px;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.strokeStyle = "#ffffff";
      ctx.strokeRect(x, y, w, w);
      ctx.setLineDash([5, 5]);
      ctx.lineDashOffset = -state.ants;
      ctx.strokeStyle = "#000000";
      ctx.strokeRect(x, y, w, w);
      ctx.setLineDash([]);
      const hs = coarsePointer.matches ? 12 : 8;
      for (const [cx, cy] of [[sr.x0, sr.y0], [sr.x1, sr.y0], [sr.x0, sr.y1], [sr.x1, sr.y1]]) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - hs / 2 + px, cy - hs / 2 + px, hs - 2 * px, hs - 2 * px);
      }
    }
  }

  // ---- drawing: model view and tip zoom ------------------------------------------------------------

  function placeholder(c, canvas, text) {
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.fillStyle = "#8b93a1";
    c.font = `500 ${Math.round(canvas.width / 22)}px system-ui, sans-serif`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillText(text, canvas.width / 2, canvas.height / 2);
    c.textAlign = "start";
  }

  // drawModelView: the 256 × 256 gray input at 2 canvas px per model px, heatmap, markers, zoom window outline
  function drawModelView() {
    const c = modelCtx;
    const canvas = el.modelCanvas;
    const r = state.result;
    if (!r) return placeholder(c, canvas, "No prediction yet");
    const k = canvas.width / MODEL_SIZE; // canvas px per model px (2)
    // unit: number, canvas px per CSS px, so markers look the same size as on the stage
    const unit = canvas.width / Math.max(1, canvas.clientWidth || canvas.width);
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.imageSmoothingEnabled = false;
    c.drawImage(r.modelImg, 0, 0, canvas.width, canvas.height);
    if (state.heat.on) {
      c.imageSmoothingEnabled = true;
      c.drawImage(heatCanvas(), 0, 0, canvas.width, canvas.height);
    }
    // image px -> model px -> canvas px (centre convention both ways)
    const toModel = (X, Y) => ({
      x: (X - r.box.left + 0.5) * (MODEL_SIZE / r.box.size) - 0.5,
      y: (Y - r.box.top + 0.5) * (MODEL_SIZE / r.box.size) - 0.5,
    });
    const map = (X, Y) => {
      const m = toModel(X, Y);
      return { x: (m.x + 0.5) * k, y: (m.y + 0.5) * k };
    };
    // the tip-zoom window, dashed
    const zx = (r.tipModel.x + 0.5 - ZOOM_WIN / 2) * k;
    const zy = (r.tipModel.y + 0.5 - ZOOM_WIN / 2) * k;
    c.setLineDash([4 * unit, 4 * unit]);
    c.lineWidth = 1 * unit;
    c.strokeStyle = "rgba(255,255,255,0.55)";
    c.strokeRect(zx, zy, ZOOM_WIN * k, ZOOM_WIN * k);
    c.setLineDash([]);
    drawResultMarkers(c, map, k * (MODEL_SIZE / r.box.size), unit);
  }

  // drawZoom: 48 × 48 model px around the tip at ×5, nearest-neighbour, with the 5 × 5 decode window
  function drawZoom() {
    const c = zoomCtx;
    const canvas = el.zoomCanvas;
    const r = state.result;
    if (!r) return placeholder(c, canvas, "Tip zoom");
    const unit = canvas.width / Math.max(1, canvas.clientWidth || canvas.width);
    // x0 / y0: number, model EDGE coordinate of the window's top-left (the tip sits in the middle)
    const x0 = r.tipModel.x + 0.5 - ZOOM_WIN / 2;
    const y0 = r.tipModel.y + 0.5 - ZOOM_WIN / 2;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = "#0a0b0e";
    c.fillRect(0, 0, canvas.width, canvas.height);
    c.setTransform(ZOOM_K, 0, 0, ZOOM_K, -x0 * ZOOM_K, -y0 * ZOOM_K);
    c.imageSmoothingEnabled = false;
    c.drawImage(r.modelImg, 0, 0, MODEL_SIZE, MODEL_SIZE);
    if (state.heat.on) c.drawImage(heatCanvas(), 0, 0, MODEL_SIZE, MODEL_SIZE);
    c.setTransform(1, 0, 0, 1, 0, 0);
    const toCanvas = (edgeX, edgeY) => ({ x: (edgeX - x0) * ZOOM_K, y: (edgeY - y0) * ZOOM_K });

    // decode window: heatmap cells peak ± 2 -> model edges [2 (ix − 2), 2 (ix + 3)] (10 model px)
    const a = toCanvas(2 * (r.peakCell.x - 2), 2 * (r.peakCell.y - 2));
    const cell = 2 * ZOOM_K; // canvas px per heatmap cell
    c.lineWidth = 1;
    c.strokeStyle = "rgba(255,255,255,0.28)";
    c.beginPath();
    for (let i = 1; i < 5; i += 1) {
      c.moveTo(a.x + i * cell + 0.5, a.y); c.lineTo(a.x + i * cell + 0.5, a.y + 5 * cell);
      c.moveTo(a.x, a.y + i * cell + 0.5); c.lineTo(a.x + 5 * cell, a.y + i * cell + 0.5);
    }
    c.stroke();
    c.setLineDash([5, 4]);
    strokeTwice(c, () => c.rect(a.x + 0.5, a.y + 0.5, 5 * cell, 5 * cell), cssVar("--tip"), 1.2);
    c.setLineDash([]);

    // ground truth and tip, in model px
    const g = gtInfo();
    if (g) {
      const gx = (g.gt.x - r.box.left + 0.5) * (MODEL_SIZE / r.box.size); // model EDGE coords of the GT centre
      const gy = (g.gt.y - r.box.top + 0.5) * (MODEL_SIZE / r.box.size);
      const p = toCanvas(gx, gy);
      if (p.x > -10 && p.y > -10 && p.x < canvas.width + 10 && p.y < canvas.height + 10) drawGtMarker(c, p.x, p.y, unit);
    }
    const t = toCanvas(r.tipModel.x + 0.5, r.tipModel.y + 0.5);
    drawTipMarker(c, t.x, t.y, BUDGET_MODEL_PX * ZOOM_K, unit * 0.9);
  }

  function drawCues() {
    drawLegend();
    drawModelView();
    drawZoom();
  }

  // ---- camera ------------------------------------------------------------------------------------

  const PHONE_HINT = " On a phone, “Open image…” can also open the camera app directly.";

  // cameraErrorMessage: (err: DOMException | Error) => string, a friendly explanation per failure
  function cameraErrorMessage(err) {
    const name = err && err.name;
    const hint = coarsePointer.matches ? PHONE_HINT : "";
    if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") {
      return `Camera permission was denied. Allow camera access for this site (the camera icon in the address bar), then press Camera again.${hint}`;
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError" || name === "OverconstrainedError") {
      return `No camera was found on this device.${hint}`;
    }
    if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") {
      return `The camera could not start; another application may be using it.${hint}`;
    }
    return `Could not start the camera (${(err && err.message) || name || "unknown error"}).${hint}`;
  }

  function insecureMessage() {
    const port = location.port ? `:${location.port}` : "";
    return `The camera needs a secure context, and this page is plain http on ${location.hostname}, which browsers do not trust ` +
      `with cameras. Open it as http://localhost${port} on the computer running the server, or serve it over https.` +
      (coarsePointer.matches ? PHONE_HINT : " On a phone, “Open image…” can open the camera app instead.");
  }

  // stopCamera: () => void, stops EVERY track so the camera light goes off, hides the preview
  function stopCamera() {
    const stream = state.camera.stream;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    state.camera.stream = null;
    el.video.pause();
    el.video.srcObject = null;
    el.cameraPanel.hidden = true;
    el.cameraPanel.dataset.state = "off";
    el.captureButton.disabled = true;
    el.cameraButton.setAttribute("aria-pressed", "false");
    el.stageEmpty.hidden = !!state.work;
  }

  // listCameras: () => Promise<void>, fills the camera picker when more than one camera exists
  async function listCameras() {
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      el.cameraSelect.hidden = devices.length < 2;
      if (devices.length < 2) return;
      const current = state.camera.stream && state.camera.stream.getVideoTracks()[0];
      const currentId = current && current.getSettings ? current.getSettings().deviceId : null;
      el.cameraSelect.replaceChildren(...devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
      if (currentId) el.cameraSelect.value = currentId;
    } catch (_) {
      el.cameraSelect.hidden = true;
    }
  }

  // openCamera: (deviceId?: string) => Promise<void>, starts the live preview inline in the stage
  async function openCamera(deviceId) {
    clearError();
    showNotice("");
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError(insecureMessage());
      return;
    }
    if (state.controller) state.controller.abort();
    if (state.camera.stream) stopCamera();
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } };
    el.cameraPanel.hidden = false;
    el.cameraPanel.dataset.state = "starting";
    el.stageEmpty.hidden = true;
    el.cameraButton.setAttribute("aria-pressed", "true");
    announce("Starting the camera…");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      stopCamera();
      showError(cameraErrorMessage(err));
      announce("The camera did not start.");
      return;
    }
    if (el.cameraPanel.hidden) { // cancelled while the permission prompt was open
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    state.camera.stream = stream;
    el.video.srcObject = stream;
    try {
      await el.video.play();
    } catch (_) { /* autoplay is allowed for muted video; ignore odd failures */ }
    await listCameras();
  }

  function onVideoPlaying() {
    if (!state.camera.stream) return;
    el.cameraPanel.dataset.state = "live";
    el.cameraPanel.dataset.videoW = String(el.video.videoWidth);
    el.cameraPanel.dataset.videoH = String(el.video.videoHeight);
    el.captureButton.disabled = false;
    announce(`Camera live (${el.video.videoWidth}×${el.video.videoHeight}). Press Capture to grab one frame.`);
    el.captureButton.focus({ preventScroll: true });
  }

  // capture: () => void, copies ONE frame into the working canvas, then stops the camera
  function capture() {
    const v = el.video;
    if (!state.camera.stream || !v.videoWidth || !v.videoHeight) return;
    const frame = document.createElement("canvas");
    frame.width = v.videoWidth;
    frame.height = v.videoHeight;
    frame.getContext("2d").drawImage(v, 0, 0, frame.width, frame.height);
    stopCamera();
    setWorkingImage(frame, frame.width, frame.height, { name: "camera frame", source: "camera" });
    showNotice(`Captured one ${frame.width}×${frame.height} frame; the camera is off again.`);
    el.stage.focus({ preventScroll: true });
  }

  // ---- samples and model card --------------------------------------------------------------------

  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.samples = data.samples || [];
      el.samplesGrid.replaceChildren(...state.samples.map((s) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "sample-btn" + (/hard|look-alike/i.test(s.note || "") ? " is-hard" : "");
        b.dataset.testid = `sample-${s.id}`;
        b.setAttribute("aria-label", `Sample ${s.id}: ${s.note || ""}`);
        const img = Object.assign(document.createElement("img"), { src: s.url, alt: "", loading: "lazy", width: 96, height: 96 });
        const label = document.createElement("span");
        label.textContent = `Sample ${s.id}`;
        const note = document.createElement("span");
        note.className = "sample-note";
        note.textContent = /hard|look-alike/i.test(s.note || "") ? "hardest (fails)" : "validation";
        b.append(img, label, note);
        b.addEventListener("click", () => openSample(s));
        return b;
      }));
    } catch (_) {
      el.samplesGrid.textContent = "Samples could not be loaded.";
    }
  }

  async function openSample(s) {
    toggleSamples(false);
    try {
      const res = await fetch(s.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      if (state.camera.stream) stopCamera();
      await loadBlob(blob, { name: `Sample ${s.id} (${s.file})`, source: "sample", sample: s });
    } catch (err) {
      showError(`Could not load sample ${s.id}: ${err.message}`);
    }
  }

  function toggleSamples(open) {
    const next = open === undefined ? el.samplesMenu.hidden : open;
    el.samplesMenu.hidden = !next;
    el.samplesButton.setAttribute("aria-expanded", String(next));
    if (next) {
      const first = el.samplesGrid.querySelector("button");
      if (first) first.focus({ preventScroll: true });
    }
  }

  async function loadModelInfo() {
    try {
      const res = await fetch("/api/model");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const m = await res.json();
      const val = m.validation || {};
      const kb = Math.round((m.file_bytes || 0) / 1024);
      el.badge.replaceChildren();
      const name = document.createElement("strong");
      name.textContent = m.name || m.served_file;
      el.badge.append(name, ` · ${kb} KB · val p99 ${val.p99_px !== undefined ? f2(val.p99_px) : "?"} px`);
      el.badge.title = `${m.served_file} · ${m.interpreter ? m.interpreter.runtime : ""}`;
      el.badge.dataset.fileBytes = String(m.file_bytes);
      const io = m.interpreter || {};
      // shape: (s: number[] | undefined) => string such as "[1, 256, 256, 1]"
      const shape = (s) => (Array.isArray(s) ? `[${s.join(", ")}]` : "?");
      const rows = [
        ["Model", `${m.name || "?"} · ${m.served_file} · ${kb} KB · ${m.params ? m.params.toLocaleString() : "?"} params`],
        ["Runtime", `${io.runtime || "?"} · ${io.num_threads || "?"} threads`],
        ["Input", `${shape(io.input_shape)} ${io.input_dtype || ""} · ${(m.input && m.input.range) || ""} · ${(m.input && m.input.layout) || ""}`],
        ["Output", `${shape(io.output_shape)} ${io.output_dtype || ""} · ${(m.output && m.output.meaning) || ""}`],
        ["Decode", m.decode || "?"],
        ["Validation", val.split ? `${val.split}: mean ${val.mean_px} px · median ${val.median_px} px · p95 ${val.p95_px} px · p99 ${val.p99_px} px · max ${val.max_px} px · ${val.n_over_7px} images over 7 px` : "?"],
        ["Training data", m.train_data || "?"],
        ["Source", m.source || "?"],
      ];
      el.aboutList.replaceChildren(...rows.map(([k, v]) => {
        const div = document.createElement("div");
        div.append(Object.assign(document.createElement("dt"), { textContent: k }), Object.assign(document.createElement("dd"), { textContent: v }));
        return div;
      }));
    } catch (_) {
      el.badge.textContent = "Model information unavailable";
    }
  }

  // ---- wiring ------------------------------------------------------------------------------------

  el.openButton.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    const file = el.fileInput.files[0];
    el.fileInput.value = ""; // allow choosing the same file again
    if (state.camera.stream) stopCamera();
    onFileChosen(file, "file");
  });

  el.cameraButton.addEventListener("click", () => {
    if (state.camera.stream || !el.cameraPanel.hidden) stopCamera();
    else openCamera();
  });
  el.cameraCancel.addEventListener("click", () => {
    stopCamera();
    announce("Camera cancelled.");
  });
  el.captureButton.addEventListener("click", capture);
  el.cameraSelect.addEventListener("change", () => openCamera(el.cameraSelect.value));
  el.video.addEventListener("playing", onVideoPlaying);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.camera.stream) stopCamera(); // never leave the camera on in a background tab
  });

  el.samplesButton.addEventListener("click", () => toggleSamples());
  document.addEventListener("pointerdown", (e) => {
    if (!el.samplesMenu.hidden && !el.samplesMenu.contains(e.target) && !el.samplesButton.contains(e.target)) toggleSamples(false);
  });
  el.samplesMenu.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      toggleSamples(false);
      el.samplesButton.focus();
    }
  });

  el.fitButton.addEventListener("click", () => {
    if (state.work) fitView();
  });
  el.predictButton.addEventListener("click", () => predict());

  el.heatToggle.addEventListener("change", () => {
    state.heat.on = el.heatToggle.checked;
    syncHeatData();
    drawCues();
    scheduleDraw();
  });
  el.heatOpacity.addEventListener("input", () => {
    state.heat.opacity = Number(el.heatOpacity.value) / 100;
    el.heatOpacityValue.textContent = `${el.heatOpacity.value}%`;
    syncHeatData();
    drawCues();
    scheduleDraw();
  });
  el.scaleRadios.forEach((radio) => radio.addEventListener("change", () => {
    if (!radio.checked) return;
    state.heat.mode = radio.value;
    syncHeatData();
    drawCues();
    scheduleDraw();
  }));

  el.canvas.addEventListener("pointerdown", onPointerDown);
  el.canvas.addEventListener("pointermove", onPointerMove);
  el.canvas.addEventListener("pointerup", onPointerUp);
  el.canvas.addEventListener("pointercancel", onPointerUp);
  el.canvas.addEventListener("pointerleave", (e) => {
    if (!state.drag) {
      state.hover = null;
      updateCursorReadout(null);
    }
    if (e.pointerType === "touch") state.pointers.delete(e.pointerId);
  });
  el.canvas.addEventListener("auxclick", (e) => e.preventDefault()); // middle-click must not autoscroll / paste
  el.canvas.addEventListener("contextmenu", (e) => { if (state.drag) e.preventDefault(); });
  el.stage.addEventListener("wheel", onWheel, { passive: false });
  el.stage.addEventListener("keydown", onStageKeyDown);
  el.stage.addEventListener("keyup", onStageKeyUp);
  window.addEventListener("blur", () => { state.spaceDown = false; });

  // Drag-and-drop onto the stage; dragDepth: number, nested enter/leave counter so children don't flicker
  let dragDepth = 0;
  el.stage.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    el.stage.classList.add("is-dragover");
  });
  el.stage.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  });
  el.stage.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) el.stage.classList.remove("is-dragover");
  });
  el.stage.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    el.stage.classList.remove("is-dragover");
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (state.camera.stream) stopCamera();
    onFileChosen(file, "drop");
  });
  // Dropping elsewhere must not navigate the tab away to the file.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  document.addEventListener("paste", (e) => {
    const items = e.clipboardData ? Array.from(e.clipboardData.items) : [];
    const item = items.find((it) => it.kind === "file");
    if (!item) return;
    e.preventDefault();
    if (state.camera.stream) stopCamera();
    onFileChosen(item.getAsFile(), "paste");
  });

  // Marching ants: advance the dash phase ~9 times a second (static when reduced motion is requested).
  setInterval(() => {
    if (!state.sel || reduceMotion.matches || document.hidden) return;
    state.ants = (state.ants + 1) % 10;
    scheduleDraw();
  }, 110);

  new ResizeObserver(() => resizeCanvas()).observe(el.stage);
  new ResizeObserver(() => drawCues()).observe(el.modelCanvas); // marker sizes follow the model view's CSS size
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    cssVarCache.clear();
    drawCues();
    scheduleDraw();
  });

  syncHeatData();
  syncView();
  resizeCanvas();
  drawCues();
  loadModelInfo();
  loadSamples();
})();
