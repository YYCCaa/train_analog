/* Val viewer page logic: fetch summary + best / worst lists, draw entries, markers, detail view.
 *
 *   /api/summary, /api/list?side&n --> state.entries --> .entry cards (lazy <img>, svg.markers overlay)
 *   scale radios / markers checkbox change attributes in place (lists are not refetched)
 *   click entry --> #detail with 512 px panels + two x8 zoom panels of the 32 x 32 px window at the true tip
 *
 * Coordinates: an image pixel i is centred at i, so a point (x, y) is drawn at (x + .5, y + .5) in an SVG
 * viewBox "0 0 256 256" (the 128 px heatmap is shown x2, so both panels share the coordinates).
 */
"use strict";

// N_MIN, N_MAX: int, allowed range of the n field
const N_MIN = 1, N_MAX = 1000;
// ZOOM_WIN: int, side of the zoom window in image px; ZOOM_K: int, css px per image px in the zoom panels
const ZOOM_WIN = 32, ZOOM_K = 8;

// state: page state. n: int; scale: "log" | "linear"; tok: int, id of the newest list load;
//   entries: {best: Entry[], worst: Entry[]}, Entry = {rank, pos, name, label:[x,y], tip:[x,y], err, conf};
//   detail: {side: "best"|"worst", idx: int (0-based)} | null; loaded: bool, a list load has succeeded
const state = { n: 24, scale: "log", tok: 0, entries: { best: [], worst: [] }, detail: null, loaded: false };

const $ = (sel) => document.querySelector(sel);
const nInput = $("#n-input"), statusEl = $("#status"), detailEl = $("#detail");

/** Escape text for innerHTML. */
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/** Marker SVG contents (string) for entry e (coordinates in image px): line, true tip, predicted tip. */
function markersInner(e) {
  const lx = e.label[0] + 0.5, ly = e.label[1] + 0.5, px = e.tip[0] + 0.5, py = e.tip[1] + 0.5;
  return `<line class="ol" x1="${px}" y1="${py}" x2="${lx}" y2="${ly}"/>`
    + `<line class="err-line" x1="${px}" y1="${py}" x2="${lx}" y2="${ly}"/>`
    + `<g class="true-tip" data-x="${lx}" data-y="${ly}" transform="translate(${lx} ${ly})">${trueSymbol()}</g>`
    + `<g class="pred-tip" data-x="${px}" data-y="${py}" transform="translate(${px} ${py})">${predSymbol()}</g>`;
}

/** True-tip symbol (string): outlines first, then green crosshair with open centre + circles r 3 and 7. */
function trueSymbol() {
  const arms = "M-10.5 0H-1.6M1.6 0H10.5M0 -10.5V-1.6M0 1.6V10.5";
  return `<path class="ol" d="${arms}"/><circle class="ol" r="3"/><circle class="ol" r="7"/>`
    + `<path class="fg" d="${arms}"/><circle class="fg ring-3" r="3"/><circle class="fg ring-7" r="7"/>`;
}

/** Predicted-tip symbol (string): orange x with dark outline. */
function predSymbol() {
  const x = "M-4 -4L4 4M-4 4L4 -4";
  return `<path class="ol" d="${x}"/><path class="fg" d="${x}"/>`;
}

/** Caption HTML (string) for entry e. */
function captionHtml(e) {
  const f = (v, d) => v.toFixed(d);
  return `<div class="l1"><b>#${e.rank}</b> val ${e.pos} · <span class="name">${esc(e.name)}</span></div>`
    + `<div class="l2">err ${f(e.err, 2)} px · conf ${f(e.conf, 3)}</div>`
    + `<div class="l3">true (${f(e.label[0], 1)}, ${f(e.label[1], 1)}) · pred (${f(e.tip[0], 1)}, ${f(e.tip[1], 1)})</div>`;
}

/** URL (string) of the heatmap PNG of val position pos at the current scale. */
const heatUrl = (pos) => `/api/heatmap/${pos}.png?scale=${state.scale}`;
/** URL (string) of the dial PNG of val position pos. */
const dialUrl = (pos) => `/api/image/${pos}.png`;

/** One entry card (HTMLElement) for side ("best"|"worst"), entry e. */
function buildEntry(side, e) {
  const el = document.createElement("article");
  el.className = "entry";
  el.dataset.side = side; el.dataset.rank = e.rank; el.dataset.pos = e.pos;
  const marks = `<svg class="markers" viewBox="0 0 256 256">${markersInner(e)}</svg>`;
  el.innerHTML = `<div class="caption">${captionHtml(e)}</div><div class="panels">`
    + `<div class="panel"><img class="dial" loading="lazy" alt="dial ${e.pos}" src="${dialUrl(e.pos)}">${marks}</div>`
    + `<div class="panel"><img class="heatmap" loading="lazy" alt="heatmap ${e.pos}" src="${heatUrl(e.pos)}">${marks}</div></div>`;
  return el;
}

/** Redraw both columns from state.entries. */
function renderLists() {
  for (const side of ["best", "worst"]) {
    const col = $(`#col-${side}`);
    const list = state.entries[side];
    col.querySelector("h2").textContent = `${side.toUpperCase()} ${list.length}`;
    const frag = document.createDocumentFragment();
    for (const e of list) frag.appendChild(buildEntry(side, e));
    col.querySelector(".list").replaceChildren(frag);
  }
}

/** Fetch JSON from url; throws on a non-200. */
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

/** Load both lists for state.n; a stale response (older request) is dropped. */
async function loadLists() {
  const tok = ++state.tok;
  try {
    const [best, worst] = await Promise.all([
      getJson(`/api/list?side=best&n=${state.n}`), getJson(`/api/list?side=worst&n=${state.n}`)]);
    if (tok !== state.tok) return;
    state.entries = { best, worst };
    state.loaded = true;
    closeDetail();
    renderLists();
    statusEl.textContent = "";
  } catch (err) {
    if (tok === state.tok) statusEl.textContent = String(err.message || err);
  }
}

/** Header text (string) from the summary dict. */
function summaryText(s) {
  return `Val viewer · TipNet v9 (${s.run}, ${s.weights}) · val ${s.n_val.toLocaleString("en-US")}`
    + ` · p99 ${s.p99.toFixed(2)} px · >3 px ${s.n_over_3} · >7 px ${s.n_over_7}`;
}

/** Apply the n field: clamp to 1..1000, show the value used, reload when it changed. */
function applyN() {
  let v = parseInt(nInput.value, 10);
  if (!Number.isFinite(v)) v = state.n;
  v = Math.min(N_MAX, Math.max(N_MIN, v));
  nInput.value = String(v);
  if (v === state.n && state.loaded) return;
  state.n = v;
  loadLists();
}

/** Point every heatmap image (list + detail) at the current scale. */
function applyScale() {
  for (const img of document.querySelectorAll(".entry img.heatmap")) {
    img.src = heatUrl(img.closest(".entry").dataset.pos);
  }
  if (state.detail) showDetail();
}

/* ---------------- detail view ---------------- */

/** Open the detail view on entry idx (int, 0-based) of side. */
function openDetail(side, idx) {
  state.detail = { side, idx };
  detailEl.hidden = false;
  document.body.style.overflow = "hidden";
  showDetail();
  $("#detail-close").focus();
}

/** Close the detail view. */
function closeDetail() {
  state.detail = null;
  detailEl.hidden = true;
  document.body.style.overflow = "";
}

/** Fill the detail view from state.detail. */
function showDetail() {
  const { side, idx } = state.detail;
  const list = state.entries[side];
  const e = list[idx];
  detailEl.dataset.side = side; detailEl.dataset.rank = e.rank; detailEl.dataset.pos = e.pos;
  $("#detail-caption").innerHTML = `<span class="side">${side.toUpperCase()}</span> ` + captionHtml(e);
  $("#detail-prev").disabled = idx <= 0;
  $("#detail-next").disabled = idx >= list.length - 1;
  const big = detailEl.querySelectorAll(".big-row .panel");
  big[0].querySelector("img").src = dialUrl(e.pos);
  big[1].querySelector("img").src = heatUrl(e.pos);
  for (const p of big) p.querySelector("svg").innerHTML = markersInner(e);
  // zoom window: 32 x 32 image px around the true tip, top-left clamped to 0..224
  const clamp = (v) => Math.min(256 - ZOOM_WIN, Math.max(0, v));
  const x0 = clamp(Math.round(e.label[0]) - ZOOM_WIN / 2), y0 = clamp(Math.round(e.label[1]) - ZOOM_WIN / 2);
  for (const [id, url] of [["#zoom-dial", dialUrl(e.pos)], ["#zoom-heat", heatUrl(e.pos)]]) {
    const z = $(id);
    z.dataset.x0 = x0; z.dataset.y0 = y0;
    const img = z.querySelector("img");
    img.src = url;
    img.style.left = `${-x0 * ZOOM_K}px`; img.style.top = `${-y0 * ZOOM_K}px`;
    const svg = z.querySelector("svg");
    svg.setAttribute("viewBox", `${x0} ${y0} ${ZOOM_WIN} ${ZOOM_WIN}`);
    svg.innerHTML = markersInner(e);
  }
}

/** Step the open detail view by delta (-1 | +1) within its list; stays put at the ends. */
function stepDetail(delta) {
  if (!state.detail) return;
  const { side, idx } = state.detail;
  const j = idx + delta;
  if (j < 0 || j >= state.entries[side].length) return;
  state.detail.idx = j;
  showDetail();
}

/* ---------------- wiring ---------------- */

nInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") applyN(); });
nInput.addEventListener("change", applyN);
nInput.addEventListener("blur", applyN);
for (const r of document.querySelectorAll("input[name=scale]")) {
  r.addEventListener("change", () => { if (r.checked) { state.scale = r.value; applyScale(); } });
}
$("#markers-toggle").addEventListener("change", (ev) => {
  document.body.classList.toggle("hide-markers", !ev.target.checked);
});
for (const side of ["best", "worst"]) {
  $(`#col-${side}`).addEventListener("click", (ev) => {
    const el = ev.target.closest(".entry");
    if (el) openDetail(side, Number(el.dataset.rank) - 1);
  });
}
$("#detail-prev").addEventListener("click", () => stepDetail(-1));
$("#detail-next").addEventListener("click", () => stepDetail(1));
$("#detail-close").addEventListener("click", closeDetail);
document.addEventListener("keydown", (ev) => {
  if (!state.detail) return;
  if (ev.key === "Escape") closeDetail();
  else if (ev.key === "ArrowRight") { ev.preventDefault(); stepDetail(1); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); stepDetail(-1); }
});

// legend symbols reuse the marker drawings
$("#legend-true").innerHTML = trueSymbol();
$("#legend-pred").innerHTML = predSymbol();

(async function init() {
  try {
    $("#summary").textContent = summaryText(await getJson("/api/summary"));
  } catch (err) {
    statusEl.textContent = String(err.message || err);
  }
  await loadLists();
})();
