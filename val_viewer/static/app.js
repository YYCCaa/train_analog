/* Val viewer page logic: fetch summary + best / worst lists of a set, draw entries, markers, detail view.
 *
 *   /api/summary?set, /api/list?set&side&n --> state.entries --> .entry cards (lazy <img>, svg.markers overlay)
 *   set radios (val | train | all) and the n field reload both; scale radios / markers checkbox change
 *   attributes in place (lists are not refetched)
 *   click entry --> #detail with 512 px panels + two x8 zoom panels of the 32 x 32 px window at the true tip
 *                   + the Parameters column (/api/params/{row}, flattened into sections)
 *   An image is addressed by its dataset row everywhere (urls, data-row attributes).
 *
 * Coordinates: an image pixel i is centred at i, so a point (x, y) is drawn at (x + .5, y + .5) in an SVG
 * viewBox "0 0 256 256" (the 128 px heatmap is shown x2, so both panels share the coordinates).
 */
"use strict";

// N_MIN, N_MAX: int, allowed range of the n field
const N_MIN = 1, N_MAX = 1000;
// ZOOM_WIN: int, side of the zoom window in image px; ZOOM_K: int, css px per image px in the zoom panels
const ZOOM_WIN = 32, ZOOM_K = 8;

// state: page state. n: int; set: "val" | "train" | "all", the chosen image set; scale: "log" | "linear";
//   tok: int, id of the newest load (summary + lists); ptok: int, id of the newest parameters request;
//   entries: {best: Entry[], worst: Entry[]},
//   Entry = {rank, row (dataset row), split ("train"|"val"), split_pos, name, label:[x,y], tip:[x,y], err, conf};
//   detail: {side: "best"|"worst", idx: int (0-based)} | null; loadedN: int|null, n of the lists on screen;
//   pendingN: int|null, n of the load in flight; opener: HTMLElement|null, entry that opened the detail view;
//   key: string, rank-cache key from /api/summary (page-load timestamp if that failed), appended as v= to PNG urls
const state = { n: 24, set: "val", scale: "log", tok: 0, ptok: 0, entries: { best: [], worst: [] }, detail: null,
  loadedN: null, pendingN: null, opener: null, key: String(Date.now()) };

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
  return `<div class="l1"><b>#${e.rank}</b> <span class="tag tag-${e.split}">${e.split}</span> `
    + `${e.split} ${e.split_pos} · <span class="name">${esc(e.name)}</span></div>`
    + `<div class="l2">err ${f(e.err, 2)} px · conf ${f(e.conf, 3)}</div>`
    + `<div class="l3">true (${f(e.label[0], 1)}, ${f(e.label[1], 1)}) · pred (${f(e.tip[0], 1)}, ${f(e.tip[1], 1)})</div>`;
}

/** URL (string) of the heatmap PNG of dataset row `row` at the current scale. */
const heatUrl = (row) => `/api/heatmap/${row}.png?scale=${state.scale}&v=${encodeURIComponent(state.key)}`;
/** URL (string) of the dial PNG of dataset row `row`. */
const dialUrl = (row) => `/api/image/${row}.png?v=${encodeURIComponent(state.key)}`;

/** One entry card (HTMLElement) for side ("best"|"worst"), entry e. */
function buildEntry(side, e) {
  const el = document.createElement("article");
  el.className = "entry";
  el.tabIndex = 0;
  el.setAttribute("role", "button");
  el.setAttribute("aria-label", `open detail of #${e.rank}, ${e.split} ${e.split_pos}`);
  el.dataset.side = side; el.dataset.rank = e.rank; el.dataset.row = e.row;
  const marks = `<svg class="markers" viewBox="0 0 256 256">${markersInner(e)}</svg>`;
  el.innerHTML = `<div class="caption">${captionHtml(e)}</div><div class="panels">`
    + `<div class="panel"><img class="dial" loading="lazy" alt="dial ${e.row}" src="${dialUrl(e.row)}">${marks}</div>`
    + `<div class="panel"><img class="heatmap" loading="lazy" alt="heatmap ${e.row}" src="${heatUrl(e.row)}">${marks}</div></div>`;
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

/** Load the summary and both lists of state.set for state.n and show them together; a stale response
 *  (older request, e.g. an earlier set) is dropped whole. A failed summary alone does not block the lists. */
async function loadLists() {
  const tok = ++state.tok;
  const n = state.n, set = state.set;  // n: int, set: string, what this load asks for
  state.pendingN = n;
  let sumErr = "";  // string, message of a failed summary request
  try {
    const [sum, best, worst] = await Promise.all([
      getJson(`/api/summary?set=${set}`).catch((err) => { sumErr = String(err.message || err); return null; }),
      getJson(`/api/list?set=${set}&side=best&n=${n}`), getJson(`/api/list?set=${set}&side=worst&n=${n}`)]);
    if (tok !== state.tok) return;
    if (sum) {
      if (sum.key) state.key = sum.key;
      $("#summary").textContent = summaryText(sum);
    }
    state.entries = { best, worst };
    state.loadedN = state.pendingN = n;
    closeDetail();
    renderLists();
    statusEl.textContent = sumErr;
  } catch (err) {
    if (tok === state.tok) { state.pendingN = null; statusEl.textContent = String(err.message || err); }
  }
}

/** Header text (string) from the summary dict. */
function summaryText(s) {
  const f = (v) => v.toLocaleString("en-US");
  const size = s.set === "all" ? `all ${f(s.n)} (train ${f(s.n_train)} · val ${f(s.n_val)})` : `${s.set} ${f(s.n)}`;
  return `Val viewer · TipNet v9 (${s.run}, ${s.weights}) · ${size}`
    + ` · p99 ${s.p99.toFixed(2)} px · >3 px ${s.n_over_3} · >7 px ${s.n_over_7}`;
}

/** Apply the n field: clamp to 1..1000, show the value used, reload when it changed. */
function applyN() {
  let v = parseInt(nInput.value, 10);
  if (!Number.isFinite(v)) v = state.n;
  v = Math.min(N_MAX, Math.max(N_MIN, v));
  nInput.value = String(v);
  if (v === state.loadedN || v === state.pendingN) return;  // already shown or loading; a failed load clears both
  state.n = v;
  loadLists();
}

/** Point every heatmap image (list + detail) at the current scale. */
function applyScale() {
  for (const img of document.querySelectorAll(".entry img.heatmap")) {
    img.src = heatUrl(img.closest(".entry").dataset.row);
  }
  if (state.detail) showDetail();
}

/* ---------------- detail view ---------------- */

/** Open the detail view on entry idx (int, 0-based) of side. */
function openDetail(side, idx, opener) {
  state.opener = opener || null;
  state.detail = { side, idx };
  detailEl.hidden = false;
  document.body.style.overflow = "hidden";
  showDetail();
  $("#detail-close").focus();
}

/** Close the detail view. */
function closeDetail() {
  const back = state.opener;
  state.opener = null;
  state.detail = null;
  detailEl.hidden = true;
  document.body.style.overflow = "";
  if (back && back.isConnected) back.focus();  // return focus to the entry that opened the view
}

/** Fill the detail view from state.detail. */
function showDetail() {
  const { side, idx } = state.detail;
  const list = state.entries[side];
  const e = list[idx];
  detailEl.dataset.side = side; detailEl.dataset.rank = e.rank; detailEl.dataset.row = e.row;
  $("#detail-caption").innerHTML = `<span class="side">${side.toUpperCase()}</span> ` + captionHtml(e);
  $("#detail-prev").disabled = idx <= 0;
  $("#detail-next").disabled = idx >= list.length - 1;
  const big = detailEl.querySelectorAll(".big-row .panel");
  big[0].querySelector("img").src = dialUrl(e.row);
  big[1].querySelector("img").src = heatUrl(e.row);
  for (const p of big) p.querySelector("svg").innerHTML = markersInner(e);
  // zoom window: 32 x 32 image px around the true tip, top-left clamped to 0..224
  const clamp = (v) => Math.min(256 - ZOOM_WIN, Math.max(0, v));
  const x0 = clamp(Math.round(e.label[0]) - ZOOM_WIN / 2), y0 = clamp(Math.round(e.label[1]) - ZOOM_WIN / 2);
  for (const [id, url] of [["#zoom-dial", dialUrl(e.row)], ["#zoom-heat", heatUrl(e.row)]]) {
    const z = $(id);
    z.dataset.x0 = x0; z.dataset.y0 = y0;
    const img = z.querySelector("img");
    img.src = url;
    img.style.left = `${-x0 * ZOOM_K}px`; img.style.top = `${-y0 * ZOOM_K}px`;
    const svg = z.querySelector("svg");
    svg.setAttribute("viewBox", `${x0} ${y0} ${ZOOM_WIN} ${ZOOM_WIN}`);
    svg.innerHTML = markersInner(e);
  }
  loadParams(e);
}

/* ---------------- parameters column ---------------- */

/** True for a plain JSON object (not null, not an array). */
const isDict = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
/** True if the JSON value v is, or contains at any depth, a plain object. */
const hasDict = (v) => isDict(v) || (Array.isArray(v) && v.some(hasDict));

/** Append the [name, text] rows of JSON value v to rows (Array<[string, string]>), names dotted from prefix
 *  (string): nested objects -> `a.b.c`, lists holding objects -> numbered `items #1.gray`, other lists inline. */
function flatten(prefix, v, rows) {
  const join = (k) => (prefix ? `${prefix}.${k}` : k);
  if (isDict(v)) {
    const keys = Object.keys(v);
    if (!keys.length && prefix) rows.push([prefix, "{}"]);
    for (const k of keys) flatten(join(k), v[k], rows);
  } else if (Array.isArray(v) && hasDict(v)) {
    v.forEach((item, i) => flatten(`${prefix} #${i + 1}`, item, rows));
  } else if (Array.isArray(v)) {
    rows.push([prefix, JSON.stringify(v)]);
  } else {
    rows.push([prefix, String(v)]);  // numbers, strings ("NaN" etc.), booleans, null shown as stored
  }
}

/** Number or string v -> text with `d` decimals if it is a finite number, else String(v) ("NaN" strings stay). */
const fixed = (v, d) => (typeof v === "number" ? v.toFixed(d) : String(v));

/** Show the record rec ({row, filename, seed, angle_deg, tip_x, tip_y, sampled}) in the Parameters column. */
function renderParams(rec) {
  const box = $("#params");
  let html = `<div class="p-head">Parameters · seed ${esc(rec.seed)} · angle ${esc(rec.angle_deg)}° · `
    + `tip (${esc(fixed(rec.tip_x, 1))}, ${esc(fixed(rec.tip_y, 1))})</div>`;
  const sampled = isDict(rec.sampled) ? rec.sampled : { sampled: rec.sampled };
  for (const sec of Object.keys(sampled)) {
    const rows = [];  // Array<[name, text]> of this section
    if (isDict(sampled[sec])) flatten("", sampled[sec], rows); else flatten(sec, sampled[sec], rows);
    html += `<details open data-section="${esc(sec)}"><summary>${esc(sec)}</summary>`
      + rows.map(([k, v]) => `<div class="prow" data-key="${esc(k)}"><span class="k">${esc(k)}</span>`
        + `<span class="v">${esc(v)}</span></div>`).join("") + `</details>`;
  }
  box.innerHTML = html;
}

/** Show one muted message line (string) in the Parameters column; "no parameters saved ..." gets the studio hint. */
function renderParamsMessage(text) {
  const msg = text.charAt(0).toUpperCase() + text.slice(1);
  const hint = /^no parameters saved/i.test(text) ? " — generate the dataset with 'save parameters' ticked in the studio" : "";
  $("#params").innerHTML = `<div class="p-msg">${esc(msg + hint)}</div>`;
}

/** Fetch and show the parameters of entry e; a stale answer (the detail moved on) is dropped. */
async function loadParams(e) {
  const tok = ++state.ptok;
  $("#params").innerHTML = `<div class="p-msg">Loading parameters…</div>`;
  try {
    const r = await fetch(`/api/params/${e.row}`);
    const body = await r.json().catch(() => ({}));  // {detail} on errors, the record on 200
    if (tok !== state.ptok) return;
    if (r.ok) renderParams(body); else renderParamsMessage(String(body.detail || `HTTP ${r.status}`));
  } catch (err) {
    if (tok === state.ptok) renderParamsMessage(`parameters unavailable: ${err.message || err}`);
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
for (const r of document.querySelectorAll("input[name=set]")) {
  r.addEventListener("change", () => { if (r.checked) { state.set = r.value; loadLists(); } });
}
for (const side of ["best", "worst"]) {
  $(`#col-${side}`).addEventListener("click", (ev) => {
    const el = ev.target.closest(".entry");
    if (el) openDetail(side, Number(el.dataset.rank) - 1, el);
  });
  $(`#col-${side}`).addEventListener("keydown", (ev) => {
    const el = ev.target.closest(".entry");
    if (el && ev.target === el && (ev.key === "Enter" || ev.key === " ")) {
      ev.preventDefault();
      openDetail(side, Number(el.dataset.rank) - 1, el);
    }
  });
}
$("#detail-prev").addEventListener("click", () => stepDetail(-1));
$("#detail-next").addEventListener("click", () => stepDetail(1));
$("#detail-close").addEventListener("click", closeDetail);
document.addEventListener("keydown", (ev) => {
  if (!state.detail) return;
  if (ev.key === "Escape") closeDetail();
  else if (ev.key === "Tab") {  // focus trap: cycle through the enabled buttons of the dialog
    const btns = [...detailEl.querySelectorAll("button:not(:disabled)")];
    const i = btns.indexOf(document.activeElement);
    ev.preventDefault();
    btns[(i + (ev.shiftKey ? -1 : 1) + btns.length) % btns.length].focus();
  }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); stepDetail(1); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); stepDetail(-1); }
});

// legend symbols reuse the marker drawings
$("#legend-true").innerHTML = trueSymbol();
$("#legend-pred").innerHTML = predSymbol();

loadLists();
