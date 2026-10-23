// Grayscale web app frontend: pick/drop/paste an image, POST it to /api/grayscale, show both previews.
"use strict";

(() => {
  // MAX_BYTES: number, client-side upload limit in bytes (server enforces the same 20 MB)
  const MAX_BYTES = 20 * 1024 * 1024;

  // $: (id: string) => HTMLElement, shorthand lookup by element id
  const $ = (id) => document.getElementById(id);

  // el: object of HTMLElement refs used throughout the UI
  const el = {
    app: document.querySelector("[data-testid=app]"),
    dropZone: $("drop-zone"),
    fileInput: $("file-input"),
    methodSelect: $("method-select"),
    methodDescription: $("method-description"),
    formatSelect: $("format-select"),
    errorBox: $("error-box"),
    results: $("results"),
    status: $("status"),
    originalImg: $("original-img"),
    originalMeta: $("original-meta"),
    resultImg: $("result-img"),
    resultFrame: $("result-frame"),
    resultMeta: $("result-meta"),
    spinner: $("spinner"),
    downloadButton: $("download-button"),
    resetButton: $("reset-button"),
  };

  // state: mutable app state
  //   file:        File | null, the image currently being shown/converted
  //   originalUrl: string | null, object URL of the original preview
  //   resultUrl:   string | null, object URL of the converted image
  //   controller:  AbortController | null, aborts an in-flight conversion when a new one starts
  //   descriptions: Map<string, string>, method id -> description text
  const state = { file: null, originalUrl: null, resultUrl: null, controller: null, descriptions: new Map() };

  // formatBytes: (n: number) => string, e.g. 1536 -> "1.5 KB"
  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }

  // setState: (name: "idle"|"loading"|"done"|"error") => void, mirrors the phase onto data-state for styling/tests
  function setState(name) {
    el.app.dataset.state = name;
  }

  function showError(message) {
    el.errorBox.textContent = message;
    el.errorBox.hidden = false;
    setState("error");
  }

  function clearError() {
    el.errorBox.textContent = "";
    el.errorBox.hidden = true;
  }

  // replaceUrl: (key: "originalUrl"|"resultUrl", blob: Blob|null) => string|null, revokes the old object URL
  function replaceUrl(key, blob) {
    if (state[key]) URL.revokeObjectURL(state[key]);
    state[key] = blob ? URL.createObjectURL(blob) : null;
    return state[key];
  }

  function setDownloadEnabled(enabled) {
    el.downloadButton.classList.toggle("btn-disabled", !enabled);
    el.downloadButton.setAttribute("aria-disabled", String(!enabled));
    if (!enabled) {
      el.downloadButton.setAttribute("href", "#");
      el.downloadButton.tabIndex = -1;
    } else {
      el.downloadButton.removeAttribute("tabindex");
    }
  }

  // filenameFromDisposition: (header: string|null) => string|null, prefers RFC 5987 filename*= over filename=
  function filenameFromDisposition(header) {
    if (!header) return null;
    const star = /filename\*=UTF-8''([^;]+)/i.exec(header);
    if (star) {
      try { return decodeURIComponent(star[1].trim()); } catch (_) { /* fall through */ }
    }
    const plain = /filename="?([^";]+)"?/i.exec(header);
    return plain ? plain[1].trim() : null;
  }

  // ---- Methods dropdown -------------------------------------------------------------------------

  async function loadMethods() {
    try {
      const res = await fetch("/api/methods");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // methods: Array<{id: string, label: string, description: string}>
      const methods = await res.json();
      const previous = el.methodSelect.value;
      el.methodSelect.replaceChildren(
        ...methods.map((m) => {
          state.descriptions.set(m.id, m.description);
          return new Option(m.label, m.id);
        }),
      );
      if (methods.some((m) => m.id === previous)) el.methodSelect.value = previous;
    } catch (err) {
      showError("Could not load the list of methods from the server. The default method will be used.");
    }
    updateMethodDescription();
  }

  function updateMethodDescription() {
    el.methodDescription.textContent = state.descriptions.get(el.methodSelect.value) || "";
  }

  // ---- File intake --------------------------------------------------------------------------------

  // validateFile: (file: File) => string|null, a friendly error message or null when acceptable
  function validateFile(file) {
    if (file.type && !file.type.startsWith("image/")) {
      return `"${file.name}" is not an image. Please choose a PNG, JPEG, WebP, GIF, BMP or TIFF file.`;
    }
    if (file.size > MAX_BYTES) {
      return `"${file.name}" is ${formatBytes(file.size)}; the maximum is 20 MB.`;
    }
    if (file.size === 0) return `"${file.name}" is empty.`;
    return null;
  }

  // handleFile: (file: File|undefined) => void, entry point for picker, drop and paste
  function handleFile(file) {
    if (!file) return;
    clearError();
    const problem = validateFile(file);
    if (problem) {
      showError(problem);
      return;
    }
    state.file = file;
    el.results.hidden = false;
    // A different picture: drop the previous result so it is never shown next to the wrong original.
    el.resultImg.hidden = true;
    el.resultImg.removeAttribute("src");
    delete el.resultImg.dataset.method;
    replaceUrl("resultUrl", null);
    el.resultMeta.textContent = "";
    el.originalImg.src = replaceUrl("originalUrl", file);
    el.originalMeta.textContent = formatBytes(file.size);
    el.originalImg.onload = () => {
      el.originalMeta.textContent =
        `${el.originalImg.naturalWidth} × ${el.originalImg.naturalHeight} px · ${formatBytes(file.size)}`;
    };
    el.originalImg.onerror = () => { el.originalMeta.textContent = formatBytes(file.size); };
    convert();
  }

  // ---- Conversion ---------------------------------------------------------------------------------

  async function convert() {
    if (!state.file) return;
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;

    const methodLabel = el.methodSelect.selectedOptions[0]?.textContent || el.methodSelect.value;
    clearError();
    setState("loading");
    el.status.textContent = `Converting with ${methodLabel}…`;
    el.spinner.hidden = false;
    el.resultFrame.classList.add("is-loading");
    setDownloadEnabled(false);

    // body: FormData with fields file (File), method (string), format (string)
    const body = new FormData();
    body.append("file", state.file, state.file.name || "image");
    body.append("method", el.methodSelect.value);
    body.append("format", el.formatSelect.value);

    try {
      const res = await fetch("/api/grayscale", { method: "POST", body, signal: controller.signal });
      if (!res.ok) {
        let detail = `The server could not convert this image (HTTP ${res.status}).`;
        try {
          const json = await res.json();
          if (typeof json.detail === "string") detail = json.detail;
        } catch (_) { /* non-JSON error body */ }
        throw new Error(detail);
      }
      // blob: Blob, the converted image bytes
      const blob = await res.blob();
      if (controller !== state.controller) return; // a newer request superseded this one

      const width = res.headers.get("X-Image-Width");
      const height = res.headers.get("X-Image-Height");
      const method = res.headers.get("X-Gray-Method") || el.methodSelect.value;
      const filename = filenameFromDisposition(res.headers.get("Content-Disposition")) || "image_gray";

      const url = replaceUrl("resultUrl", blob);
      await new Promise((resolve) => {
        el.resultImg.onload = resolve;
        el.resultImg.onerror = resolve;
        el.resultImg.src = url;
      });
      if (controller !== state.controller) return;

      el.resultImg.hidden = false;
      el.resultImg.dataset.method = method;
      el.resultImg.dataset.format = el.formatSelect.value;
      el.resultMeta.textContent = `${width} × ${height} px · ${formatBytes(blob.size)}`;
      el.downloadButton.href = url;
      el.downloadButton.download = filename;
      el.downloadButton.title = `Save as ${filename}`;
      setDownloadEnabled(true);
      el.status.textContent = `Converted with ${methodLabel}.`;
      setState("done");
    } catch (err) {
      if (err.name === "AbortError") return;
      if (controller !== state.controller) return;
      replaceUrl("resultUrl", null);
      el.resultImg.hidden = true;
      el.resultImg.removeAttribute("src");
      delete el.resultImg.dataset.method;
      el.resultMeta.textContent = "";
      el.status.textContent = "Conversion failed.";
      showError(err.message || "Network error: could not reach the server.");
    } finally {
      if (controller === state.controller) {
        el.spinner.hidden = true;
        el.resultFrame.classList.remove("is-loading");
        state.controller = null;
      }
    }
  }

  function reset() {
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.file = null;
    replaceUrl("originalUrl", null);
    replaceUrl("resultUrl", null);
    el.originalImg.removeAttribute("src");
    el.resultImg.removeAttribute("src");
    el.resultImg.hidden = true;
    delete el.resultImg.dataset.method;
    el.originalMeta.textContent = "";
    el.resultMeta.textContent = "";
    el.status.textContent = "";
    el.spinner.hidden = true;
    el.resultFrame.classList.remove("is-loading");
    setDownloadEnabled(false);
    el.results.hidden = true;
    el.fileInput.value = "";
    clearError();
    setState("idle");
    el.dropZone.focus();
  }

  // ---- Event wiring -------------------------------------------------------------------------------

  el.dropZone.addEventListener("click", () => el.fileInput.click());
  el.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.fileInput.click();
    }
  });
  el.fileInput.addEventListener("click", (e) => e.stopPropagation()); // avoid re-triggering the zone
  el.fileInput.addEventListener("change", () => {
    handleFile(el.fileInput.files[0]);
    el.fileInput.value = ""; // allow choosing the same file again
  });

  // dragDepth: number, nested dragenter/dragleave counter so child elements don't flicker the highlight
  let dragDepth = 0;
  el.dropZone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    el.dropZone.classList.add("is-dragover");
  });
  el.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  });
  el.dropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) el.dropZone.classList.remove("is-dragover");
  });
  el.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    el.dropZone.classList.remove("is-dragover");
    handleFile(e.dataTransfer && e.dataTransfer.files[0]);
  });
  // Dropping outside the zone must not navigate the tab away to the file.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  document.addEventListener("paste", (e) => {
    const items = e.clipboardData ? Array.from(e.clipboardData.items) : [];
    const item = items.find((it) => it.kind === "file");
    if (!item) return;
    e.preventDefault();
    const file = item.getAsFile();
    if (file && !file.name) {
      handleFile(new File([file], "pasted-image.png", { type: file.type }));
    } else {
      handleFile(file);
    }
  });

  el.methodSelect.addEventListener("change", () => {
    updateMethodDescription();
    convert();
  });
  el.formatSelect.addEventListener("change", () => convert());
  el.resetButton.addEventListener("click", reset);

  setDownloadEnabled(false);
  loadMethods();
})();
