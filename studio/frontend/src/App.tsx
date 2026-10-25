// Studio root: schema-driven controls (left), live preview + generate panel (right).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { Config, Schema, Tile, Value } from "./api";
import ParamControl from "./components/ParamControl";
import PreviewGrid from "./components/PreviewGrid";
import GeneratePanel from "./components/GeneratePanel";

/** Preview debounce after the last control change (ms, int). */
const DEBOUNCE_MS = 150;
/** Default number of preview tiles (int). */
const DEFAULT_COUNT = 9;
/** Smallest / largest preview tile count the backend accepts (int, int). */
const MIN_COUNT = 1;
const MAX_COUNT = 16;
/** Initial fixed preview seeds, list[int] of length DEFAULT_COUNT. */
const INITIAL_SEEDS = Array.from({ length: DEFAULT_COUNT }, (_, i) => i);
/** A fresh random preview seed, int in [0, 2^31). */
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

/** Errors of any thrown value as list[str]. */
function errorsOf(e: unknown): string[] {
  return e instanceof api.ApiError ? e.errors : [String(e)];
}

/** Studio root component (no props). Loads the schema + default config once,
 * renders one ParamControl per schema leaf, re-renders the 9 preview tiles
 * (debounced, newest response wins), handles YAML load/save and hosts the
 * GeneratePanel. Returns the whole page as a JSX element. */
export default function App() {
  const [schema, setSchema] = useState<Schema | null>(null); // section -> key -> Leaf
  const [config, setConfig] = useState<Config | null>(null); // current UI config
  // seed pool, list[int] of length >= count; never shrinks on count edits so shrink-then-grow
  // (or typing "16" via "1") restores the same tiles; only a reshuffle replaces it
  const [pool, setPool] = useState<number[]>(INITIAL_SEEDS);
  const [count, setCount] = useState(DEFAULT_COUNT); // preview tile count, int in [1,16]
  // text typed in the count box, str; may be non-numeric until committed or blurred
  const [countDraft, setCountDraft] = useState(String(DEFAULT_COUNT));
  // seed scheme used when growing: "seq" = max+1.. (initial), "random" = after a reshuffle
  // seeds sent to the preview, list[int] of length count; identity changes only when the shown list does
  const seeds = useMemo(() => pool.slice(0, count), [pool, count]);
  const [scheme, setScheme] = useState<"seq" | "random">("seq");
  // built-in default config (configs/v7_default.yaml), section -> key -> Value; set once at
  // startup and never replaced by a loaded YAML; source of the per-control reset buttons
  const [defaults, setDefaults] = useState<Config | null>(null);
  const [tiles, setTiles] = useState<Tile[]>([]); // latest applied tiles
  // config snapshot the applied tiles were rendered from (section -> key -> Value), or null
  // before the first preview; per-tile Save bundles with this, never the newer slider state
  const [appliedConfig, setAppliedConfig] = useState<Config | null>(null);
  // errors of the last per-tile Save (list[str]); cleared by a successful save
  const [saveErrors, setSaveErrors] = useState<string[]>([]);
  const [overlay, setOverlay] = useState(false); // tip markers on/off
  // errors of the last preview request (list[str]); cleared by a successful preview
  const [previewErrors, setPreviewErrors] = useState<string[]>([]);
  // errors of the last YAML load / save / startup fetch (list[str]); a preview never
  // clears these, only a later successful load or a control change does
  const [fileErrors, setFileErrors] = useState<string[]>([]);
  // adjustments the last YAML load made to an old file (list[str], "section.key: message");
  // non-blocking, shown until dismissed or the next load
  const [loadWarnings, setLoadWarnings] = useState<string[]>([]);
  // live too-tight pointer gray range warnings of the current config (list[str],
  // "section.key: message"); recomputed on every config change, [] when fine
  const [rangeWarnings, setRangeWarnings] = useState<string[]>([]);
  const checkSeqRef = useRef(0); // newest sent config-check request number (int)
  const [sentSeq, setSentSeq] = useState(0); // newest preview request number (int)
  const [appliedSeq, setAppliedSeq] = useState(0); // number of the response shown (int)
  const [busy, setBusy] = useState(true); // debounce pending or latest request in flight

  const seqRef = useRef(0); // newest sent request number, read inside callbacks (int)
  const pendingRef = useRef(false); // a debounce timer is armed (bool)
  const fileRef = useRef<HTMLInputElement>(null); // hidden YAML file picker

  // load schema + default config once
  useEffect(() => {
    Promise.all([api.getSchema(), api.getDefaultConfig()])
      .then(([s, c]) => {
        setSchema(s);
        setConfig(c);
        setDefaults(c);
      })
      .catch((e) => setFileErrors(errorsOf(e)));
  }, []);

  // debounced preview; only the response of the newest request is applied
  useEffect(() => {
    if (!config) return;
    pendingRef.current = true;
    setBusy(true);
    const timer = setTimeout(() => { // debounce timer handle (number)
      pendingRef.current = false;
      const seq = ++seqRef.current; // this request's number (int)
      const sent = config; // config snapshot sent with this request (Config, never mutated)
      setSentSeq(seq);
      api
        .preview(sent, seeds)
        .then((t) => {
          if (seq !== seqRef.current) return; // stale: a newer request was sent
          setTiles(t);
          setAppliedConfig(sent);
          setPreviewErrors([]);
        })
        .catch((e) => {
          if (seq !== seqRef.current) return;
          setPreviewErrors(errorsOf(e)); // keep the old tiles on screen
        })
        .finally(() => {
          if (seq !== seqRef.current) return;
          setAppliedSeq(seq);
          if (!pendingRef.current) setBusy(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [config, seeds]);

  // debounced range check on every config change; only the newest response is applied
  useEffect(() => {
    if (!config) return;
    const timer = setTimeout(() => { // debounce timer handle (number)
      const seq = ++checkSeqRef.current; // this request's number (int)
      api
        .checkConfig(config)
        .then((w) => {
          if (seq === checkSeqRef.current) setRangeWarnings(w);
        })
        .catch(() => {
          // an invalid config is reported by the preview; show no stale warnings
          if (seq === checkSeqRef.current) setRangeWarnings([]);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [config]);

  // set one leaf: (section str, key str, v Value) => void; also drops stale file errors
  const setParam = useCallback((section: string, key: string, v: Value) => {
    setFileErrors([]);
    setConfig((c) => (c ? { ...c, [section]: { ...c[section], [key]: v } } : c));
  }, []);

  // pick `count` new random preview seeds (ints in [0, 2^31)); () => void
  const reshuffle = () => {
    setScheme("random");
    setPool(Array.from({ length: count }, randomSeed));
  };

  // apply the count box text (str): whole numbers are clamped to 1..16 and resize the
  // shown seeds (grow the pool per scheme if needed); anything else is ignored
  const commitCount = (text: string) => {
    const t = text.trim(); // trimmed text (str)
    if (!/^[+-]?\d+$/.test(t)) return;
    const n = Math.min(MAX_COUNT, Math.max(MIN_COUNT, parseInt(t, 10))); // clamped count (int)
    setCountDraft(String(n));
    if (n === count) return;
    setCount(n);
    setPool((prev) => {
      if (n <= prev.length) return prev; // pool already covers n; slice happens in `seeds`
      const next = prev.slice(); // grown pool (list[int])
      const top = prev.length ? Math.max(...prev) : -1; // highest seed so far (int)
      for (let i = prev.length; i < n; i++) {
        next.push(scheme === "seq" ? top + 1 + (i - prev.length) : randomSeed());
      }
      return next;
    });
  };

  // read a YAML File, parse it server-side, adopt the config or show its errors;
  // (file: File) => Promise<void>
  const loadYaml = async (file: File) => {
    try {
      // parsed config + adjustments made to an old YAML (ParsedConfig)
      const parsed = await api.parseConfig(await file.text());
      setConfig(parsed.config);
      setFileErrors([]);
      setLoadWarnings(parsed.warnings);
    } catch (e) {
      setFileErrors(errorsOf(e));
      setLoadWarnings([]);
    }
  };

  // dump the current config to YAML and download it as v7_config.yaml; () => Promise<void>
  const saveYaml = async () => {
    if (!config) return;
    try {
      const text = await api.dumpConfig(config); // YAML text (str)
      // object URL (str) of a Blob holding the YAML text
      const url = URL.createObjectURL(new Blob([text], { type: "application/x-yaml" }));
      const a = document.createElement("a"); // temporary download link (HTMLAnchorElement)
      a.href = url;
      a.download = "v7_config.yaml";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setFileErrors(errorsOf(e));
    }
  };

  // everything shown in config-errors (list[str])
  const errors = [...fileErrors, ...previewErrors, ...saveErrors];
  const imgSize = (config?.canvas?.img_size as number) ?? 256; // tile side (px, int)

  return (
    <div className="app">
      <header>
        <h1>
          v7 Dial Studio <span className="sub">native-gray generator tuning</span>
        </h1>
        <div className="toolbar">
          <input
            ref={fileRef}
            type="file"
            accept=".yaml,.yml"
            data-testid="load-yaml"
            className="hidden-file"
            onChange={(e) => {
              const f = e.target.files?.[0]; // chosen YAML file (File | undefined)
              if (f) loadYaml(f);
              e.target.value = ""; // allow re-loading the same file
            }}
          />
          <button onClick={() => fileRef.current?.click()}>Load YAML</button>
          <button data-testid="save-yaml" onClick={saveYaml} disabled={!config}>
            Save YAML
          </button>
          <label className="bool">
            <input type="checkbox" data-testid="overlay-toggle" checked={overlay}
              onChange={(e) => setOverlay(e.target.checked)} />
            tip overlay
          </label>
          <label className="count">
            Preview images
            <input type="number" data-testid="preview-count" min={MIN_COUNT} max={MAX_COUNT}
              step={1} value={countDraft}
              onChange={(e) => { setCountDraft(e.target.value); commitCount(e.target.value); }}
              onKeyDown={(e) => { if (e.key === "Enter") commitCount(countDraft); }}
              onBlur={() => setCountDraft(String(count))} />
          </label>
          <button data-testid="reshuffle" onClick={reshuffle}>Reshuffle seeds</button>
        </div>
      </header>

      <aside className="controls">
        {schema && config &&
          Object.entries(schema).map(([section, keys]) => (
            <section key={section} data-section={section}>
              <h2>{section}</h2>
              {Object.entries(keys).map(([key, leaf]) => (
                <ParamControl key={key} section={section} name={key} leaf={leaf}
                  value={config[section][key]}
                  defaultValue={defaults?.[section]?.[key]}
                  onChange={(v) => setParam(section, key, v)} />
              ))}
            </section>
          ))}
      </aside>

      <main>
        <div data-testid="config-errors" className={"errors" + (errors.length ? " show" : "")}>
          {errors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
        </div>
        {loadWarnings.length > 0 && (
          <div data-testid="config-warnings" className="notice">
            <button type="button" className="dismiss" aria-label="Dismiss"
              onClick={() => setLoadWarnings([])}>Dismiss</button>
            <div>
              Loaded with {loadWarnings.length} adjustment{loadWarnings.length === 1 ? "" : "s"}:
            </div>
            <ul>
              {loadWarnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        {rangeWarnings.length > 0 && (
          <div data-testid="range-warnings" className="notice">
            <div>
              Pointer gray range{rangeWarnings.length === 1 ? " is" : "s are"} too tight:
            </div>
            <ul>
              {rangeWarnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        <PreviewGrid tiles={tiles} imgSize={imgSize} overlay={overlay}
          sentSeq={sentSeq} appliedSeq={appliedSeq} busy={busy} count={count}
          appliedConfig={appliedConfig} onSaveErrors={setSaveErrors} />
        <GeneratePanel config={config} />
      </main>
    </div>
  );
}
