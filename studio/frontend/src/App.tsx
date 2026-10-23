// Studio root: schema-driven controls (left), live preview + generate panel (right).
import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Config, Schema, Tile, Value } from "./api";
import ParamControl from "./components/ParamControl";
import PreviewGrid from "./components/PreviewGrid";
import GeneratePanel from "./components/GeneratePanel";

/** Preview debounce after the last control change (ms, int). */
const DEBOUNCE_MS = 150;
/** Number of preview tiles (int). */
const TILE_COUNT = 9;
/** Initial fixed preview seeds, list[int] of length TILE_COUNT. */
const INITIAL_SEEDS = Array.from({ length: TILE_COUNT }, (_, i) => i);

/** Errors of any thrown value as list[str]. */
function errorsOf(e: unknown): string[] {
  return e instanceof api.ApiError ? e.errors : [String(e)];
}

export default function App() {
  const [schema, setSchema] = useState<Schema | null>(null); // section -> key -> Leaf
  const [config, setConfig] = useState<Config | null>(null); // current UI config
  const [seeds, setSeeds] = useState<number[]>(INITIAL_SEEDS); // preview seeds (9 ints)
  const [tiles, setTiles] = useState<Tile[]>([]); // latest applied tiles
  const [overlay, setOverlay] = useState(false); // tip markers on/off
  const [errors, setErrors] = useState<string[]>([]); // config / preview errors shown
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
      })
      .catch((e) => setErrors(errorsOf(e)));
  }, []);

  // debounced preview; only the response of the newest request is applied
  useEffect(() => {
    if (!config) return;
    pendingRef.current = true;
    setBusy(true);
    const timer = setTimeout(() => {
      pendingRef.current = false;
      const seq = ++seqRef.current; // this request's number (int)
      setSentSeq(seq);
      api
        .preview(config, seeds)
        .then((t) => {
          if (seq !== seqRef.current) return; // stale: a newer request was sent
          setTiles(t);
          setErrors([]);
        })
        .catch((e) => {
          if (seq !== seqRef.current) return;
          setErrors(errorsOf(e)); // keep the old tiles on screen
        })
        .finally(() => {
          if (seq !== seqRef.current) return;
          setAppliedSeq(seq);
          if (!pendingRef.current) setBusy(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [config, seeds]);

  const setParam = useCallback((section: string, key: string, v: Value) => {
    setConfig((c) => (c ? { ...c, [section]: { ...c[section], [key]: v } } : c));
  }, []);

  const reshuffle = () =>
    setSeeds(Array.from({ length: TILE_COUNT }, () => Math.floor(Math.random() * 2 ** 31)));

  const loadYaml = async (file: File) => {
    try {
      const cfg = await api.parseConfig(await file.text());
      setConfig(cfg);
      setErrors([]);
    } catch (e) {
      setErrors(errorsOf(e));
    }
  };

  const saveYaml = async () => {
    if (!config) return;
    try {
      const text = await api.dumpConfig(config);
      const url = URL.createObjectURL(new Blob([text], { type: "application/x-yaml" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = "v7_config.yaml";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setErrors(errorsOf(e));
    }
  };

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
              const f = e.target.files?.[0];
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
        <PreviewGrid tiles={tiles} imgSize={imgSize} overlay={overlay}
          sentSeq={sentSeq} appliedSeq={appliedSeq} busy={busy} />
        <GeneratePanel config={config} />
      </main>
    </div>
  );
}
