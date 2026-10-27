// Generate-to-disk panel: run settings, start/cancel, progress polled every 500 ms.
import { useEffect, useState } from "react";
import { ApiError, cancelJob, getJob, startGenerate } from "../api";
import type { Config, JobStatus } from "../api";

/** Job poll interval while running (ms, int). */
const POLL_MS = 500;

type Props = { config: Config | null }; // current UI config sent with Start

/** Parse a typed whole number. text (str) as typed; label (str) names the field.
 * -> {value: int} or {error: str}. Uses Number() + Number.isInteger, so "abc",
 * "" and "2.5" are refused instead of becoming 0 or being truncated, and "1e3"
 * reads as 1000 (not parseInt's 1). */
function parseWhole(text: string, label: string): { value: number } | { error: string } {
  const t = text.trim(); // typed text without surrounding blanks (str)
  const n = Number(t); // numeric value (float | NaN); Number("") is 0, so "" is checked first
  if (t === "" || !Number.isInteger(n)) {
    return { error: `${label}: must be a whole number, got "${text}"` };
  }
  return { value: n };
}

/** Generate-to-disk panel component. Props: config (Config | null), the current
 * UI config sent with Start. Holds the typed run settings (all strings until
 * Start parses them), starts / cancels the single backend job, and polls its
 * status every POLL_MS while it runs. Returns the panel as a JSX element. */
export default function GeneratePanel({ config }: Props) {
  const [outputDir, setOutputDir] = useState(""); // absolute folder (str)
  const [numImages, setNumImages] = useState("1000"); // image count, as typed (str)
  const [seed, setSeed] = useState("0"); // base seed, as typed (str)
  const [workers, setWorkers] = useState(""); // worker count as typed; "" = auto (str)
  const [overwrite, setOverwrite] = useState(false); // replace old dataset files
  const [saveParams, setSaveParams] = useState(false); // also write params.jsonl (bool)
  const [job, setJob] = useState<JobStatus | null>(null); // last polled job status
  const [message, setMessage] = useState<string[]>([]); // start/cancel errors

  // fetch the current job once on mount (a job may already be running)
  useEffect(() => {
    getJob().then(setJob).catch(() => undefined);
  }, []);

  const running = job?.state === "running"; // a job is in progress (bool)

  // poll while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => { // poll interval handle (number)
      getJob().then(setJob).catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(id);
  }, [running]);

  // POST /api/generate with the typed settings; shows 400/409/422 errors; () => Promise<void>
  const start = async () => {
    if (!config) return;
    setMessage([]);
    const n = parseWhole(numImages, "Images"); // requested image count, or its error
    const s = parseWhole(seed, "Seed"); // base seed, or its error
    // worker count, or null when blank (backend picks all cores but one)
    const w = workers.trim() === "" ? null : parseWhole(workers, "Workers");
    // input errors (list[str]); any one refuses Start before a request is sent
    const bad = [n, s, w].flatMap((p) => (p !== null && "error" in p ? [p.error] : []));
    if (bad.length || !("value" in n) || !("value" in s)) {
      setMessage(bad);
      return;
    }
    try {
      await startGenerate({
        config,
        output_dir: outputDir.trim(),
        num_images: n.value,
        seed: s.value,
        workers: w !== null && "value" in w ? w.value : null,
        overwrite,
        save_params: saveParams,
      });
      setJob({ state: "running", done: 0, total: n.value, output_dir: outputDir.trim(),
        error: null });
    } catch (e) {
      setMessage(e instanceof ApiError ? e.errors : [String(e)]);
    }
  };

  // POST cancel; the running poll then picks up "cancelled"; () => Promise<void>
  const cancel = async () => {
    try {
      setJob(await cancelJob());
    } catch (e) {
      setMessage(e instanceof ApiError ? e.errors : [String(e)]);
    }
  };

  return (
    <section className="gen">
      <h2>Generate dataset</h2>
      <div className="gen-grid">
        <label className="wide">
          Output folder (absolute)
          <input data-testid="gen-output-dir" type="text" value={outputDir}
            placeholder="C:\data\sv7_new" onChange={(e) => setOutputDir(e.target.value)} />
        </label>
        <label>
          Images
          {/* text, not number: the browser would hide bad input as "" */}
          <input data-testid="gen-num-images" type="text" inputMode="numeric" value={numImages}
            onChange={(e) => setNumImages(e.target.value)} />
        </label>
        <label>
          Seed
          <input data-testid="gen-seed" type="text" inputMode="numeric" value={seed}
            onChange={(e) => setSeed(e.target.value)} />
        </label>
        <label>
          Workers
          <input data-testid="gen-workers" type="text" inputMode="numeric" value={workers}
            placeholder="auto" onChange={(e) => setWorkers(e.target.value)} />
        </label>
        <label className="bool">
          <input data-testid="gen-overwrite" type="checkbox" checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)} />
          overwrite
        </label>
        <label className="bool"
          title="also write params.jsonl: every image's sampled parameters (~3 KB per image)">
          <input data-testid="gen-save-params" type="checkbox" checked={saveParams}
            onChange={(e) => setSaveParams(e.target.checked)} />
          save parameters
        </label>
      </div>
      <div className="gen-actions">
        <button data-testid="gen-start" className="primary" disabled={running || !config}
          onClick={start}>
          Start
        </button>
        <button data-testid="gen-cancel" disabled={!running} onClick={cancel}>
          Cancel
        </button>
        <progress data-testid="gen-progress" max={Math.max(job?.total ?? 0, 1)}
          value={job?.done ?? 0} />
      </div>
      <div data-testid="gen-status" className={"gen-status state-" + (job?.state ?? "idle")}>
        <span className="state">{job?.state ?? "idle"}</span>
        {job && job.total > 0 && (
          <span>
            {" "}— {job.done}/{job.total}
            {job.output_dir ? ` → ${job.output_dir}` : ""}
          </span>
        )}
        {job?.error && <div className="err">{job.error}</div>}
        {message.map((m, i) => (
          <div key={i} className="err">{m}</div>
        ))}
      </div>
    </section>
  );
}
