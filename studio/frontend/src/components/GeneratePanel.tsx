// Generate-to-disk panel: run settings, start/cancel, progress polled every 500 ms.
import { useEffect, useState } from "react";
import { ApiError, cancelJob, getJob, startGenerate } from "../api";
import type { Config, JobStatus } from "../api";

/** Job poll interval while running (ms, int). */
const POLL_MS = 500;

type Props = { config: Config | null }; // current UI config sent with Start

export default function GeneratePanel({ config }: Props) {
  const [outputDir, setOutputDir] = useState(""); // absolute folder (str)
  const [numImages, setNumImages] = useState("1000"); // image count, as typed (str)
  const [seed, setSeed] = useState("0"); // base seed, as typed (str)
  const [workers, setWorkers] = useState(""); // worker count as typed; "" = auto (str)
  const [overwrite, setOverwrite] = useState(false); // replace old dataset files
  const [job, setJob] = useState<JobStatus | null>(null); // last polled job status
  const [message, setMessage] = useState<string[]>([]); // start/cancel errors

  // fetch the current job once on mount (a job may already be running)
  useEffect(() => {
    getJob().then(setJob).catch(() => undefined);
  }, []);

  const running = job?.state === "running";

  // poll while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      getJob().then(setJob).catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(id);
  }, [running]);

  const start = async () => {
    if (!config) return;
    setMessage([]);
    const n = parseInt(numImages, 10);
    const w = workers.trim() === "" ? null : parseInt(workers, 10);
    try {
      await startGenerate({
        config,
        output_dir: outputDir.trim(),
        num_images: Number.isNaN(n) ? 0 : n,
        seed: parseInt(seed, 10) || 0,
        workers: w === null || Number.isNaN(w) ? null : w,
        overwrite,
      });
      setJob({ state: "running", done: 0, total: n, output_dir: outputDir.trim(), error: null });
    } catch (e) {
      setMessage(e instanceof ApiError ? e.errors : [String(e)]);
    }
  };

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
          <input data-testid="gen-num-images" type="number" min={1} value={numImages}
            onChange={(e) => setNumImages(e.target.value)} />
        </label>
        <label>
          Seed
          <input data-testid="gen-seed" type="number" value={seed}
            onChange={(e) => setSeed(e.target.value)} />
        </label>
        <label>
          Workers
          <input data-testid="gen-workers" type="number" min={1} value={workers}
            placeholder="auto" onChange={(e) => setWorkers(e.target.value)} />
        </label>
        <label className="bool">
          <input data-testid="gen-overwrite" type="checkbox" checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)} />
          overwrite
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
