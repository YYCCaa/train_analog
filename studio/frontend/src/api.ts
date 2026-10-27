// Typed fetch wrappers for the studio FastAPI backend (all under /api).

/** Schema leaf for one parameter: control type, bounds, step, help text. */
export type Leaf = {
  type: "int" | "float" | "bool" | "int_range" | "float_range"; // control kind (str)
  min: number; // lower bound (for ranges: of both ends)
  max: number; // upper bound (for ranges: of both ends)
  step: number; // slider step
  help: string; // one-line help text
  readonly?: boolean; // true -> control is disabled
};

/** Schema: section name -> key -> Leaf. */
export type Schema = Record<string, Record<string, Leaf>>;

/** One config value: scalar number, bool, or [lo, hi] range (2 numbers). */
export type Value = number | boolean | [number, number];

/** Config: section name -> key -> Value (same nesting as the schema). */
export type Config = Record<string, Record<string, Value>>;

/** One rendered preview tile (256x256 gray PNG + its pointer-tip label). */
export type Tile = {
  seed: number; // image seed (int)
  png_b64: string; // base64 PNG ("L" mode, img_size x img_size)
  tip_x: number; // tip x in image px (float)
  tip_y: number; // tip y in image px (float)
  angle_deg: number; // pointer angle in degrees (float)
  // the sampled per-image values, grouped by section: section name (str) -> key -> JSON value
  params: Record<string, unknown>;
};

/** State of the single generation job. */
export type JobStatus = {
  state: "idle" | "running" | "done" | "cancelled" | "error"; // job phase (str enum)
  done: number; // finished images (int)
  total: number; // requested images (int)
  output_dir: string | null; // absolute output folder or null
  error: string | null; // failure message or null
};

/** Non-2xx API response, network failure (status 0) or timeout (status 0). */
export class ApiError extends Error {
  status: number; // HTTP status code (int); 0 = no HTTP response
  errors: string[]; // one message per problem (the backend's {"errors": [str]})

  constructor(status: number, errors: string[]) {
    super(errors.join("\n"));
    this.status = status;
    this.errors = errors;
  }
}

/** Default per-request timeout (ms, int); a hung backend becomes an error. */
const TIMEOUT_MS = 30_000;

/** Send one JSON request; resolve with the parsed body or throw ApiError.
 * Aborts after timeoutMs (ms, int) and throws a "timed out" ApiError. */
async function call<T>(method: "GET" | "POST", path: string, body?: unknown,
  timeoutMs: number = TIMEOUT_MS): Promise<T> {
  const ctl = new AbortController(); // aborts the fetch on timeout
  const timer = setTimeout(() => ctl.abort(), timeoutMs); // timeout handle (number)
  try {
    let r: Response; // HTTP response
    try {
      r = await fetch(path, {
        method,
        signal: ctl.signal,
        // every POST is JSON: the backend refuses other POSTs with 415 (cross-site guard)
        headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
        body: method === "POST" ? JSON.stringify(body ?? {}) : undefined,
      });
    } catch (e) {
      if (ctl.signal.aborted) {
        throw new ApiError(0, [`${path}: request timed out after ${timeoutMs / 1000} s`]);
      }
      throw new ApiError(0, [`${path}: network error: ${String(e)}`]);
    }
    let data: any = null; // parsed JSON body, or null when the body is not JSON
    try {
      data = await r.json();
    } catch {
      data = null;
    }
    if (!r.ok) {
      // error messages (list[str]) from the body, or a generic HTTP status line
      const errs = Array.isArray(data?.errors) ? data.errors.map(String) : [`HTTP ${r.status}`];
      throw new ApiError(r.status, errs);
    }
    return data as T;
  } finally {
    clearTimeout(timer);
  }
}

/** GET the schema -> Promise<Schema>. */
export const getSchema = () => call<Schema>("GET", "/api/schema");

/** GET the default config -> Promise<Config>. */
export const getDefaultConfig = async () =>
  (await call<{ config: Config }>("GET", "/api/config/default")).config;

/** Result of parsing a YAML text: the migrated config plus what had to be adjusted. */
export interface ParsedConfig {
  config: Config; // migrated + validated config, section -> key -> Value
  warnings: string[]; // one "section.key: message" per adjustment to an old YAML; [] if none
}

/** YAML text (str) -> Promise<ParsedConfig> (old keys migrated, defaults filled, values
 * repaired); only YAML syntax errors / non-mapping documents -> ApiError (422). */
export const parseConfig = async (text: string): Promise<ParsedConfig> => {
  const r = await call<{ config: Config; warnings?: string[] }>(
    "POST", "/api/config/parse", { text });
  return { config: r.config, warnings: Array.isArray(r.warnings) ? r.warnings.map(String) : [] };
};

/** Config -> Promise<string[]>: one "section.key: message" per too-tight pointer gray
 * range (no headroom, < 32 levels, contrast fallback share); [] when fine. */
export const checkConfig = async (config: Config): Promise<string[]> => {
  const r = await call<{ warnings?: string[] }>("POST", "/api/config/check", { config });
  return Array.isArray(r.warnings) ? r.warnings.map(String) : [];
};

/** Config -> Promise<str> YAML text; invalid config -> ApiError. */
export const dumpConfig = async (config: Config) =>
  (await call<{ text: string }>("POST", "/api/config/dump", { config })).text;

/** Render config for seeds (1..16 ints) -> Promise<Tile[]> (same order as seeds). */
export const preview = async (config: Config, seeds: number[]) =>
  (await call<{ tiles: Tile[] }>("POST", "/api/preview", { config, seeds })).tiles;

/** Bundle one image: POST /api/bundle {config, seed int} -> Promise<Blob> of the zip
 * (v7_seed<seed>.png + params.json + config.yaml). Same timeout / error style as `call`:
 * non-2xx -> ApiError with the server's {"errors": [str]}, network failure or timeout ->
 * ApiError status 0. */
export async function bundle(config: Config, seed: number,
  timeoutMs: number = TIMEOUT_MS): Promise<Blob> {
  const path = "/api/bundle"; // endpoint (str)
  const ctl = new AbortController(); // aborts the fetch on timeout
  const timer = setTimeout(() => ctl.abort(), timeoutMs); // timeout handle (number)
  try {
    let r: Response; // HTTP response
    try {
      r = await fetch(path, {
        method: "POST",
        signal: ctl.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config, seed }),
      });
    } catch (e) {
      if (ctl.signal.aborted) {
        throw new ApiError(0, [`${path}: request timed out after ${timeoutMs / 1000} s`]);
      }
      throw new ApiError(0, [`${path}: network error: ${String(e)}`]);
    }
    if (!r.ok) {
      let data: any = null; // parsed JSON error body, or null when it is not JSON
      try {
        data = await r.json();
      } catch {
        data = null;
      }
      // error messages (list[str]) from the body, or a generic HTTP status line
      const errs = Array.isArray(data?.errors) ? data.errors.map(String) : [`HTTP ${r.status}`];
      throw new ApiError(r.status, errs);
    }
    try {
      return await r.blob();
    } catch (e) {
      if (ctl.signal.aborted) {
        throw new ApiError(0, [`${path}: request timed out after ${timeoutMs / 1000} s`]);
      }
      throw new ApiError(0, [`${path}: network error: ${String(e)}`]);
    }
  } finally {
    clearTimeout(timer);
  }
}

/** Body of POST /api/generate; workers null = backend default (all cores but one). */
export type GenerateRequest = {
  config: Config; // render config to generate with
  output_dir: string; // absolute path
  num_images: number; // int, 1..1_000_000
  seed: number; // int base seed
  workers: number | null; // int 1..cpu count, or null = auto
  overwrite: boolean; // true -> replace existing dial_*.png / labels.csv / params.jsonl / config.yaml
  save_params: boolean; // true -> also write params.jsonl (every image's sampled parameters)
};

/** Start the generation job -> Promise<{state: "running"}>; 400/409/422 -> ApiError. */
export const startGenerate = (req: GenerateRequest) =>
  call<{ state: string }>("POST", "/api/generate", req);

/** Current job status -> Promise<JobStatus>. */
export const getJob = () => call<JobStatus>("GET", "/api/jobs/current");

/** Request cancel of the running job -> Promise<JobStatus> (status at that moment). */
export const cancelJob = () => call<JobStatus>("POST", "/api/jobs/current/cancel");
