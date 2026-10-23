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
};

/** State of the single generation job. */
export type JobStatus = {
  state: "idle" | "running" | "done" | "cancelled" | "error";
  done: number; // finished images (int)
  total: number; // requested images (int)
  output_dir: string | null; // absolute output folder or null
  error: string | null; // failure message or null
};

/** Non-2xx API response; errors is the backend's {"errors": [str]} list. */
export class ApiError extends Error {
  status: number; // HTTP status code (int)
  errors: string[]; // one message per problem

  constructor(status: number, errors: string[]) {
    super(errors.join("\n"));
    this.status = status;
    this.errors = errors;
  }
}

/** Send one JSON request; resolve with the parsed body or throw ApiError. */
async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(0, [`network error: ${String(e)}`]);
  }
  // parsed JSON body, or null when the body is not JSON
  let data: any = null;
  try {
    data = await r.json();
  } catch {
    data = null;
  }
  if (!r.ok) {
    const errs = Array.isArray(data?.errors) ? data.errors.map(String) : [`HTTP ${r.status}`];
    throw new ApiError(r.status, errs);
  }
  return data as T;
}

export const getSchema = () => call<Schema>("GET", "/api/schema");

export const getDefaultConfig = async () =>
  (await call<{ config: Config }>("GET", "/api/config/default")).config;

export const parseConfig = async (text: string) =>
  (await call<{ config: Config }>("POST", "/api/config/parse", { text })).config;

export const dumpConfig = async (config: Config) =>
  (await call<{ text: string }>("POST", "/api/config/dump", { config })).text;

export const preview = async (config: Config, seeds: number[]) =>
  (await call<{ tiles: Tile[] }>("POST", "/api/preview", { config, seeds })).tiles;

/** Body of POST /api/generate; workers null = generator default. */
export type GenerateRequest = {
  config: Config;
  output_dir: string; // absolute path
  num_images: number; // int >= 1
  seed: number; // int base seed
  workers: number | null; // int or null
  overwrite: boolean;
};

export const startGenerate = (req: GenerateRequest) =>
  call<{ state: string }>("POST", "/api/generate", req);

export const getJob = () => call<JobStatus>("GET", "/api/jobs/current");

export const cancelJob = () => call<JobStatus>("POST", "/api/jobs/current/cancel");
