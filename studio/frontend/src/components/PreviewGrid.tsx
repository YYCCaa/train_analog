// Preview grid (ceil(sqrt(count)) columns); optional client-side tip marker overlay on each
// tile; a per-tile Save button downloads v7_seed<seed>.zip (PNG + params.json + config.yaml).
import { useState } from "react";
import * as api from "../api";
import type { Config, Tile } from "../api";

type Props = {
  tiles: Tile[]; // latest applied preview tiles (9)
  imgSize: number; // image side in px (int), canvas.img_size
  overlay: boolean; // draw tip markers
  sentSeq: number; // newest preview request number sent (int)
  appliedSeq: number; // request number whose response is shown (int)
  busy: boolean; // debounce pending or latest request in flight
  count: number; // preview tile count, int in [1,16]; sets the column count
  // config the shown tiles were rendered from (section -> key -> Value), null before the
  // first preview; Save bundles with this snapshot, not the current slider state
  appliedConfig: Config | null;
  onSaveErrors: (errs: string[]) => void; // report Save errors (list[str]; [] = success)
};

/** Marker side in CSS px (int); the marker is centred on the tip. */
const MARKER = 18;

/** Preview grid component: the tiles, optional tip markers, and a Save button per tile.
 * Returns the grid as a JSX element. */
export default function PreviewGrid({ tiles, imgSize, overlay, sentSeq, appliedSeq, busy, count,
  appliedConfig, onSaveErrors }: Props) {
  // seeds (list[int]) whose bundle request is in flight; their Save buttons are disabled
  const [saving, setSaving] = useState<number[]>([]);

  // fetch the bundle of seed (int) for appliedConfig and download it via a temporary <a>;
  // (seed: number) => Promise<void>
  const save = async (seed: number) => {
    if (!appliedConfig) return;
    setSaving((s) => [...s, seed]);
    try {
      const blob = await api.bundle(appliedConfig, seed); // zip (Blob)
      const url = URL.createObjectURL(blob); // object URL (str) of the zip
      const a = document.createElement("a"); // temporary download link (HTMLAnchorElement)
      a.href = url;
      a.download = `v7_seed${seed}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      onSaveErrors([]);
    } catch (e) {
      onSaveErrors(e instanceof api.ApiError ? e.errors : [String(e)]);
    } finally {
      setSaving((s) => s.filter((x) => x !== seed));
    }
  };

  return (
    <div
      className={"grid" + (busy ? " busy" : "")}
      data-testid="preview-grid"
      data-sent-seq={sentSeq}
      data-applied-seq={appliedSeq}
      data-busy={busy ? "true" : "false"}
      // ceil(sqrt(count)) equal columns (str)
      style={{ gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(count))}, minmax(0, 1fr))` }}
    >
      {tiles.slice(0, count).map((t, i) => (
        <figure key={i} data-testid="tile" data-seed={t.seed}>
          <div className="frame">
            <img src={`data:image/png;base64,${t.png_b64}`} alt={`seed ${t.seed}`} draggable={false} />
            {overlay && (
              // position in displayed px: (tip + 0.5) / imgSize as a % of the shown
              // image width.  Labels use the pixel-centre convention (pixel i's
              // centre is at i) while on screen pixel i spans [i, i + 1), so the
              // centre of pixel i sits at (i + 0.5) / imgSize of the width.
              <svg
                data-testid="tip-marker"
                className="marker"
                width={MARKER}
                height={MARKER}
                viewBox="0 0 18 18"
                style={{
                  left: `calc(${((t.tip_x + 0.5) / imgSize) * 100}% - ${MARKER / 2}px)`,
                  top: `calc(${((t.tip_y + 0.5) / imgSize) * 100}% - ${MARKER / 2}px)`,
                }}
              >
                <circle cx="9" cy="9" r="6" fill="none" stroke="#ff3b6b" strokeWidth="2" />
                <path d="M9 0v5M9 13v5M0 9h5M13 9h5" stroke="#ff3b6b" strokeWidth="1.5" />
              </svg>
            )}
          </div>
          <figcaption>
            <span>seed {t.seed} · {t.angle_deg.toFixed(0)}°</span>
            <button type="button" className="tile-save" data-testid="tile-save"
              aria-label={`Save image and parameters for seed ${t.seed}`}
              title="Download PNG + params.json + config.yaml"
              disabled={!appliedConfig || saving.includes(t.seed)}
              onClick={() => save(t.seed)}>
              ⤓ Save
            </button>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
