// 3x3 preview grid; optional client-side tip marker overlay on each tile.
import type { Tile } from "../api";

type Props = {
  tiles: Tile[]; // latest applied preview tiles (9)
  imgSize: number; // image side in px (int), canvas.img_size
  overlay: boolean; // draw tip markers
  sentSeq: number; // newest preview request number sent (int)
  appliedSeq: number; // request number whose response is shown (int)
  busy: boolean; // debounce pending or latest request in flight
};

/** Marker side in CSS px (int); the marker is centred on the tip. */
const MARKER = 18;

export default function PreviewGrid({ tiles, imgSize, overlay, sentSeq, appliedSeq, busy }: Props) {
  return (
    <div
      className={"grid" + (busy ? " busy" : "")}
      data-testid="preview-grid"
      data-sent-seq={sentSeq}
      data-applied-seq={appliedSeq}
      data-busy={busy ? "true" : "false"}
    >
      {tiles.map((t, i) => (
        <figure key={i} data-testid="tile" data-seed={t.seed}>
          <div className="frame">
            <img src={`data:image/png;base64,${t.png_b64}`} alt={`seed ${t.seed}`} draggable={false} />
            {overlay && (
              // position in displayed px: tip / imgSize as a % of the shown image width
              <svg
                data-testid="tip-marker"
                className="marker"
                width={MARKER}
                height={MARKER}
                viewBox="0 0 18 18"
                style={{
                  left: `calc(${(t.tip_x / imgSize) * 100}% - ${MARKER / 2}px)`,
                  top: `calc(${(t.tip_y / imgSize) * 100}% - ${MARKER / 2}px)`,
                }}
              >
                <circle cx="9" cy="9" r="6" fill="none" stroke="#ff3b6b" strokeWidth="2" />
                <path d="M9 0v5M9 13v5M0 9h5M13 9h5" stroke="#ff3b6b" strokeWidth="1.5" />
              </svg>
            )}
          </div>
          <figcaption>
            seed {t.seed} · {t.angle_deg.toFixed(0)}°
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
