// One control per config leaf: slider(s) + number input(s), or a checkbox.
import { useEffect, useState } from "react";
import type { Leaf, Value } from "../api";

/** Display a number without float noise (e.g. 0.30000000000000004 -> "0.3"). */
function fmt(v: number): string {
  return String(Number(v.toPrecision(10)));
}

/** Clamp n (number) into [lo, hi]. */
function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

type NumberFieldProps = {
  value: number; // committed value (number)
  min: number; // lowest allowed value (number): schema min, or lo for a hi box
  max: number; // highest allowed value (number): schema max, or hi for a lo box
  step: number; // spinner step (number)
  isInt: boolean; // true -> integers only
  handle?: "lo" | "hi"; // range end, if part of a range control
  disabled: boolean; // true -> read-only field
  onCommit: (v: number) => void; // called with a parsed, clamped value
};

/** Number input that keeps a local draft and commits on Enter or blur. */
function NumberField({ value, min, max, step, isInt, handle, disabled, onCommit }: NumberFieldProps) {
  // text currently typed in the box (str); resynced whenever value changes
  const [draft, setDraft] = useState(fmt(value));
  useEffect(() => setDraft(fmt(value)), [value]);

  // parse the draft, clamp it into [min, max] and commit if it changed; () => void
  const commit = () => {
    const n = isInt ? parseInt(draft, 10) : parseFloat(draft); // parsed draft (number|NaN)
    if (Number.isNaN(n)) {
      setDraft(fmt(value));
      return;
    }
    const c = clamp(isInt ? Math.round(n) : n, min, max); // value to commit (number)
    if (c !== value) onCommit(c);
    setDraft(fmt(c));
  };

  return (
    <input
      type="number"
      className="num"
      data-handle={handle}
      min={min}
      max={max}
      step={step}
      value={draft}
      disabled={disabled}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
      }}
    />
  );
}

type Props = {
  section: string; // section name (str), e.g. "ticks"
  name: string; // key inside the section (str), e.g. "len_range"
  leaf: Leaf; // schema leaf: type, min/max/step, help, readonly
  value: Value; // current value: number | boolean | [lo, hi] (2 numbers)
  onChange: (v: Value) => void; // receives the new value for this leaf
};

/** Control for section.name, rendered according to leaf.type. */
export default function ParamControl({ section, name, leaf, value, onChange }: Props) {
  const disabled = !!leaf.readonly; // read-only leaf -> every input disabled (bool)
  const isInt = leaf.type === "int" || leaf.type === "int_range"; // integer-valued (bool)
  // slider string -> number (int or float per leaf type); (s: str) => number
  const parse = (s: string) => (isInt ? parseInt(s, 10) : parseFloat(s));
  // value -> position on the schema span in % (0..100); fills the slider track
  const pct = (v: number) => ((v - leaf.min) / (leaf.max - leaf.min || 1)) * 100;

  let body: JSX.Element; // the inputs for this leaf type (JSX)
  if (leaf.type === "bool") {
    body = (
      <label className="bool">
        <input
          type="checkbox"
          checked={value as boolean}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>{(value as boolean) ? "on" : "off"}</span>
      </label>
    );
  } else if (leaf.type === "int_range" || leaf.type === "float_range") {
    const [lo, hi] = value as [number, number]; // current range ends (numbers)
    // lo in the right half -> lo slider on top, so a lo == hi == max pair can still
    // be pulled apart leftwards (hi on top keeps working for lo == hi == min)
    const loOnTop = pct(lo) > 50; // (bool)
    body = (
      <div className="range-row">
        <NumberField value={lo} min={leaf.min} max={hi} step={leaf.step} isInt={isInt}
          handle="lo" disabled={disabled} onCommit={(v) => onChange([v, hi])} />
        <div className="dual" style={{ ["--a" as any]: `${pct(lo)}%`, ["--b" as any]: `${pct(hi)}%` }}>
          <input type="range" data-handle="lo" className={loOnTop ? "on-top" : undefined}
            min={leaf.min} max={leaf.max} step={leaf.step} value={lo} disabled={disabled}
            onChange={(e) => onChange([Math.min(parse(e.target.value), hi), hi])} />
          <input type="range" data-handle="hi" min={leaf.min} max={leaf.max} step={leaf.step}
            value={hi} disabled={disabled}
            onChange={(e) => onChange([lo, Math.max(parse(e.target.value), lo)])} />
        </div>
        <NumberField value={hi} min={lo} max={leaf.max} step={leaf.step} isInt={isInt}
          handle="hi" disabled={disabled} onCommit={(v) => onChange([lo, v])} />
      </div>
    );
  } else {
    const v = value as number; // current scalar value (number)
    body = (
      <div className="scalar-row">
        <input type="range" className="single" min={leaf.min} max={leaf.max} step={leaf.step}
          value={v} disabled={disabled}
          style={{ ["--b" as any]: `${pct(v)}%` }}
          onChange={(e) => onChange(parse(e.target.value))} />
        <NumberField value={v} min={leaf.min} max={leaf.max} step={leaf.step} isInt={isInt}
          disabled={disabled} onCommit={onChange} />
      </div>
    );
  }

  return (
    <div className={"param" + (disabled ? " readonly" : "")} data-param={`${section}.${name}`}>
      <div className="param-name">
        {name}
        {disabled && <span className="tag">read-only</span>}
      </div>
      {body}
      <div className="help">{leaf.help}</div>
    </div>
  );
}
