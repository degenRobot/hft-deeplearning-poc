export type VisualEvent = {
  id: number;
  timestamp_ms: number;
  kind: "book" | "buy" | "sell";
  price: number;
  size: number;
  scores: number[];
  signal: number;
  gate_revision: number;
};
export type VisualWindow = { timestamp_ms: number; values: number[] }[];
export type VisualTelemetry = {
  events: VisualEvent[];
  window: VisualWindow;
  proposed: Record<string, number>;
  previous: Record<string, number>;
  influence: number;
  gate_timestamp_ms: number;
  snapshot_interval_ms: number;
};
const numeric = (n: unknown): n is number =>
  typeof n === "number" && Number.isFinite(n);
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const timestamp = (n: unknown): n is number =>
  numeric(n) && n >= 0 && !Number.isNaN(new Date(n).valueOf());
const vector = (v: unknown, length: number): v is number[] =>
  Array.isArray(v) && v.length === length && v.every(numeric);
const probability = (v: unknown) =>
  object(v) &&
  ["microprice", "flow", "reversion"].every(
    (k) => numeric(v[k]) && v[k] >= 0 && v[k] <= 1,
  ) &&
  Math.abs(Number(v.microprice) + Number(v.flow) + Number(v.reversion) - 1) <
    0.00001;

// Telemetry is optional for older servers. Invalid telemetry never becomes a plausible chart.
export function parseVisual(v: unknown): VisualTelemetry | null {
  if (
    !object(v) ||
    !Array.isArray(v.events) ||
    v.events.length > 48 ||
    !Array.isArray(v.window) ||
    v.window.length > 30 ||
    !probability(v.proposed) ||
    !probability(v.previous) ||
    !numeric(v.influence) ||
    v.influence < 0 ||
    v.influence > 1 ||
    !timestamp(v.gate_timestamp_ms) ||
    !numeric(v.snapshot_interval_ms) ||
    v.snapshot_interval_ms <= 0
  )
    return null;
  if (
    !v.events.every(
      (e, i, all) =>
        object(e) &&
        Number.isSafeInteger(e.id) &&
        Number(e.id) > 0 &&
        timestamp(e.timestamp_ms) &&
        ["book", "buy", "sell"].includes(String(e.kind)) &&
        numeric(e.price) &&
        e.price > 0 &&
        numeric(e.size) &&
        e.size >= 0 &&
        vector(e.scores, 3) &&
        numeric(e.signal) &&
        Number.isSafeInteger(e.gate_revision) &&
        Number(e.gate_revision) >= 0 &&
        (i === 0 ||
          ((e.id as number) > all[i - 1].id &&
            e.timestamp_ms >= all[i - 1].timestamp_ms)),
    )
  )
    return null;
  if (
    !v.window.every(
      (f, i, all) =>
        object(f) &&
        timestamp(f.timestamp_ms) &&
        vector(f.values, 10) &&
        (i === 0 || f.timestamp_ms > all[i - 1].timestamp_ms),
    )
  )
    return null;
  return v as VisualTelemetry;
}

export const FEATURE_ROWS = [
  ["1s return", 0.0001],
  ["5s return", 0.0002],
  ["Volatility", 0.0001],
  ["Spread · bps", 2],
  ["Book imbalance", 1],
  ["Microprice gap", 0.0001],
  ["Trade flow", 1],
  ["Trade count", 10],
  ["Book updates", 10],
  ["Fair-value gap", 0.0002],
] as const;
export const EXPERT_KEYS = ["microprice", "flow", "reversion"] as const;
export const signed = (n: number, digits = 2) =>
  `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
export const eventTime = (ms: number) =>
  new Date(ms).toISOString().slice(11, 23);
