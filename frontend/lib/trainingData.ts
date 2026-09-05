export interface PublicTrainingData {
  selected: {
    id: string;
    label: string;
    source: string;
    symbol: string;
    path: string;
    event_count: number;
    book_count: number;
    trade_count: number;
    bytes: number;
    first_event_ts_ms: number | null;
    last_event_ts_ms: number | null;
    sha256: string | null;
    training_ready: boolean;
    error: string | null;
  };
  capture: {
    id: string | null;
    status: string;
    can_stop: boolean;
    error?: string | null;
    elapsed_seconds?: number;
    requested_seconds?: number;
    progress?: number;
    events?: number;
    bytes?: number;
    updated_at?: string;
  };
}
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
export const ACTIVE_CAPTURE_STATES = ["running", "stopping", "validating"];
const count = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
const validDate = (v: unknown): v is string =>
  typeof v === "string" && Number.isFinite(Date.parse(v));
const nonnegative = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v) && v >= 0;
export function parsePublicTrainingData(v: unknown): PublicTrainingData | null {
  if (
    !object(v) ||
    v.schema_version !== 1 ||
    !object(v.selected) ||
    !object(v.capture)
  )
    return null;
  const s = v.selected,
    c = v.capture;
  if (
    ![s.id, s.label, s.source, s.symbol, s.path].every(
      (x) => typeof x === "string",
    ) ||
    ![s.event_count, s.book_count, s.trade_count, s.bytes].every(count) ||
    ![s.first_event_ts_ms, s.last_event_ts_ms].every(
      (x) =>
        x === null ||
        (nonnegative(x) && Number.isFinite(new Date(x).getTime())),
    ) ||
    !(
      s.sha256 === null ||
      (typeof s.sha256 === "string" && /^[a-f0-9]{64}$/.test(s.sha256))
    ) ||
    typeof s.training_ready !== "boolean" ||
    !(s.error === null || typeof s.error === "string") ||
    ![
      "idle",
      "running",
      "stopping",
      "validating",
      "completed",
      "incomplete",
      "stopped",
      "failed",
    ].includes(String(c.status)) ||
    typeof c.can_stop !== "boolean" ||
    !(c.id === null || typeof c.id === "string") ||
    !(
      c.error === undefined ||
      c.error === null ||
      typeof c.error === "string"
    ) ||
    ![c.elapsed_seconds, c.requested_seconds].every(
      (x) => x === undefined || nonnegative(x),
    ) ||
    ![c.events, c.bytes].every((x) => x === undefined || count(x)) ||
    !(c.updated_at === undefined || validDate(c.updated_at)) ||
    !(c.progress === undefined || (nonnegative(c.progress) && c.progress <= 1))
  )
    return null;
  if (
    (s.first_event_ts_ms === null) !== (s.last_event_ts_ms === null) ||
    (typeof s.first_event_ts_ms === "number" &&
      typeof s.last_event_ts_ms === "number" &&
      s.first_event_ts_ms > s.last_event_ts_ms) ||
    Number(s.book_count) + Number(s.trade_count) !== s.event_count
  )
    return null;
  const active = ACTIVE_CAPTURE_STATES.includes(String(c.status));
  if (c.can_stop && !active) return null;
  if (
    active &&
    (typeof c.id !== "string" ||
      !c.id ||
      !validDate(c.updated_at) ||
      !nonnegative(c.elapsed_seconds) ||
      !count(c.requested_seconds) ||
      c.requested_seconds < 30 ||
      c.requested_seconds > 1800 ||
      !count(c.events) ||
      !count(c.bytes) ||
      !nonnegative(c.progress))
  )
    return null;
  return v as unknown as PublicTrainingData;
}

export const CAPTURE_STALE_MS = 10_000;
export function captureIsStale(
  capture: PublicTrainingData["capture"],
  now: number,
) {
  return (
    ACTIVE_CAPTURE_STATES.includes(capture.status) &&
    (!capture.updated_at ||
      now - Date.parse(capture.updated_at) > CAPTURE_STALE_MS)
  );
}
