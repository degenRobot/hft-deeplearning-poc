import { EMPTY_STATE, type MarketGateState } from "./types";

const finite = (value: unknown, fallback = 0) => {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const text = (value: unknown, fallback: string) =>
  typeof value === "string" && value.length > 0 ? value : fallback;

const record = (value: unknown) =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
const nonNegative = (value: unknown, fallback = 0) =>
  Math.max(0, finite(value, fallback));

export function normalizeState(input: unknown): MarketGateState | null {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const raw = record(input);
  const rawMarket = record(raw.market);
  const rawGate = record(raw.gate);
  const rawWeights = record(rawGate.weights);
  const rawPaper = record(raw.paper);
  const rawHealth = record(raw.health);
  const rawExperts = Array.isArray(raw.experts) ? raw.experts : [];

  const experts = EMPTY_STATE.experts.map((placeholder, index) => {
    const candidate = record(rawExperts[index]);
    return {
      id: text(candidate.id, placeholder.id),
      label: text(candidate.label, placeholder.label),
      score: finite(candidate.score),
      weight: finite(candidate.weight),
      contribution: finite(candidate.contribution),
    };
  });

  const timestamp =
    typeof raw.timestamp === "number" &&
    Number.isFinite(raw.timestamp) &&
    raw.timestamp > 0
      ? Number.isNaN(new Date(raw.timestamp).valueOf())
        ? ""
        : new Date(raw.timestamp).toISOString()
      : text(raw.timestamp, "");
  const nextRefresh = Math.max(
    0,
    finite(
      rawGate.next_refresh_ms,
      finite(raw.next_refresh_ms, finite(rawGate.cadence_ms, 1000)),
    ),
  );

  return {
    timestamp,
    source: raw.source === "binance" ? "binance" : "replay",
    symbol: text(raw.symbol, EMPTY_STATE.symbol),
    market: {
      mid: finite(rawMarket.mid),
      spread_bps: finite(rawMarket.spread_bps),
      imbalance: finite(rawMarket.imbalance),
      last_trade: finite(rawMarket.last_trade),
      trade_flow: finite(rawMarket.trade_flow),
    },
    gate: {
      mode:
        rawGate.mode === "uniform" ||
        rawGate.mode === "static" ||
        rawGate.mode === "uniform-fallback"
          ? rawGate.mode
          : "neural",
      regime: text(rawGate.regime, EMPTY_STATE.gate.regime),
      weights: {
        microprice: Math.max(0, finite(rawWeights.microprice)),
        flow: Math.max(0, finite(rawWeights.flow)),
        reversion: Math.max(0, finite(rawWeights.reversion)),
      },
      cadence_ms: Math.max(1, finite(rawGate.cadence_ms, 1000)),
      next_refresh_ms: nextRefresh,
      revision: finite(rawGate.revision, -1),
      model_version: text(rawGate.model_version, "—"),
    },
    experts,
    quote:
      raw.quote && typeof raw.quote === "object"
        ? {
            bid: finite(record(raw.quote).bid),
            ask: finite(record(raw.quote).ask),
          }
        : null,
    paper: { inventory: finite(rawPaper.inventory), pnl: finite(rawPaper.pnl) },
    health: {
      status: text(rawHealth.status, "unknown"),
      ready: rawHealth.ready === true,
      feed_status: text(
        rawHealth.feed_status,
        text(raw.feed_status, text(rawHealth.status, "unknown")),
      ),
      feed_error: text(rawHealth.feed_error, text(raw.feed_error, "")),
      risk_reason: text(rawHealth.risk_reason, text(raw.risk_reason, "")),
      run_id: text(rawHealth.run_id, text(raw.run_id, "")),
      message_age_ms: nonNegative(rawHealth.message_age_ms),
      reconnects: nonNegative(rawHealth.reconnects),
      events_processed: nonNegative(
        rawHealth.events_processed,
        finite(raw.events_processed),
      ),
      late_events_dropped: nonNegative(
        rawHealth.late_events_dropped,
        finite(raw.late_events_dropped),
      ),
      feed_generation: nonNegative(
        rawHealth.feed_generation,
        finite(raw.feed_generation),
      ),
    },
  };
}

export function parseStateMessage(message: string): MarketGateState | null {
  try {
    return normalizeState(JSON.parse(message));
  } catch {
    return null;
  }
}
