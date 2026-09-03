import { EMPTY_STATE, type MarketGateState } from "./types";

const finite = (value: unknown, fallback = 0) => {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const text = (value: unknown, fallback: string) =>
  typeof value === "string" && value.length > 0 ? value : fallback;

export function normalizeState(input: unknown): MarketGateState | null {
  if (!input || typeof input !== "object") return null;
  const raw = input as Record<string, unknown>;
  const rawMarket = (
    raw.market && typeof raw.market === "object" ? raw.market : {}
  ) as Record<string, unknown>;
  const rawGate = (
    raw.gate && typeof raw.gate === "object" ? raw.gate : {}
  ) as Record<string, unknown>;
  const rawWeights = (
    rawGate.weights && typeof rawGate.weights === "object"
      ? rawGate.weights
      : {}
  ) as Record<string, unknown>;
  const rawPaper = (
    raw.paper && typeof raw.paper === "object" ? raw.paper : {}
  ) as Record<string, unknown>;
  const rawHealth = (
    raw.health && typeof raw.health === "object" ? raw.health : {}
  ) as Record<string, unknown>;
  const rawExperts = Array.isArray(raw.experts) ? raw.experts : [];

  const experts = EMPTY_STATE.experts.map((placeholder, index) => {
    const candidate = (
      rawExperts[index] && typeof rawExperts[index] === "object"
        ? rawExperts[index]
        : {}
    ) as Record<string, unknown>;
    return {
      id: text(candidate.id, placeholder.id),
      label: text(candidate.label, placeholder.label),
      score: finite(candidate.score),
      weight: finite(candidate.weight),
      contribution: finite(candidate.contribution),
    };
  });

  const timestamp =
    typeof raw.timestamp === "number" && Number.isFinite(raw.timestamp)
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
            bid: finite((raw.quote as Record<string, unknown>).bid),
            ask: finite((raw.quote as Record<string, unknown>).ask),
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
      risk_reason: text(rawHealth.risk_reason, text(raw.risk_reason, "")),
      run_id: text(rawHealth.run_id, text(raw.run_id, "")),
      message_age_ms: Math.max(0, finite(rawHealth.message_age_ms)),
      reconnects: Math.max(0, finite(rawHealth.reconnects)),
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
