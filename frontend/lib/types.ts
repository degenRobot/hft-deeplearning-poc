export type GateMode = "neural" | "uniform" | "static";
export type FeedSource = "replay" | "binance";

export interface Expert {
  id: string;
  label: string;
  score: number;
  weight: number;
  contribution: number;
}

export interface MarketGateState {
  timestamp: string;
  source: FeedSource;
  symbol: string;
  market: {
    mid: number;
    spread_bps: number;
    imbalance: number;
    last_trade: number;
    trade_flow: number;
  };
  gate: {
    mode: GateMode;
    regime: string;
    weights: { microprice: number; flow: number; reversion: number };
    cadence_ms: number;
    next_refresh_ms: number;
    revision: number;
    model_version: string;
  };
  experts: Expert[];
  quote: { bid: number; ask: number } | null;
  paper: { inventory: number; pnl: number };
  health: { status: string; ready: boolean; message_age_ms: number; reconnects: number };
  feed_status: string;
  risk_reason: string;
  run_id: string;
}

export interface AppConfig {
  source: FeedSource;
  symbol: string;
  gate_mode: GateMode;
  higher_level_influence: number;
  gate_interval_ms: number;
  expert_strength: number;
  base_spread_bps: number;
  max_inventory: number;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export const DEFAULT_CONFIG: AppConfig = {
  source: "replay",
  symbol: "BTCUSDT",
  gate_mode: "neural",
  higher_level_influence: 0.65,
  gate_interval_ms: 1000,
  expert_strength: 1,
  base_spread_bps: 8,
  max_inventory: 0.01,
};

export const EMPTY_STATE: MarketGateState = {
  timestamp: "",
  source: "replay",
  symbol: "BTCUSDT",
  market: { mid: 0, spread_bps: 0, imbalance: 0, last_trade: 0, trade_flow: 0 },
  gate: {
    mode: "neural",
    regime: "awaiting feed",
    weights: { microprice: 0, flow: 0, reversion: 0 },
    cadence_ms: 1000,
    next_refresh_ms: 0,
    revision: -1,
    model_version: "—",
  },
  experts: [
    { id: "microprice", label: "Microprice pressure", score: 0, weight: 0, contribution: 0 },
    { id: "flow", label: "Trade-flow impulse", score: 0, weight: 0, contribution: 0 },
    { id: "reversion", label: "Short reversion", score: 0, weight: 0, contribution: 0 },
  ],
  quote: null,
  paper: { inventory: 0, pnl: 0 },
  health: { status: "waiting", ready: false, message_age_ms: 0, reconnects: 0 },
  feed_status: "waiting",
  risk_reason: "Waiting for a ready backend snapshot",
  run_id: "",
};
