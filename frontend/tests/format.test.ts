import { describe, expect, it } from "vitest";
import { formatTimestamp, percent, signed } from "../lib/format";
import { normalizeState, parseStateMessage } from "../lib/normalize";
import { MARKET_WS_PATH, websocketUrl } from "../lib/connection";
import {
  PRESETS,
  configsEqual,
  presetMatches,
  validateConfig,
} from "../lib/config";
import { DEFAULT_CONFIG } from "../lib/types";
import { parseResetResponse } from "../lib/reset";

describe("format helpers", () => {
  it("formats signed values and percentages consistently", () => {
    expect(signed(0.125)).toBe("+0.13");
    expect(signed(-0.125)).toBe("-0.13");
    expect(percent(0.654)).toBe("65%");
  });

  it("does not invent a timestamp for a disconnected state", () => {
    expect(formatTimestamp("")).toBe("No snapshot yet");
  });
});

describe("state normalization", () => {
  it("normalizes readiness, revisions, and malformed payloads", () => {
    expect(parseStateMessage("not-json")).toBeNull();
    const state = normalizeState({
      health: { ready: true },
      gate: { revision: 7, next_refresh_ms: 420, weights: { microprice: 0.6 } },
      experts: [],
    });
    expect(state?.health.ready).toBe(true);
    expect(state?.gate.revision).toBe(7);
    expect(state?.gate.next_refresh_ms).toBe(420);
    expect(state?.experts).toHaveLength(3);
    expect(state?.experts[0].id).toBe("microprice");
  });

  it("reads feed truth and fallback mode from the backend health contract", () => {
    const state = normalizeState({
      gate: { mode: "uniform-fallback" },
      health: {
        ready: false,
        feed_status: "reconnecting",
        risk_reason: "stale feed",
        run_id: "run-2",
        events_processed: 12,
        late_events_dropped: 3,
        feed_generation: 4,
      },
    });

    expect(state?.gate.mode).toBe("uniform-fallback");
    expect(state?.health.feed_status).toBe("reconnecting");
    expect(state?.health.risk_reason).toBe("stale feed");
    expect(state?.health.run_id).toBe("run-2");
    expect(state?.health.events_processed).toBe(12);
    expect(state?.health.late_events_dropped).toBe(3);
    expect(state?.health.feed_generation).toBe(4);
  });

  it("accepts backend epoch-millisecond timestamps", () => {
    const state = normalizeState({ timestamp: 1_725_350_400_000 });
    expect(state?.timestamp).toBe("2024-09-03T08:00:00.000Z");
  });
});

describe("backend connection contract", () => {
  it("targets the market stream endpoint", () => {
    expect(MARKET_WS_PATH).toBe("/ws/market");
    expect(websocketUrl("http://localhost:8000/")).toBe(
      "ws://localhost:8000/ws/market",
    );
  });
});

describe("draft presets and reset receipts", () => {
  it("keeps presets as draft values until the caller applies them", () => {
    const fast = PRESETS.find((preset) => preset.id === "fast");
    expect(fast?.patch).toEqual({
      gate_mode: "neural",
      gate_interval_ms: 500,
      higher_level_influence: 0.65,
      flow_window_trades: 16,
      expert_strength: 1.25,
    });
    const currentDraft = {
      ...DEFAULT_CONFIG,
      symbol: "ETHUSDT",
      max_inventory: 0.25,
    };
    expect(presetMatches({ ...currentDraft, ...fast!.patch }, fast!)).toBe(
      true,
    );
    expect({ ...currentDraft, ...fast!.patch }.symbol).toBe("ETHUSDT");
    expect(
      configsEqual(DEFAULT_CONFIG, { ...currentDraft, ...fast!.patch }),
    ).toBe(false);
    expect(
      validateConfig({ ...DEFAULT_CONFIG, flow_window_trades: 3 }),
    ).toHaveLength(1);
    expect(
      validateConfig({ ...DEFAULT_CONFIG, symbol: "btc-usdt" })[0],
    ).toContain("5–20 uppercase");
  });

  it("accepts only complete reset run receipts", () => {
    expect(parseResetResponse({ run_id: "run-4", feed_generation: 9 })).toEqual(
      { run_id: "run-4", feed_generation: 9 },
    );
    expect(parseResetResponse({ run_id: "run-4" })).toBeNull();
  });
});
