import { describe, expect, it } from "vitest";
import { formatTimestamp, percent, signed } from "../lib/format";
import { normalizeState, parseStateMessage } from "../lib/normalize";
import { MARKET_WS_PATH, websocketUrl } from "../lib/connection";

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
  it("clamps confidence and ignores malformed payloads", () => {
    expect(parseStateMessage("not-json")).toBeNull();
    const state = normalizeState({ gate: { confidence: 5, weights: { microprice: 0.6 } }, experts: [] });
    expect(state?.gate.confidence).toBe(1);
    expect(state?.experts).toHaveLength(3);
    expect(state?.experts[0].id).toBe("microprice");
  });

  it("accepts backend epoch-millisecond timestamps", () => {
    const state = normalizeState({ timestamp: 1_725_350_400_000 });
    expect(state?.timestamp).toBe("2024-09-03T08:00:00.000Z");
  });
});

describe("backend connection contract", () => {
  it("targets the market stream endpoint", () => {
    expect(MARKET_WS_PATH).toBe("/ws/market");
    expect(websocketUrl("http://localhost:8000/")).toBe("ws://localhost:8000/ws/market");
  });
});
