import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  ActivationNetwork,
  activationOpacity,
} from "../components/ActivationNetwork";
import { CandleChart } from "../components/CandleChart";
import { parseVisual } from "../lib/visual";

const candle = {
  timestamp_ms: 1000,
  open: 100,
  high: 103,
  low: 98,
  close: 101,
  volume: 2,
};
const telemetry = {
  events: [],
  window: [],
  proposed: { microprice: 0.5, flow: 0.3, reversion: 0.2 },
  previous: { microprice: 0.5, flow: 0.3, reversion: 0.2 },
  influence: 0.35,
  gate_timestamp_ms: 1000,
  snapshot_interval_ms: 100,
  candles: [candle],
  activations: { hidden_1: Array(64).fill(0.3), hidden_2: Array(32).fill(0) },
};
describe("terminal observations", () => {
  it("rejects impossible candle prices and invalid activation vectors", () => {
    expect(parseVisual(telemetry)).not.toBeNull();
    for (const candles of [
      [{ ...candle, high: 99 }],
      [{ ...candle, volume: -1 }],
      [candle, candle],
      Array(91).fill(candle),
    ])
      expect(parseVisual({ ...telemetry, candles })).toBeNull();
    for (const activations of [
      { hidden_1: [0.3], hidden_2: Array(32).fill(0) },
      { ...telemetry.activations, hidden_2: Array(32).fill(-1) },
      { ...telemetry.activations, hidden_1: Array(64).fill(Infinity) },
    ])
      expect(parseVisual({ ...telemetry, activations })).toBeNull();
  });
  it("renders all 96 actual hidden values and scales zero and nonzero activity distinctly", () => {
    const html = renderToStaticMarkup(
      createElement(ActivationNetwork, {
        mode: "neural",
        activations: telemetry.activations,
        outputs: [0.5, 0.3, 0.2],
        revision: 7,
      }),
    );
    expect(html.match(/data-activation=/g)).toHaveLength(96);
    expect(html).toContain('data-activation="0.3"');
    expect(html).toContain('data-activation="0"');
    expect(html).toContain("Actual ReLU activations");
    expect(activationOpacity(0, 0)).toBe(0.08);
    expect(activationOpacity(2, 2)).toBe(1);
    expect(activationOpacity(0.1, 2)).toBeLessThan(activationOpacity(1, 2));
    const bypass = renderToStaticMarkup(
      createElement(ActivationNetwork, {
        mode: "uniform",
        activations: telemetry.activations,
        outputs: [0.5, 0.3, 0.2],
        revision: 7,
      }),
    );
    expect(bypass).not.toContain("data-activation=");
  });
  it("shows real OHLC, gaps, and a forming-candle label; empty data is not a price", () => {
    const html = renderToStaticMarkup(
      createElement(CandleChart, {
        symbol: "BTCUSDT",
        candles: [candle, { ...candle, timestamp_ms: 3000 }],
      }),
    );
    expect(html).toContain("2 observed one-second trade candles");
    expect(html).toContain("H 103.00");
    expect(html).toContain("L 98.00");
    expect(html).toContain("Latest candle may be forming");
    const empty = renderToStaticMarkup(
      createElement(CandleChart, { symbol: "BTCUSDT", candles: [] }),
    );
    expect(empty).toContain("Waiting for trades");
    expect(empty).not.toContain("100.00");
  });
});
