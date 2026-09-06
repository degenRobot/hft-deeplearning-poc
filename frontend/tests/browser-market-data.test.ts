import { describe, expect, it } from "vitest";
import {
  collectMarketCandles,
  capturedReturns,
  liveSgdDataset,
  MAX_CAPTURE_CANDLES,
  type BrowserCapture,
} from "../lib/browserMarketData";
import { initialSgdState, sgdStep } from "../lib/sgdDemo";
import { EMPTY_STATE, type MarketGateState } from "../lib/types";
import type { Candle } from "../lib/visual";

const candles = (length: number): Candle[] =>
  Array.from({ length }, (_, i) => {
    const price = 100 + 0.05 * Math.sin(i * 0.7) + i * 0.001;
    return {
      timestamp_ms: i * 1000,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: 1,
    };
  });
const capture = (length = 90): BrowserCapture => ({
  key: "run:1:BTCUSDT",
  symbol: "BTCUSDT",
  candles: candles(length),
});
const snapshot = (cs = candles(90)): MarketGateState => ({
  ...EMPTY_STATE,
  source: "binance",
  symbol: "BTCUSDT",
  health: {
    ...EMPTY_STATE.health,
    ready: true,
    run_id: "run",
    feed_generation: 1,
  },
  visual: {
    candles: cs,
    events: [],
    window: [],
    proposed: { microprice: 1 / 3, flow: 1 / 3, reversion: 1 / 3 },
    previous: { microprice: 1 / 3, flow: 1 / 3, reversion: 1 / 3 },
    influence: 0.5,
    gate_timestamp_ms: 0,
    snapshot_interval_ms: 100,
  },
});

describe("browser candle capture", () => {
  it("excludes the open candle and does not duplicate repeated UI snapshots", () => {
    const first = collectMarketCandles(null, snapshot())!;
    expect(first.candles).toHaveLength(89);
    expect(first.candles.at(-1)!.timestamp_ms).toBe(88000);
    expect(collectMarketCandles(first, snapshot())).toBe(first);
    expect(
      collectMarketCandles(first, snapshot(candles(91)))!.candles,
    ).toHaveLength(90);
  });
  it("clears replay/stale data and isolates feed runs, symbols and generations", () => {
    const first = collectMarketCandles(null, snapshot())!;
    expect(
      collectMarketCandles(first, { ...snapshot(), source: "replay" }),
    ).toBeNull();
    expect(
      collectMarketCandles(first, {
        ...snapshot(),
        health: { ...snapshot().health, ready: false },
      }),
    ).toBeNull();
    for (const next of [
      { ...snapshot(candles(4)), symbol: "ETHUSDT" },
      {
        ...snapshot(candles(4)),
        health: { ...snapshot().health, run_id: "new" },
      },
      {
        ...snapshot(candles(4)),
        health: { ...snapshot().health, feed_generation: 2 },
      },
    ])
      expect(collectMarketCandles(first, next)!.candles).toHaveLength(3);
  });
  it("bounds the browser buffer and never invents returns across missing seconds", () => {
    const bounded = collectMarketCandles(null, snapshot(candles(500)))!;
    expect(bounded.candles).toHaveLength(MAX_CAPTURE_CANDLES);
    const withGap = capture(6);
    withGap.candles.splice(2, 1);
    const rows = capturedReturns(withGap);
    expect(rows).toHaveLength(1);
    expect(rows[0].timestamp).toBe(5000);
  });
  it("requires enough real examples and freezes a chronological training/holdout split", () => {
    expect(liveSgdDataset(capture(41))).toBeNull();
    const source = capture();
    const dataset = liveSgdDataset(source)!;
    const raw = capturedReturns(source);
    const split = Math.floor(raw.length * 0.8);
    expect(dataset.points).toHaveLength(split);
    expect(dataset.validation).toHaveLength(raw.length - split - 1);
    expect(dataset.validation[0].x).toBeCloseTo(
      raw[split + 1].x / dataset.xScale!,
    );
    expect(dataset.validation[0].y).toBeCloseTo(
      raw[split + 1].y / dataset.yScale!,
    );
    const frozen = structuredClone(dataset);
    source.candles.push(...candles(95).slice(90));
    expect(dataset).toEqual(frozen);
  });
  it("never fits normalization or gradients to the held-out targets", () => {
    const before = liveSgdDataset(capture())!;
    const changed = capture();
    changed.candles[85].close *= 1.1;
    const after = liveSgdDataset(changed)!;
    expect(after.xScale).toBe(before.xScale);
    expect(after.yScale).toBe(before.yScale);
    expect(after.points).toEqual(before.points);
    const a = sgdStep(initialSgdState(before), 0.1, before);
    const b = sgdStep(initialSgdState(after), 0.1, after);
    expect(b.weight).toBe(a.weight);
    expect(b.bias).toBe(a.bias);
    expect(b.history.at(-1)!.loss).toBe(a.history.at(-1)!.loss);
    expect(b.history.at(-1)!.validationLoss).not.toBe(
      a.history.at(-1)!.validationLoss,
    );
  });
});
