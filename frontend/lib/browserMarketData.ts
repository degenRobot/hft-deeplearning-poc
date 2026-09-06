import type { MarketGateState } from "./types";
import type { Candle } from "./visual";
import type { SgdDataset } from "./sgdDemo";

export const MAX_CAPTURE_CANDLES = 360;
export const MIN_CAPTURE_ROWS = 40;
export type BrowserCapture = { key: string; symbol: string; candles: Candle[] };

/** Only completed trade candles; no interpolation, repeated samples or mixed runs. */
export function collectMarketCandles(
  previous: BrowserCapture | null,
  snapshot: MarketGateState,
): BrowserCapture | null {
  if (
    snapshot.source !== "binance" ||
    !snapshot.health.ready ||
    !snapshot.visual
  )
    return null;
  const key = `${snapshot.health.run_id}:${snapshot.health.feed_generation}:${snapshot.symbol}`;
  const incoming = snapshot.visual.candles ?? [];
  const newest = incoming.at(-1)?.timestamp_ms;
  const candles = new Map(
    (previous?.key === key ? previous.candles : []).map((c) => [
      c.timestamp_ms,
      c,
    ]),
  );
  for (const candle of incoming) {
    // A later observed trade second establishes that this candle has closed.
    if (newest !== undefined && candle.timestamp_ms < newest)
      candles.set(candle.timestamp_ms, candle);
  }
  const sorted = [...candles.values()]
    .sort((a, b) => a.timestamp_ms - b.timestamp_ms)
    .slice(-MAX_CAPTURE_CANDLES);
  if (
    previous?.key === key &&
    sorted.length === previous.candles.length &&
    sorted.at(-1)?.timestamp_ms === previous.candles.at(-1)?.timestamp_ms
  )
    return previous;
  return { key, symbol: snapshot.symbol, candles: sorted };
}

export function capturedReturns(capture: BrowserCapture) {
  const rows: { x: number; y: number; timestamp: number }[] = [];
  for (let i = 2; i < capture.candles.length; i++) {
    const [a, b, c] = capture.candles.slice(i - 2, i + 1);
    if (
      b.timestamp_ms - a.timestamp_ms !== 1000 ||
      c.timestamp_ms - b.timestamp_ms !== 1000
    )
      continue;
    rows.push({
      x: (b.close / a.close - 1) * 10000,
      y: (c.close / b.close - 1) * 10000,
      timestamp: c.timestamp_ms,
    });
  }
  return rows;
}

export function liveSgdDataset(capture: BrowserCapture): SgdDataset | null {
  const rows = capturedReturns(capture);
  if (rows.length < MIN_CAPTURE_ROWS) return null;
  const split = Math.floor(rows.length * 0.8);
  const train = rows.slice(0, split);
  // One row is purged so the last training target is not a holdout input.
  const holdout = rows.slice(split + 1);
  const xScale = Math.max(0.01, ...train.map((point) => Math.abs(point.x)));
  const yScale = Math.max(0.01, ...train.map((point) => Math.abs(point.y)));
  const scale = ({ x, y }: { x: number; y: number }) => ({
    x: x / xScale,
    y: y / yScale,
  });
  const mean =
    train.reduce((sum, point) => sum + point.y / yScale, 0) / train.length;
  const validation = holdout.map(scale);
  return {
    id: `${capture.key}:${rows[0].timestamp}:${rows.at(-1)!.timestamp}`,
    label: `${capture.symbol} · captured live trades`,
    points: train.map(scale),
    validation,
    baselineLoss:
      validation.reduce((sum, point) => sum + (mean - point.y) ** 2, 0) /
      validation.length,
    capturedAt: rows.at(-1)!.timestamp + 1000,
    xScale,
    yScale,
  };
}
