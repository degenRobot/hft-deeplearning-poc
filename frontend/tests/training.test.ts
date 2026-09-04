import { describe, expect, it } from "vitest";
import {
  formatTrainingCount,
  formatTrainingDuration,
  formatTrainingLoss,
  normalizeTrainingReceipt,
} from "../lib/training";

const receipt = {
  schema_version: 1,
  generated_at: "2026-09-04T10:00:00Z",
  source: {
    name: "Binance public websocket",
    symbol: "BTCUSDT",
    venue: "Binance",
    url: "https://data.binance.vision/",
    recording_path: "data/binance-btcusdt-sample.jsonl",
    started_at: "2026-09-04T09:59:00Z",
    ended_at: "2026-09-04T10:00:00Z",
    duration_seconds: 60,
    event_counts: { book: 30, trade: 12, total: 42 },
  },
  dataset: {
    feature_names: ["mid", "spread_bps"],
    frame_seconds: 1,
    lookback_frames: 30,
    horizon_frames: 2,
    frames: 38,
    examples: 7,
    train_examples: 5,
    validation_examples: 2,
  },
  training: {
    seed: 7,
    epochs: 20,
    learning_rate: 0.001,
    parameter_count: 123,
    first_train_loss: 0.23456,
    last_train_loss: 0.01234,
    validation_loss: 0.04567,
    model_path: "models/gate-binance-demo.npz",
  },
  limitations: ["Tiny sample.", "Illustrative only."],
};

describe("training receipt", () => {
  it("normalizes the documented receipt contract into safe values", () => {
    const normalized = normalizeTrainingReceipt(receipt);
    expect(normalized?.source.event_counts.total).toBe(42);
    expect(normalized?.dataset.feature_names).toEqual(["mid", "spread_bps"]);
    expect(normalized?.training.last_train_loss).toBe(0.01234);
  });

  it("rejects malformed or incomplete payloads", () => {
    expect(
      normalizeTrainingReceipt({ ...receipt, schema_version: 2 }),
    ).toBeNull();
    expect(
      normalizeTrainingReceipt({
        ...receipt,
        training: { ...receipt.training, validation_loss: "unknown" },
      }),
    ).toBeNull();
    expect(
      normalizeTrainingReceipt({
        ...receipt,
        limitations: ["ok", { hidden: "value" }],
      }),
    ).toBeNull();
    expect(
      normalizeTrainingReceipt({
        ...receipt,
        source: { ...receipt.source, url: "javascript:alert(1)" },
      }),
    ).toBeNull();
  });
});

describe("training display helpers", () => {
  it("keeps counts, durations, and losses compact", () => {
    expect(formatTrainingCount(12345)).toBe("12,345");
    expect(formatTrainingDuration(12.34)).toBe("12.3 s");
    expect(formatTrainingDuration(75)).toBe("1m 15s");
    expect(formatTrainingLoss(0.01234)).toBe("0.0123");
    expect(formatTrainingDuration(Number.NaN)).toBe("—");
  });
});
