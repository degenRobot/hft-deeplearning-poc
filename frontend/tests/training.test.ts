import { describe, expect, it } from "vitest";
import {
  formatTrainingCount,
  formatTrainingDuration,
  formatTrainingLoss,
  normalizeTrainingReceipt,
} from "../lib/training";

const receipt = {
  schema_version: 1,
  source: {
    symbol: "BTCUSDT",
    venue: "Binance",
    url: "https://data.binance.vision/",
    duration_seconds: 60,
    event_counts: { total: 42 },
  },
  dataset: {
    train_examples: 5,
    validation_examples: 2,
  },
  training: {
    parameter_count: 123,
    first_train_loss: 0.23456,
    last_train_loss: 0.01234,
    validation_loss: 0.04567,
  },
  limitations: ["Tiny sample.", "Illustrative only."],
};

describe("training receipt", () => {
  it("normalizes the documented receipt contract into safe values", () => {
    const normalized = normalizeTrainingReceipt(receipt);
    expect(normalized?.source.event_counts.total).toBe(42);
    expect(normalized?.dataset.train_examples).toBe(5);
    expect(normalized?.training.parameter_count).toBe(123);
    expect(normalized?.training.last_train_loss).toBe(0.01234);
  });

  it.each([
    ["non-object", null],
    ["wrong schema", { ...receipt, schema_version: 2 }],
    ["missing source", { ...receipt, source: null }],
    ["missing dataset", { ...receipt, dataset: null }],
    ["missing training", { ...receipt, training: null }],
    [
      "non-finite headline metric",
      {
        ...receipt,
        training: { ...receipt.training, validation_loss: Number.NaN },
      },
    ],
    ["missing limitations", { ...receipt, limitations: null }],
  ])("rejects %s", (_, payload) => {
    expect(normalizeTrainingReceipt(payload)).toBeNull();
  });
});

describe("training display helpers", () => {
  it("keeps counts, durations, and losses compact", () => {
    expect(formatTrainingCount(12345)).toBe("12,345");
    expect(formatTrainingDuration(12.34)).toBe("12.3 s");
    expect(formatTrainingDuration(75)).toBe("1m 15s");
    expect(formatTrainingLoss(0.01234)).toBe("0.0123");
    expect(formatTrainingDuration(Number.NaN)).toBe("—");
    expect(formatTrainingDuration(-1)).toBe("—");
    expect(formatTrainingCount(Number.POSITIVE_INFINITY)).toBe("—");
    expect(formatTrainingLoss(Number.NaN)).toBe("—");
  });
});
