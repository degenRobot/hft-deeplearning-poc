import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import { TrainingLab } from "../components/TrainingLab";
import { useLiveTraining } from "../hooks/useLiveTraining";
import type { LiveTraining } from "../lib/liveTraining";
vi.mock("../hooks/useLiveTraining", () => ({ useLiveTraining: vi.fn() }));
it("labels historical features from the run and explains candle-only limitations", () => {
  const data: LiveTraining = {
    schema_version: 1,
    run_id: "history",
    status: "completed",
    backend: "local",
    updated_at: null,
    dataset: {
      symbol: "ETHUSDT",
      event_count: 900,
      frame_count: 900,
      supervised_examples: 500,
      rl_examples: 200,
      holdout_examples: 100,
      sha256: "a".repeat(64),
      source_mode: "historical_candles_1s",
      limitations: ["No recorded order book"],
      feature_names: Array.from(
        { length: 10 },
        (_, i) => `Candle feature ${i}`,
      ),
    },
    latest: {
      step: 1,
      phase: "rl",
      loss: 0,
      reward: 0,
      action: 1,
      input_end_ts_ms: 1700000000000,
      target_ts_ms: 1700000005000,
      features: Array.from({ length: 30 }, () => Array(10).fill(0)),
      activations: { hidden_1: Array(64).fill(0), hidden_2: Array(32).fill(0) },
      outputs_before: [0.2, 0.3, 0.5],
      outputs_after: [0.2, 0.3, 0.5],
      layers: [],
    },
    history: [],
    evaluation: null,
    error: null,
  };
  vi.mocked(useLiveTraining).mockReturnValue({
    data,
    error: "",
    loading: false,
    pending: false,
    start: vi.fn(),
    stop: vi.fn(),
  });
  const html = renderToStaticMarkup(createElement(TrainingLab));
  expect(html).toContain("Historical candle proxy training");
  expect(html).toContain("five-second close-price move");
  expect(html).toContain("not compatible with the live terminal");
  expect(html).toContain("No recorded order book");
  expect(html).toContain("900 candles");
  expect(html).toContain("Candle feature 0");
  expect(html).toContain("Candle feature 9");
});
