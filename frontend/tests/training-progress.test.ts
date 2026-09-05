import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TrainingCost, TrainingProgress } from "../components/TrainingProgress";
import { parseLiveTraining, type LiveTraining } from "../lib/liveTraining";
import { parsePublicTrainingData } from "../lib/trainingData";
const idle: LiveTraining = {
  schema_version: 1,
  run_id: null,
  status: "running",
  backend: "modal",
  updated_at: null,
  dataset: null,
  latest: null,
  history: [],
  evaluation: null,
  error: null,
};
describe("training progress and cost", () => {
  it("shows indeterminate setup and does not invent compute usage", () => {
    const html = renderToStaticMarkup(
      createElement(TrainingProgress, { data: idle }),
    );
    expect(html).toContain("Starting Modal");
    expect(html).not.toContain('value="0"');
    expect(html).not.toContain("Estimated compute since");
  });
  it("keeps finalization below100 until artifacts are saved", () => {
    const data = {
      ...idle,
      progress: {
        stage: "finalizing",
        completed_steps: 44,
        total_steps: 44,
        percent: 99,
        elapsed_seconds: 55,
        remote_elapsed_seconds: 30,
        compute_estimate_usd: 0.0009192,
      },
    };
    const html = renderToStaticMarkup(
      createElement(TrainingProgress, { data }),
    );
    expect(html).toContain("99%");
    expect(html).toContain("saving checkpoints");
    expect(html).toContain("44 / 44");
    expect(
      parseLiveTraining({
        ...data,
        progress: { ...data.progress, percent: 101 },
      }),
    ).toBeNull();
  });
  it("prices the explicit one-minute assumption and disclaims billing", () => {
    const pricing = {
      checked_on: "2026-09-05",
      source_url: "https://modal.com/pricing",
      cpu_core_second_usd: 0.0000131,
      memory_gib_second_usd: 0.00000222,
      cpu_cores: 2,
      memory_gib: 2,
      timeout_seconds: 600,
    };
    const html = renderToStaticMarkup(createElement(TrainingCost, { pricing }));
    expect(html).toContain("$0.00184");
    expect(html).toContain("$0.01838");
    expect(html).toContain("Assumed compute minutes");
    expect(html).toContain("billed usage are unavailable");
  });
});
it("rejects impossible data capture status and preserves missing-data metadata", () => {
  const state = {
    schema_version: 1,
    selected: {
      id: "builtin",
      label: "Built-in",
      path: "data/training-public.jsonl",
      source: "binance_public",
      symbol: "BTCUSDT",
      event_count: 0,
      book_count: 0,
      trade_count: 0,
      bytes: 0,
      first_event_ts_ms: null,
      last_event_ts_ms: null,
      sha256: null,
      training_ready: false,
      error: "Missing",
    },
    capture: { id: null, status: "idle", can_stop: false },
  };
  expect(parsePublicTrainingData(state)).not.toBeNull();
  expect(
    parsePublicTrainingData({
      ...state,
      capture: { ...state.capture, progress: 2 },
    }),
  ).toBeNull();
  expect(
    parsePublicTrainingData({
      ...state,
      selected: { ...state.selected, sha256: "bad" },
    }),
  ).toBeNull();
});
