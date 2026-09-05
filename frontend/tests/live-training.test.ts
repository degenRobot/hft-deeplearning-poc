import { createElement, useEffect } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  parseLiveTraining,
  trainingIsStale,
  DEFAULT_TRAINING_OPTIONS,
  type LiveTraining,
  type TrainingStep,
} from "../lib/liveTraining";
import { useLiveTraining } from "../hooks/useLiveTraining";
const step: TrainingStep = {
  step: 1,
  phase: "rl",
  loss: -0.02,
  reward: 0.1,
  action: 0,
  input_end_ts_ms: 1_700_000_000_000,
  target_ts_ms: 1_700_000_005_000,
  features: Array.from({ length: 30 }, () => Array(10).fill(0)),
  activations: { hidden_1: Array(64).fill(1), hidden_2: Array(32).fill(1) },
  outputs_before: [0.2, 0.3, 0.5],
  outputs_after: [0.21, 0.3, 0.49],
  layers: [
    {
      name: "fc1.weight",
      gradient_norm: 0.01,
      weight_delta_norm: 0.001,
      weight_norm: 2,
    },
  ],
};
function wire(run = "r1"): LiveTraining {
  return {
    schema_version: 1,
    run_id: run,
    status: "running",
    backend: "local",
    updated_at: new Date().toISOString(),
    dataset: {
      symbol: "BTCUSDT",
      event_count: 10000,
      frame_count: 500,
      supervised_examples: 200,
      rl_examples: 100,
      holdout_examples: 50,
      sha256: "a".repeat(64),
    },
    latest: structuredClone(step),
    history: [structuredClone(step)],
    evaluation: null,
    error: null,
  };
}
describe("training contract", () => {
  it("validates optional candle feature metadata without rejecting legacy receipts", () => {
    const value = wire();
    value.dataset!.source_mode = "historical_candles_1s";
    value.dataset!.limitations = ["No order book data"];
    value.dataset!.feature_names = Array.from(
      { length: 10 },
      (_, i) => `Feature ${i}`,
    );
    expect(parseLiveTraining(value)).not.toBeNull();
    value.dataset!.feature_names.pop();
    expect(parseLiveTraining(value)).toBeNull();
    expect(parseLiveTraining(wire())).not.toBeNull();
  });
  it("accepts actual signed RL loss and empty idle telemetry", () => {
    expect(parseLiveTraining(wire())).not.toBeNull();
    expect(
      parseLiveTraining({
        ...wire(),
        status: "idle",
        run_id: null,
        latest: null,
        history: [],
        dataset: null,
        updated_at: null,
      }),
    ).not.toBeNull();
  });
  it.each([
    "features",
    "probabilities",
    "nan",
    "backwards",
    "gradient",
    "activation",
    "history",
    "hash",
  ])("rejects corrupt %s instead of inventing values", (kind) => {
    const v = wire();
    if (kind === "features") v.latest!.features[0].pop();
    if (kind === "probabilities") v.latest!.outputs_after = [0.4, 0.4, 0.4];
    if (kind === "nan") v.latest!.loss = NaN;
    if (kind === "backwards") v.latest!.target_ts_ms = 1;
    if (kind === "gradient") v.latest!.layers[0].gradient_norm = -1;
    if (kind === "activation") v.latest!.activations.hidden_2.pop();
    if (kind === "history") v.history = Array(401).fill(step);
    if (kind === "hash") v.dataset!.sha256 = "unknown";
    expect(parseLiveTraining(v)).toBeNull();
  });
  it("uses actual model widths with capped activation samples and accepts old receipts", () => {
    const large = wire();
    large.dataset!.hidden_sizes = [1024, 512];
    large.dataset!.parameter_count = 834563;
    expect(parseLiveTraining(large)).not.toBeNull();
    const small = wire();
    small.dataset!.hidden_sizes = [8, 8];
    small.dataset!.parameter_count = 2507;
    small.latest!.activations = {
      hidden_1: Array(8).fill(1),
      hidden_2: Array(8).fill(1),
    };
    small.history = [small.latest!];
    expect(parseLiveTraining(small)).not.toBeNull();
    small.latest!.activations.hidden_1.pop();
    expect(parseLiveTraining(small)).toBeNull();
    expect(parseLiveTraining(wire())).not.toBeNull();
  });
  it("expires running updates without expiring a completed historical run", () => {
    const v = wire();
    const now = Date.parse(v.updated_at!) + 10001;
    expect(trainingIsStale(v, now)).toBe(true);
    expect(trainingIsStale({ ...v, status: "completed" }, now)).toBe(false);
  });
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
let latest: ReturnType<typeof useLiveTraining>;
let renderer: ReactTestRenderer | undefined;
const replies: Promise<unknown>[] = [];
const signals: AbortSignal[] = [];
function Harness() {
  const value = useLiveTraining();
  useEffect(() => {
    latest = value;
  });
  return null;
}
async function mount() {
  await act(async () => {
    renderer = create(createElement(Harness));
  });
}
const response = (body: unknown) => ({ ok: true, json: async () => body });
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  replies.length = 0;
  signals.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn((_url: string, init: RequestInit = {}) => {
      signals.push(init.signal as AbortSignal);
      return (replies.shift() ?? Promise.resolve(wire())).then(response);
    }),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(async () => {
  if (renderer) await act(async () => renderer?.unmount());
  renderer = undefined;
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
describe("training polling lifecycle", () => {
  it("clears old measurements on malformed or stale telemetry", async () => {
    await mount();
    expect(latest.data?.run_id).toBe("r1");
    replies.push(Promise.resolve({}));
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    expect(latest.data).toBeNull();
    expect(latest.error).toContain("invalid shape");
    replies.push(
      Promise.resolve({
        ...wire(),
        updated_at: new Date(Date.now() - 20000).toISOString(),
      }),
    );
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    expect(latest.data).toBeNull();
    expect(latest.error).toContain("stale");
  });
  it("clears measurements when a telemetry request goes silent", async () => {
    await mount();
    const silent = deferred<unknown>();
    replies.push(silent.promise);
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(latest.data).toBeNull();
    expect(latest.error).toContain("timed out");
    await act(async () => {
      silent.resolve(wire());
    });
    expect(latest.data).toBeNull();
  });
  it("aborts pending reads and ignores their response after a start acknowledgement", async () => {
    const old = deferred<unknown>();
    replies.push(old.promise);
    await mount();
    replies.push(Promise.resolve({}), Promise.resolve(wire("new")));
    await act(async () => {
      await latest.start();
    });
    expect(signals[0].aborted).toBe(true);
    expect(latest.data?.run_id).toBe("new");
    await act(async () => {
      old.resolve(wire("old"));
    });
    expect(latest.data?.run_id).toBe("new");
    expect(
      vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === "POST"),
    ).toHaveLength(1);
  });
  it("sends explicitly selected training options as JSON", async () => {
    await mount();
    const options = {
      ...DEFAULT_TRAINING_OPTIONS,
      backend: "modal" as const,
      hidden_1: 256,
      hidden_2: 128,
      epochs: 4,
      learning_rate: 0.002,
    };
    await act(async () => {
      await latest.start(options);
    });
    const command = vi
      .mocked(fetch)
      .mock.calls.find(([, init]) => init?.method === "POST");
    expect(command?.[1]?.headers).toEqual({
      "Content-Type": "application/json",
    });
    expect(JSON.parse(command?.[1]?.body as string)).toEqual(options);
  });
  it("prevents duplicate control mutations and aborts outstanding work on unmount", async () => {
    await mount();
    const pending = deferred<unknown>();
    replies.push(pending.promise);
    await act(async () => {
      void latest.start();
      void latest.start();
    });
    expect(latest.pending).toBe(true);
    expect(
      vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === "POST"),
    ).toHaveLength(1);
    await act(async () => renderer?.unmount());
    renderer = undefined;
    expect(signals.at(-1)?.aborted).toBe(true);
    await act(async () => pending.resolve({}));
    expect(vi.getTimerCount()).toBe(0);
  });
});
