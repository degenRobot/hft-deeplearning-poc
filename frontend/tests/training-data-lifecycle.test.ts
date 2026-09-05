import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingDataset } from "../components/TrainingDataset";
import { TrainingProgress } from "../components/TrainingProgress";
import { captureIsStale, parsePublicTrainingData } from "../lib/trainingData";
import type { LiveTraining } from "../lib/liveTraining";
const selected = {
  id: "builtin",
  label: "Built-in",
  source: "binance_public",
  symbol: "BTCUSDT",
  path: "data/training-public.jsonl",
  event_count: 0,
  book_count: 0,
  trade_count: 0,
  bytes: 0,
  first_event_ts_ms: null,
  last_event_ts_ms: null,
  sha256: null,
  training_ready: false,
  error: "Missing",
};
const idle = () => ({
  schema_version: 1,
  selected,
  capture: { id: null, status: "idle", can_stop: false },
});
const running = (age = 0) => ({
  ...idle(),
  capture: {
    id: "run1",
    status: "running",
    can_stop: true,
    updated_at: new Date(Date.now() - age).toISOString(),
    elapsed_seconds: 12,
    requested_seconds: 900,
    events: 234,
    bytes: 5678,
    progress: 12 / 900,
  },
});
const response = (body: unknown) => ({ ok: true, json: async () => body });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { resolve, promise };
}
let renderer: ReactTestRenderer | undefined;
const replies: Promise<unknown>[] = [];
const root = () => renderer!.root;
const captureButton = () => root().findAllByType("button")[0];
const stopButton = () => root().findAllByType("button")[1];
async function mount() {
  await act(async () => {
    renderer = create(createElement(TrainingDataset));
  });
}
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  replies.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn((_url: string, _init?: RequestInit) =>
      (replies.shift() ?? Promise.resolve(idle())).then(response),
    ),
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
describe("public data capture lifecycle", () => {
  it("hides stale capture measurements while retaining a currently owned Stop action", async () => {
    replies.push(Promise.resolve(running()));
    await mount();
    expect(root().findAllByType("progress")).toHaveLength(1);
    replies.push(Promise.resolve(running(11000)));
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(root().findAllByType("progress")).toHaveLength(0);
    expect(root().findByProps({ role: "alert" }).children.join(" ")).toContain(
      "heartbeat is stale",
    );
    expect(stopButton().props.disabled).toBe(false);
    expect(captureButton().props.disabled).toBe(true);
    replies.push(Promise.resolve({}), Promise.resolve(idle()));
    await act(async () => {
      await stopButton().props.onClick();
    });
    expect(
      vi
        .mocked(fetch)
        .mock.calls.some(
          ([url, init]) =>
            String(url).endsWith("/stop") && init?.method === "POST",
        ),
    ).toBe(true);
  });
  it("clears malformed telemetry instead of showing missing fields as zeros", async () => {
    replies.push(Promise.resolve(running()));
    await mount();
    replies.push(
      Promise.resolve({
        ...idle(),
        capture: { id: "run1", status: "running", can_stop: true },
      }),
    );
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(root().findAllByType("progress")).toHaveLength(0);
    expect(captureButton().props.disabled).toBe(true);
    expect(stopButton().props.disabled).toBe(true);
    expect(root().findByProps({ role: "alert" }).children.join(" ")).toContain(
      "invalid",
    );
  });
  it("clears a silent response at the deadline and ignores it if it later resolves", async () => {
    replies.push(Promise.resolve(running()));
    await mount();
    const late = deferred<unknown>();
    replies.push(late.promise);
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });
    expect(root().findAllByType("progress")).toHaveLength(0);
    expect(root().findByProps({ role: "alert" }).children.join(" ")).toContain(
      "timed out",
    );
    await act(async () => {
      late.resolve(running());
    });
    expect(root().findAllByType("progress")).toHaveLength(0);
  });
  it("serializes duplicate capture starts and never performs the follow-up GET after unmount", async () => {
    await mount();
    const pending = deferred<unknown>();
    replies.push(pending.promise);
    await act(async () => {
      void captureButton().props.onClick();
      void captureButton().props.onClick();
    });
    const calls = vi.mocked(fetch).mock.calls;
    expect(calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
    expect(JSON.parse(calls.at(-1)?.[1]?.body as string)).toEqual({
      seconds: 900,
    });
    const signal = calls.at(-1)?.[1]?.signal;
    await act(async () => renderer?.unmount());
    renderer = undefined;
    expect(signal?.aborted).toBe(true);
    const before = vi.mocked(fetch).mock.calls.length;
    await act(async () => pending.resolve({}));
    expect(vi.mocked(fetch).mock.calls).toHaveLength(before);
    expect(vi.getTimerCount()).toBe(0);
  });
});
describe("capture and terminal progress validity", () => {
  it.each([
    "elapsed_seconds",
    "requested_seconds",
    "events",
    "bytes",
    "progress",
    "updated_at",
  ])("requires %s for active captures", (key) => {
    const wire = running();
    delete (wire.capture as Record<string, unknown>)[key];
    expect(parsePublicTrainingData(wire)).toBeNull();
  });
  it("rejects invalid timestamps, fractional counts, backwards selected ranges and idle Stop", () => {
    const wire = running();
    expect(
      parsePublicTrainingData({
        ...wire,
        capture: { ...wire.capture, updated_at: "yesterday" },
      }),
    ).toBeNull();
    expect(
      parsePublicTrainingData({
        ...wire,
        capture: { ...wire.capture, events: 1.5 },
      }),
    ).toBeNull();
    expect(
      parsePublicTrainingData({
        ...idle(),
        selected: {
          ...selected,
          first_event_ts_ms: 3000,
          last_event_ts_ms: 2000,
        },
      }),
    ).toBeNull();
    expect(
      parsePublicTrainingData({
        ...idle(),
        capture: { ...idle().capture, can_stop: true },
      }),
    ).toBeNull();
    expect(parsePublicTrainingData(idle())).not.toBeNull();
    expect(captureIsStale(running(11000).capture, Date.now())).toBe(true);
    expect(
      captureIsStale(
        { ...running(11000).capture, status: "completed" },
        Date.now(),
      ),
    ).toBe(false);
  });
  it.each(["failed", "stopped"] as const)(
    "does not animate an unknown %s completion",
    (status) => {
      const data: LiveTraining = {
        schema_version: 1,
        run_id: "test",
        status,
        backend: "modal",
        updated_at: null,
        dataset: null,
        latest: null,
        history: [],
        evaluation: null,
        error: null,
      };
      const html = renderToStaticMarkup(
        createElement(TrainingProgress, { data }),
      );
      expect(html).not.toContain("<progress");
      expect(html).not.toContain("Preparing…");
      expect(html).toContain("Run ended before optimizer totals");
    },
  );
});
