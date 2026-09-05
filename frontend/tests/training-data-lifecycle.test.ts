import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingDataset } from "../components/TrainingDataset";
import { TrainingProgress } from "../components/TrainingProgress";
import {
  captureIsStale,
  parsePublicTrainingData,
  historicalRequest,
  defaultHistoricalRange,
} from "../lib/trainingData";
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
    await act(async () =>
      root()
        .findAllByType("select")[0]
        .props.onChange({ target: { value: "live" } }),
    );
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
      expect(html).toContain("Step totals were not recorded");
    },
  );
});

describe("historical acquisition", () => {
  it("defaults to yesterday UTC and sends the selected pair with an exclusive UTC end", async () => {
    vi.setSystemTime(new Date("2026-09-05T23:45:00Z"));
    await mount();
    expect(captureButton().children.join("")).toBe("Fetch historical data");
    expect(
      root()
        .findAllByType("input")
        .map((i) => i.props.value),
    ).toEqual(["2026-09-04T00:00:00", "2026-09-04T00:15:00"]);
    await act(async () =>
      root()
        .findAllByType("select")[1]
        .props.onChange({ target: { value: "ETHUSDT" } }),
    );
    await act(async () => captureButton().props.onClick());
    const post = vi
      .mocked(fetch)
      .mock.calls.find(([, init]) => init?.method === "POST");
    expect(String(post?.[0])).toMatch(/\/training\/data\/history$/);
    expect(JSON.parse(post?.[1]?.body as string)).toEqual({
      symbol: "ETHUSDT",
      start: "2026-09-04T00:00:00.000Z",
      end: "2026-09-04T00:15:00.000Z",
    });
  });
  it("submits edited datetime input values instead of the initial range", async () => {
    vi.setSystemTime(new Date("2026-09-05T23:45:00Z"));
    await mount();
    await act(async () => {
      root()
        .findAllByType("input")[0]
        .props.onInput({ currentTarget: { value: "2026-09-01T00:00" } });
      root()
        .findAllByType("input")[1]
        .props.onInput({ currentTarget: { value: "2026-09-01T01:00" } });
    });
    await act(async () => captureButton().props.onClick());
    const post = vi
      .mocked(fetch)
      .mock.calls.find(([, init]) => init?.method === "POST");
    expect(JSON.parse(post?.[1]?.body as string)).toEqual({
      symbol: "BTCUSDT",
      start: "2026-09-01T00:00:00.000Z",
      end: "2026-09-01T01:00:00.000Z",
    });
  });
  it("rejects malformed dates, fractional seconds, future ends and out-of-bounds spans", () => {
    const now = Date.parse("2026-09-05T00:00:00Z");
    expect(defaultHistoricalRange(now)).toEqual({
      start: "2026-09-04T00:00:00",
      end: "2026-09-04T00:15:00",
    });
    expect(
      historicalRequest("SOLUSDT", "2026-09-04T00:00", "2026-09-04T05:00", now)
        ?.end,
    ).toBe("2026-09-04T05:00:00.000Z");
    for (const [start, end] of [
      ["2026-09-04T00:00", "2026-09-04T00:09:59"],
      ["2026-09-04T00:00", "2026-09-04T05:00:01"],
      ["2026-09-04T00:00:00.1", "2026-09-04T00:15"],
      ["2026-09-04T00:15", "2026-09-04T00:00"],
      ["2026-09-05T00:00", "2026-09-05T00:15"],
      ["2026-02-30T00:00", "2026-02-30T00:15"],
    ])
      expect(historicalRequest("BTCUSDT", start, end, now)).toBeNull();
    expect(
      historicalRequest("OTHER", "2026-09-04T00:00", "2026-09-04T00:15", now),
    ).toBeNull();
  });
  it("accepts historical candle counts and long coverage but rejects inconsistent totals", () => {
    const v = {
      ...running(),
      selected: {
        ...selected,
        event_count: 800,
        candle_count: 800,
        source_mode: "historical_candles_1s",
      },
      capture: {
        ...running().capture,
        mode: "historical",
        requested_seconds: 18000,
        candle_count: 234,
      },
    };
    expect(parsePublicTrainingData(v)).not.toBeNull();
    expect(
      parsePublicTrainingData({
        ...v,
        selected: { ...v.selected, candle_count: 799 },
      }),
    ).toBeNull();
    expect(
      parsePublicTrainingData({
        ...v,
        selected: { ...v.selected, candle_count: 1.5 },
      }),
    ).toBeNull();
    expect(
      parsePublicTrainingData({
        ...v,
        capture: { ...v.capture, mode: "live" },
      }),
    ).toBeNull();
  });
  it("shows native candle coverage separately from download elapsed time", async () => {
    replies.push(
      Promise.resolve({
        ...running(),
        selected: {
          ...selected,
          event_count: 800,
          candle_count: 800,
          source_mode: "historical_candles_1s",
          symbol: "SOLUSDT",
        },
        capture: {
          ...running().capture,
          mode: "historical",
          requested_seconds: 18000,
          candle_count: 234,
          progress: 234 / 18000,
        },
      }),
    );
    await mount();
    expect(root().findByType("progress").props.value).toBe(234 / 18000);
    const text = JSON.stringify(renderer?.toJSON());
    expect(text).toContain("requested market span");
    expect(text).toContain("Elapsed:");
    expect(text).toContain("cannot validate order-book strategies");
    expect(text).not.toContain("BTCUSDT best bid");
  });
  it("surfaces a bounded backend validation detail", async () => {
    await mount();
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: "This range is not yet closed." }),
    } as Response);
    await act(async () => captureButton().props.onClick());
    expect(root().findByProps({ role: "alert" }).children.join(" ")).toContain(
      "This range is not yet closed.",
    );
  });
});
