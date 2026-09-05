import {
  createElement,
  useEffect,
  type ReactElement,
  type ReactNode,
} from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  LiveLearningPanel,
  parseLearning,
  useLiveLearning,
  type LearningController,
  type LearningStatus,
} from "../components/LiveLearning";
import { TrainingLab } from "../components/TrainingLab";
import { useLiveTraining } from "../hooks/useLiveTraining";
import type { LiveTraining } from "../lib/liveTraining";

// Navigation prefetch is outside this Node-renderer lifecycle test.
vi.mock("next/link", () => ({
  default: (props: { children?: ReactNode; href: string }) =>
    createElement("a", props),
}));

const learningWire = (patch: Partial<LearningStatus> = {}): LearningStatus => ({
  enabled: false,
  interval_seconds: 10,
  stage: "off",
  updates: 0,
  source: "binance",
  symbol: "BTCUSDT",
  run_id: "terminal",
  model_version: "demo",
  weight_delta: 0,
  reward: null,
  step_duration_ms: null,
  next_update_in_seconds: null,
  error: null,
  ...patch,
});
const trainingWire = (run: string): LiveTraining => ({
  schema_version: 1,
  run_id: run,
  status: "idle",
  backend: "local",
  updated_at: null,
  dataset: null,
  latest: null,
  history: [],
  evaluation: null,
  error: null,
});
const response = (body: unknown) =>
  ({ ok: true, json: async () => body }) as Response;
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
let renderer: ReactTestRenderer | undefined;
let latest: LearningController;
let training: ReturnType<typeof useLiveTraining>;
const replies: Promise<unknown>[] = [];
const signals: AbortSignal[] = [];
function LearningHarness() {
  const value = useLiveLearning();
  useEffect(() => {
    latest = value;
  });
  return null;
}
function TrainingHarness({ mode }: { mode: "manual" | "live" }) {
  const value = useLiveTraining(mode);
  useEffect(() => {
    training = value;
  });
  return null;
}
async function mount(element: ReactElement = createElement(LearningHarness)) {
  await act(async () => {
    renderer = create(element);
  });
}
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  replies.length = 0;
  signals.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn((_url: string, init: RequestInit = {}) => {
      signals.push(init.signal as AbortSignal);
      return (replies.shift() ?? Promise.resolve(learningWire())).then(
        response,
      );
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

describe("live RL telemetry contract", () => {
  it("accepts paused state and signed reward with nullable timings", () => {
    expect(parseLearning(learningWire())).not.toBeNull();
    expect(
      parseLearning(
        learningWire({ stage: "paused", updates: 3, reward: -0.5 }),
      ),
    ).not.toBeNull();
  });
  it.each([
    { enabled: "yes" },
    { updates: 1.5 },
    { updates: -1 },
    { weight_delta: NaN },
    { interval_seconds: 4 },
    { interval_seconds: 61 },
    { interval_seconds: 5.5 },
    { step_duration_ms: -1 },
    { next_update_in_seconds: -1 },
    { reward: Infinity },
    { run_id: null },
    { error: {} },
  ])("rejects malformed or impossible telemetry %j", (patch) => {
    expect(parseLearning({ ...learningWire(), ...patch })).toBeNull();
  });
});

describe("live RL controls", () => {
  it("patches enable, interval and explicit paused reset, and disables controls while pending", async () => {
    const change = vi.fn();
    const controller = {
      data: learningWire(),
      error: "",
      pending: false,
      change,
    };
    await mount(createElement(LiveLearningPanel, { learning: controller }));
    const root = renderer!.root;
    await act(async () =>
      root
        .findByProps({ role: "switch" })
        .props.onChange({ target: { checked: true } }),
    );
    await act(async () =>
      root.findByType("select").props.onChange({ target: { value: "30" } }),
    );
    await act(async () => root.findByType("button").props.onClick());
    expect(change.mock.calls.map(([patch]) => patch)).toEqual([
      { enabled: true },
      { interval_seconds: 30 },
      { reset: true, enabled: false },
    ]);
    await act(async () =>
      renderer!.update(
        createElement(LiveLearningPanel, {
          learning: { ...controller, pending: true },
        }),
      ),
    );
    expect(root.findByProps({ role: "switch" }).props.disabled).toBe(true);
    expect(root.findByType("select").props.disabled).toBe(true);
    expect(root.findByType("button").props.disabled).toBe(true);
  });
});

describe("live RL polling", () => {
  it("clears prior measurements on malformed responses", async () => {
    await mount();
    expect(latest.data?.run_id).toBe("terminal");
    replies.push(Promise.resolve({ enabled: true }));
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(latest.data).toBeNull();
    expect(latest.error).toContain("telemetry unavailable");
  });
  it("aborts an old GET and ignores its late result after a PATCH", async () => {
    const old = deferred<unknown>();
    replies.push(old.promise);
    await mount();
    replies.push(Promise.resolve(learningWire({ enabled: true })));
    await act(async () => latest.change({ enabled: true }));
    expect(signals[0].aborted).toBe(true);
    expect(latest.data?.enabled).toBe(true);
    await act(async () => old.resolve(learningWire()));
    expect(latest.data?.enabled).toBe(true);
    const patch = vi
      .mocked(fetch)
      .mock.calls.find(([, init]) => init?.method === "PATCH");
    expect(String(patch?.[0])).toMatch(/\/learning$/);
    expect(JSON.parse(patch?.[1]?.body as string)).toEqual({ enabled: true });
  });
  it("serializes duplicate patches and aborts commands on unmount", async () => {
    await mount();
    const pending = deferred<unknown>();
    replies.push(pending.promise);
    await act(async () => {
      latest.change({ enabled: true });
      latest.change({ enabled: true });
    });
    expect(latest.pending).toBe(true);
    expect(
      vi
        .mocked(fetch)
        .mock.calls.filter(([, init]) => init?.method === "PATCH"),
    ).toHaveLength(1);
    await act(async () => renderer?.unmount());
    renderer = undefined;
    expect(signals.at(-1)?.aborted).toBe(true);
    const calls = vi.mocked(fetch).mock.calls.length;
    await act(async () => pending.resolve(learningWire({ enabled: true })));
    expect(vi.mocked(fetch).mock.calls).toHaveLength(calls);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("clears every scheduled timer immediately when an unresolved request unmounts", async () => {
    const pending = deferred<unknown>();
    replies.push(pending.promise);
    await mount();
    await act(async () => renderer?.unmount());
    renderer = undefined;
    expect(signals[0].aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("unlocks a timed-out PATCH and retries status without accepting its late acknowledgement", async () => {
    await mount();
    const pending = deferred<unknown>();
    replies.push(pending.promise);
    await act(async () => latest.change({ enabled: true }));
    expect(latest.pending).toBe(true);
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(latest.pending).toBe(false);
    expect(latest.data).toBeNull();
    replies.push(Promise.resolve(learningWire({ enabled: false, updates: 2 })));
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(latest.data?.updates).toBe(2);
    await act(async () =>
      pending.resolve(learningWire({ enabled: true, updates: 0 })),
    );
    expect(latest.data?.updates).toBe(2);
    expect(latest.data?.enabled).toBe(false);
    expect(
      vi
        .mocked(fetch)
        .mock.calls.filter(([, init]) => init?.method === "PATCH"),
    ).toHaveLength(1);
  });
  it("clears stale data at the deadline even when the transport never rejects", async () => {
    await mount();
    const silent = deferred<unknown>();
    replies.push(silent.promise);
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(latest.data).toBeNull();
    expect(latest.error).toContain("timed out");
  });
  it("does not accept a success response after its abort deadline", async () => {
    const late = deferred<unknown>();
    replies.push(late.promise);
    await mount();
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(signals[0].aborted).toBe(true);
    await act(async () => late.resolve(learningWire({ enabled: true })));
    expect(latest.data).toBeNull();
  });
});

describe("Training Lab manual and live telemetry", () => {
  it("retains the selected history view when an enabled learning feed recovers from an outage", async () => {
    let unavailable = false;
    vi.mocked(fetch).mockImplementation(async (url) => {
      const path = String(url);
      if (path.endsWith("/learning")) {
        if (unavailable) throw new Error("Temporary learning outage");
        return response(learningWire({ enabled: true }));
      }
      if (path.endsWith("/learning/training"))
        return response(trainingWire("live"));
      if (path.endsWith("/training/live"))
        return response(trainingWire("manual"));
      return response({});
    });
    await mount(createElement(TrainingLab));
    const root = renderer!.root;
    const tabs = () =>
      root
        .findByProps({ "aria-label": "Training view" })
        .findAllByType("button");
    expect(tabs()[0].props["aria-pressed"]).toBe(true);
    await act(async () => tabs()[1].props.onClick());
    expect(root.findByProps({ role: "switch" }).props.checked).toBe(true);
    await act(async () => tabs()[0].props.onClick());
    unavailable = true;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(root.findAllByProps({ role: "switch" })).toHaveLength(0);
    expect(tabs()[0].props["aria-pressed"]).toBe(true);
    unavailable = false;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(root.findAllByProps({ role: "switch" })).toHaveLength(0);
    expect(tabs()[0].props["aria-pressed"]).toBe(true);
    expect(tabs()[1].props["aria-pressed"]).toBe(false);
    expect(
      root.findAllByProps({ "aria-label": "Configure training" }),
    ).toHaveLength(1);
  });
  it("switches endpoints, immediately hides the previous mode, and prevents manual commands in live mode", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response(trainingWire("manual")));
    await mount(createElement(TrainingHarness, { mode: "manual" }));
    expect(training.data?.run_id).toBe("manual");
    const pending = deferred<Response>();
    vi.mocked(fetch).mockReturnValueOnce(pending.promise);
    await act(async () =>
      renderer!.update(createElement(TrainingHarness, { mode: "live" })),
    );
    expect(training.data).toBeNull();
    expect(String(vi.mocked(fetch).mock.calls.at(-1)?.[0])).toMatch(
      /\/learning\/training$/,
    );
    const before = vi.mocked(fetch).mock.calls.length;
    await act(async () => {
      void training.start();
      void training.stop();
    });
    expect(vi.mocked(fetch).mock.calls).toHaveLength(before);
    await act(async () => pending.resolve(response(trainingWire("live"))));
    expect(training.data?.run_id).toBe("live");
    vi.mocked(fetch).mockResolvedValueOnce(response(trainingWire("manual")));
    await act(async () =>
      renderer!.update(createElement(TrainingHarness, { mode: "manual" })),
    );
    expect(String(vi.mocked(fetch).mock.calls.at(-1)?.[0])).toMatch(
      /\/training\/live$/,
    );
    expect(training.data?.run_id).toBe("manual");
  });
  it("ignores an old mode response that resolves after the mode changes", async () => {
    const old = deferred<Response>();
    vi.mocked(fetch).mockReturnValueOnce(old.promise);
    await mount(createElement(TrainingHarness, { mode: "manual" }));
    const oldSignal = vi.mocked(fetch).mock.calls[0][1]?.signal;
    vi.mocked(fetch).mockResolvedValueOnce(response(trainingWire("live")));
    await act(async () =>
      renderer!.update(createElement(TrainingHarness, { mode: "live" })),
    );
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => old.resolve(response(trainingWire("manual"))));
    expect(training.data?.run_id).toBe("live");
  });
  it("opens live controls only on selection and never starts training through navigation", async () => {
    let enabled = false;
    vi.mocked(fetch).mockImplementation(async (url, init) => {
      const path = String(url);
      if (path.endsWith("/learning")) {
        if (init?.method === "PATCH")
          enabled = JSON.parse(init.body as string).enabled;
        return response(learningWire({ enabled }));
      }
      if (path.endsWith("/learning/training"))
        return response(trainingWire("live"));
      if (path.endsWith("/training/live"))
        return response(trainingWire("manual"));
      return response({});
    });
    await mount(createElement(TrainingLab));
    const root = renderer!.root;
    const tabs = () =>
      root
        .findByProps({ "aria-label": "Training view" })
        .findAllByType("button");
    expect(tabs()[0].props["aria-pressed"]).toBe(true);
    expect(root.findAllByProps({ role: "switch" })).toHaveLength(0);
    await act(async () => tabs()[1].props.onClick());
    expect(
      vi
        .mocked(fetch)
        .mock.calls.every(([, init]) => !init?.method || init.method === "GET"),
    ).toBe(true);
    await act(async () =>
      root
        .findByProps({ role: "switch" })
        .props.onChange({ target: { checked: true } }),
    );
    expect(tabs()[1].props["aria-pressed"]).toBe(true);
    expect(
      vi
        .mocked(fetch)
        .mock.calls.some(([url]) => String(url).endsWith("/learning/training")),
    ).toBe(true);
    expect(
      root.findAllByProps({ "aria-label": "Configure training" }),
    ).toHaveLength(0);
    await act(async () => tabs()[0].props.onClick());
    expect(tabs()[0].props["aria-pressed"]).toBe(true);
    expect(
      root.findAllByProps({ "aria-label": "Configure training" }),
    ).toHaveLength(1);
  });
});
