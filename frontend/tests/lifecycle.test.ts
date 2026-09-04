import { createElement, useEffect } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SNAPSHOT_TIMEOUT_MS, useMarketGate } from "../hooks/useMarketGate";
import { DEFAULT_CONFIG, EMPTY_STATE } from "../lib/types";
import { normalizeConfig, validateConfig } from "../lib/config";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
class FakeSocket {
  static instances: FakeSocket[] = [];
  onopen?: () => void;
  onclose?: () => void;
  onerror?: () => void;
  onmessage?: (event: { data: string }) => void;
  constructor() {
    FakeSocket.instances.push(this);
  }
  close() {
    this.onclose?.();
  }
  sendState(run = "run-1", generation = 1) {
    this.onmessage?.({
      data: JSON.stringify({
        ...EMPTY_STATE,
        timestamp: 1000,
        market: { ...EMPTY_STATE.market, mid: 100 },
        gate: { ...EMPTY_STATE.gate, revision: 1 },
        quote: { bid: 99, ask: 101 },
        health: {
          ...EMPTY_STATE.health,
          ready: true,
          feed_status: "running",
          run_id: run,
          feed_generation: generation,
        },
      }),
    });
  }
}
let latest: ReturnType<typeof useMarketGate>;
let renderer: ReactTestRenderer;
let serverConfig = { ...DEFAULT_CONFIG };
let getReplies: Promise<unknown>[];
let mutations: {
  method: string;
  body: unknown;
  reply: ReturnType<typeof deferred<unknown>>;
}[];
function Harness() {
  const market = useMarketGate();
  useEffect(() => {
    latest = market;
  });
  return null;
}
async function mount() {
  await act(async () => {
    renderer = create(createElement(Harness));
  });
}
async function flush(action: () => void) {
  await act(async () => {
    action();
  });
}
const response = (body: unknown) => ({ ok: true, json: async () => body });
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  FakeSocket.instances = [];
  getReplies = [];
  mutations = [];
  serverConfig = { ...DEFAULT_CONFIG };
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn((_url: string, init: RequestInit = {}) => {
      if (!init.method)
        return (getReplies.shift() ?? Promise.resolve(serverConfig)).then(
          response,
        );
      const reply = deferred<unknown>();
      mutations.push({
        method: init.method,
        body: init.body ? JSON.parse(String(init.body)) : null,
        reply,
      });
      return reply.promise.then(response);
    }),
  );
  vi.spyOn(console, "error").mockImplementation((message) => {
    if (!String(message).includes("react-test-renderer is deprecated"))
      throw new Error(String(message));
  });
});
afterEach(async () => {
  if (renderer) await act(async () => renderer.unmount());
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("market hook lifecycle", () => {
  it("expires a silent open socket and recovers only on a valid snapshot", async () => {
    await mount();
    const socket = FakeSocket.instances[0];
    await flush(() => {
      socket.onopen?.();
      socket.sendState();
    });
    expect(latest.hasSnapshot).toBe(true);
    await flush(() => vi.advanceTimersByTime(SNAPSHOT_TIMEOUT_MS));
    expect(latest.hasSnapshot).toBe(false);
    expect(latest.state.quote).toBeNull();
    expect(latest.lastError).toContain("stopped arriving");
    await flush(() => socket.sendState());
    expect(latest.hasSnapshot).toBe(true);
  });

  it.each(["not-json", "{}", JSON.stringify({ health: { ready: true } })])(
    "clears previously ready data for malformed payload %s",
    async (data) => {
      await mount();
      const socket = FakeSocket.instances[0];
      await flush(() => socket.sendState());
      await flush(() => socket.onmessage?.({ data }));
      expect(latest.hasSnapshot).toBe(false);
      expect(latest.state.quote).toBeNull();
      expect(latest.status).toBe("error");
    },
  );

  it("ignores old socket callbacks while a new connection accepts restarted generations", async () => {
    await mount();
    const old = FakeSocket.instances[0];
    await flush(() => old.sendState("run-8", 8));
    await flush(() => latest.reconnect());
    const current = FakeSocket.instances[1];
    await flush(() => current.sendState("restarted", 1));
    await flush(() => {
      old.sendState("old", 9);
      old.onerror?.();
      old.onclose?.();
      old.onopen?.();
    });
    expect(latest.state.health.run_id).toBe("restarted");
    expect(latest.status).toBe("connected");
  });

  it("suppresses reset snapshots in flight and fences older or mismatched acknowledged runs", async () => {
    await mount();
    const socket = FakeSocket.instances[0];
    await flush(() => socket.sendState());
    let reset!: Promise<void>;
    await flush(() => {
      reset = latest.resetRun();
      void latest.resetRun();
      void latest.applyConfig();
      socket.sendState();
    });
    expect(mutations).toHaveLength(1);
    expect(latest.hasSnapshot).toBe(false);
    await act(async () => {
      mutations[0].reply.resolve({ run_id: "run-2", feed_generation: 2 });
      await reset;
    });
    await flush(() => {
      socket.sendState();
      socket.sendState("wrong", 2);
    });
    expect(latest.hasSnapshot).toBe(false);
    await flush(() => socket.sendState("run-2", 2));
    expect(latest.hasSnapshot).toBe(true);
    await flush(() => socket.sendState("run-3", 3));
    await flush(() => socket.sendState("run-2", 2));
    expect(latest.state.health.run_id).toBe("run-3");
  });

  it("projects config receipts, retains edits during apply, and refuses concurrent mutations", async () => {
    await mount();
    const socket = FakeSocket.instances[0];
    await flush(() => {
      socket.sendState();
      latest.updateDraft("gate_mode", "uniform");
    });
    let apply!: Promise<void>;
    await flush(() => {
      apply = latest.applyConfig();
      void latest.resetRun();
      latest.updateDraft("expert_strength", 2);
      socket.sendState();
    });
    expect(mutations).toHaveLength(1);
    expect(latest.hasSnapshot).toBe(false);
    serverConfig = { ...DEFAULT_CONFIG, gate_mode: "uniform" };
    await act(async () => {
      mutations[0].reply.resolve({
        ...serverConfig,
        run_id: "run-2",
        feed_generation: 2,
      });
      await apply;
    });
    expect(latest.draftConfig.expert_strength).toBe(2);
    expect(latest.appliedConfig.expert_strength).toBe(1);
    await flush(() => socket.sendState());
    expect(latest.hasSnapshot).toBe(false);
    await flush(() => socket.sendState("run-2", 2));
    expect(latest.hasSnapshot).toBe(true);
    await flush(() => {
      apply = latest.applyConfig();
    });
    expect(Object.keys(mutations[1].body as object).sort()).toEqual(
      Object.keys(DEFAULT_CONFIG).sort(),
    );
    await act(async () => {
      mutations[1].reply.resolve({
        ...serverConfig,
        expert_strength: 2,
        run_id: "run-3",
        feed_generation: 3,
      });
      await apply;
    });
  });

  it("ignores a late config read after apply and preserves an unsaved draft across reset", async () => {
    const stale = deferred<unknown>();
    getReplies.push(stale.promise);
    await mount();
    await flush(() => latest.updateDraft("gate_mode", "uniform"));
    let apply!: Promise<void>;
    await flush(() => {
      apply = latest.applyConfig();
    });
    serverConfig = { ...DEFAULT_CONFIG, gate_mode: "uniform" };
    await act(async () => {
      mutations[0].reply.resolve({
        ...serverConfig,
        run_id: "run-2",
        feed_generation: 2,
      });
      await apply;
    });
    await flush(() => stale.resolve(DEFAULT_CONFIG));
    expect(latest.appliedConfig.gate_mode).toBe("uniform");
    await flush(() => latest.updateDraft("expert_strength", 2));
    let reset!: Promise<void>;
    await flush(() => {
      reset = latest.resetRun();
    });
    await act(async () => {
      mutations[1].reply.resolve({ run_id: "run-3", feed_generation: 3 });
      await reset;
    });
    expect(latest.draftConfig.expert_strength).toBe(2);
    expect(latest.dirty).toBe(true);
  });

  it("rejects a config mutation without a complete run receipt", async () => {
    await mount();
    await flush(() => latest.updateDraft("gate_mode", "uniform"));
    let apply!: Promise<void>;
    await flush(() => {
      apply = latest.applyConfig();
    });
    await act(async () => {
      mutations[0].reply.resolve({ gate_mode: "uniform" });
      await apply;
    });
    expect(latest.settingsError).toContain("invalid run receipt");
    expect(latest.hasSnapshot).toBe(false);
    expect(FakeSocket.instances).toHaveLength(2);
    await flush(() => FakeSocket.instances[0].sendState());
    expect(latest.hasSnapshot).toBe(false);
    await flush(() => FakeSocket.instances[1].sendState("recovered", 2));
    expect(latest.hasSnapshot).toBe(true);
  });

  it.each(["reset", "apply"])(
    "reconnects after %s acknowledgement when socket changed in flight",
    async (action) => {
      await mount();
      await flush(() => latest.updateDraft("gate_mode", "uniform"));
      let mutation!: Promise<void>;
      await flush(() => {
        mutation =
          action === "reset" ? latest.resetRun() : latest.applyConfig();
      });
      await flush(() => latest.reconnect());
      const beforeAck = FakeSocket.instances[1];
      await flush(() => beforeAck.sendState());
      await flush(() => vi.advanceTimersByTime(SNAPSHOT_TIMEOUT_MS * 2));
      expect(latest.hasSnapshot).toBe(false);
      expect(latest.status).toBe("connecting");
      await act(async () => {
        mutations[0].reply.resolve({
          ...DEFAULT_CONFIG,
          run_id: "run-2",
          feed_generation: 2,
        });
        await mutation;
      });
      expect(FakeSocket.instances).toHaveLength(3);
      await flush(() => beforeAck.sendState());
      expect(latest.hasSnapshot).toBe(false);
      await flush(() => FakeSocket.instances[2].sendState("run-2", 2));
      expect(latest.hasSnapshot).toBe(true);
    },
  );
});

it("projects only known config fields and validates replay symbols", () => {
  expect(
    normalizeConfig({
      ...DEFAULT_CONFIG,
      run_id: "r",
      feed_generation: 3,
      stale_after_ms: 2500,
    }),
  ).toEqual(DEFAULT_CONFIG);
  expect(validateConfig({ ...DEFAULT_CONFIG, symbol: "ETHUSDT" })).toContain(
    "Replay fixture supports BTCUSDT only.",
  );
  expect(
    validateConfig({ ...DEFAULT_CONFIG, source: "binance", symbol: "ETHUSDT" }),
  ).toEqual([]);
});
