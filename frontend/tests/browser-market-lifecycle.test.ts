import { createElement, useEffect } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useBrowserMarketData } from "../hooks/useBrowserMarketData";
import { EMPTY_STATE } from "../lib/types";

class Socket {
  static instances: Socket[] = [];
  onmessage?: (event: { data: string }) => void;
  onclose?: () => void;
  onerror?: () => void;
  constructor() {
    Socket.instances.push(this);
  }
  close = vi.fn(() => this.onclose?.());
}
let latest: ReturnType<typeof useBrowserMarketData>;
let renderer: ReactTestRenderer;
function Harness() {
  const value = useBrowserMarketData();
  useEffect(() => {
    latest = value;
  });
  return null;
}
const wire = () => ({
  ...EMPTY_STATE,
  timestamp: Date.now(),
  source: "binance",
  gate: { ...EMPTY_STATE.gate, revision: 1 },
  market: { ...EMPTY_STATE.market, mid: 100 },
  health: {
    ...EMPTY_STATE.health,
    ready: true,
    feed_status: "running",
    run_id: "test-run",
    feed_generation: 1,
  },
  visual: {
    events: [],
    window: [],
    proposed: { microprice: 1 / 3, flow: 1 / 3, reversion: 1 / 3 },
    previous: { microprice: 1 / 3, flow: 1 / 3, reversion: 1 / 3 },
    influence: 0.5,
    gate_timestamp_ms: 0,
    snapshot_interval_ms: 100,
    candles: [0, 1000, 2000].map((timestamp_ms) => ({
      timestamp_ms,
      open: 100,
      high: 100,
      low: 100,
      close: 100,
      volume: 1,
    })),
  },
});
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.stubGlobal("WebSocket", Socket);
  vi.stubGlobal("fetch", vi.fn());
  Socket.instances = [];
});
afterEach(async () => {
  if (renderer) await act(async () => renderer.unmount());
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
it("clears stale captures, reconnects read-only and ignores late events after unmount", async () => {
  await act(async () => {
    renderer = create(createElement(Harness));
  });
  const first = Socket.instances[0];
  await act(async () => first.onmessage?.({ data: JSON.stringify(wire()) }));
  expect(latest.capture?.candles).toHaveLength(2);
  await act(async () => {
    vi.advanceTimersByTime(5000);
  });
  expect(latest.capture).toBeNull();
  expect(first.close).toHaveBeenCalledOnce();
  await act(async () => {
    vi.advanceTimersByTime(2000);
  });
  expect(Socket.instances).toHaveLength(2);
  const second = Socket.instances[1];
  await act(async () => renderer.unmount());
  expect(second.close).toHaveBeenCalledOnce();
  await act(async () => {
    second.onmessage?.({ data: JSON.stringify(wire()) });
    vi.advanceTimersByTime(10000);
  });
  expect(Socket.instances).toHaveLength(2);
  expect(fetch).not.toHaveBeenCalled();
});
it("clears an invalid snapshot instead of retaining an apparently live capture", async () => {
  await act(async () => {
    renderer = create(createElement(Harness));
  });
  const socket = Socket.instances[0];
  await act(async () => socket.onmessage?.({ data: JSON.stringify(wire()) }));
  expect(latest.capture).not.toBeNull();
  await act(async () => socket.onmessage?.({ data: '{"bad":true}' }));
  expect(latest.capture).toBeNull();
  expect(latest.status).toContain("invalid snapshot");
});
