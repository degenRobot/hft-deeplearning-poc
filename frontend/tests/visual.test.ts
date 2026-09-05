import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SignalFlow } from "../components/SignalFlow";
import { EMPTY_STATE } from "../lib/types";
import { parseVisual, type VisualTelemetry } from "../lib/visual";

const visual: VisualTelemetry = {
  events: [
    {
      id: 1,
      timestamp_ms: 1000,
      kind: "buy",
      price: 12345.67,
      size: 0.2,
      scores: [0.1, 0.2, -0.3],
      signal: 0.1,
      gate_revision: 1,
    },
  ],
  window: [{ timestamp_ms: 999, values: [0, 0, 0, 1, 0.5, 0, 1, 2, 3, 0] }],
  proposed: { microprice: 0.5, flow: 0.3, reversion: 0.2 },
  previous: { microprice: 0.4, flow: 0.3, reversion: 0.3 },
  influence: 0.35,
  gate_timestamp_ms: 1000,
  snapshot_interval_ms: 100,
};
describe("visual telemetry", () => {
  it("rejects corrupt charts instead of coercing them into real data", () => {
    expect(parseVisual(visual)).toEqual(visual);
    expect(
      parseVisual({ ...visual, events: Array(49).fill(visual.events[0]) }),
    ).toBeNull();
    expect(
      parseVisual({
        ...visual,
        events: [{ ...visual.events[0], price: "12345.67" }],
      }),
    ).toBeNull();
    expect(
      parseVisual({
        ...visual,
        window: [{ timestamp_ms: 999, values: [1, 2] }],
      }),
    ).toBeNull();
    expect(
      parseVisual({
        ...visual,
        window: [{ timestamp_ms: 999, values: Array(10).fill(Infinity) }],
      }),
    ).toBeNull();
    expect(
      parseVisual({ ...visual, events: [visual.events[0], visual.events[0]] }),
    ).toBeNull();
  });
  it("shows observed events and padding with no invented activations", () => {
    const state = { ...EMPTY_STATE, visual };
    const html = renderToStaticMarkup(
      createElement(SignalFlow, { state, ready: true, nextRefresh: 300 }),
    );
    expect(html).toContain("12,345.67");
    expect(html).toContain("29 zero-padded frames");
    expect(html).toContain("Waiting for activation telemetry from this pass");
    expect(html).not.toContain("data-activation=");
    expect(html).toContain("50.0%");
    expect(html).toContain("Inference, not live retraining");
  });
  it("hides retained data and stops motion when readiness is lost", () => {
    const html = renderToStaticMarkup(
      createElement(SignalFlow, {
        state: { ...EMPTY_STATE, visual },
        ready: false,
        nextRefresh: 0,
      }),
    );
    expect(html).not.toContain("12,345.67");
    expect(html).toContain("motion-off");
    expect(html).toContain("Quote withheld");
    expect(html).toContain("WAITING FOR POLICY");
  });
  it("labels baseline mode as bypassing the neural network", () => {
    const state = {
      ...EMPTY_STATE,
      visual,
      gate: { ...EMPTY_STATE.gate, mode: "uniform" as const },
    };
    const html = renderToStaticMarkup(
      createElement(SignalFlow, { state, ready: true, nextRefresh: 0 }),
    );
    expect(html).toContain("Neural gate bypassed");
    expect(html).not.toContain("data-activation=");
  });
});
