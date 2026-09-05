import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  ExpertSignalHeatmap,
  signalCellColor,
  signalDirection,
} from "../components/ExpertSignalHeatmap";
import { SignalFlow } from "../components/SignalFlow";
import { EMPTY_STATE } from "../lib/types";
import { parseVisual, type VisualTelemetry } from "../lib/visual";
const event = {
  id: 1,
  timestamp_ms: 1000,
  kind: "book" as const,
  price: 100,
  size: 2,
  scores: [-0.5, 0, 1],
  signal: 0.1,
  gate_revision: 1,
};
const visual: VisualTelemetry = {
  events: [
    event,
    { ...event, id: 3, timestamp_ms: 1200, scores: [0.4, -0.2, 0.1] },
  ],
  window: [],
  proposed: { microprice: 0.4, flow: 0.3, reversion: 0.3 },
  previous: { microprice: 0.4, flow: 0.3, reversion: 0.3 },
  influence: 0.35,
  gate_timestamp_ms: 1000,
  snapshot_interval_ms: 100,
};
const neural = {
  score: 0.4,
  model_version: "tiny-expert-test",
  parameter_count: 41,
  inference_us: 3.5,
  inputs: [0.4, -0.2, 0.1],
};
const render = (value: VisualTelemetry | null) =>
  renderToStaticMarkup(createElement(ExpertSignalHeatmap, { visual: value }));
describe("event signal history", () => {
  it("renders only retained distinct IDs and never grows history when a snapshot repeats", () => {
    const html = render(visual);
    expect(html.match(/data-event-id=/g) ?? []).toHaveLength(6);
    expect(html).toContain('data-event-id="1"');
    expect(html).toContain('data-event-id="3"');
    expect(html).not.toContain('data-event-id="2"');
    expect(render(visual)).toBe(html);
    expect(html).toContain("5.0 events/s");
    expect(html).toContain("00:00:01.000 UTC");
    expect(html).toContain("100 ms UI snapshots");
    expect(html).toContain("not calibrated confidence");
    expect(html).toContain("Buy bias +0.4000");
    expect(html).toContain("Sell bias -0.2000");
  });
  it("clears every history cell when readiness is lost", () => {
    const html = renderToStaticMarkup(
      createElement(SignalFlow, {
        state: { ...EMPTY_STATE, visual: { ...visual, neural_expert: neural } },
        ready: false,
        nextRefresh: 0,
      }),
    );
    expect(html).not.toContain("data-event-id=");
    expect(html).not.toContain("tiny-expert-test");
    expect(html).toContain(
      "New columns appear only when real event IDs arrive",
    );
  });
  it("adds the shadow row without inventing historical scores or including it in the quote mix", () => {
    const current = {
      ...visual,
      neural_expert: neural,
      events: [event, { ...visual.events[1], neural_score: 0.4 }],
    };
    const html = render(current);
    expect(html.match(/data-event-id=/g) ?? []).toHaveLength(8);
    expect(html).toContain("No neural score recorded");
    expect(html).toContain("Tiny NN · shadow");
    const flow = renderToStaticMarkup(
      createElement(SignalFlow, {
        state: { ...EMPTY_STATE, visual: current },
        ready: true,
        nextRefresh: 0,
      }),
    );
    expect(flow).toContain("HAND-CODED RULE");
    expect(flow).toContain("learned combination of the three rule scores");
    expect(flow).toContain(
      "excluded from the expert allocation, mixed signal and synthetic quote",
    );
    expect(flow).toContain("41 parameters");
    expect(flow).toContain("Microprice +0.4000");
    expect(flow).toContain("Three hand-coded experts");
  });
  it("uses equal intensity for equal signed magnitude on every row", () => {
    expect(signalCellColor(-0.5).split(", ").at(-1)).toBe(
      signalCellColor(0.5).split(", ").at(-1),
    );
    expect(signalCellColor(null)).toBe("transparent");
    expect(signalDirection(0)).toBe("Neutral");
  });
});
describe("optional neural expert contract", () => {
  it("retains legacy payloads and accepts exactly 64 ordered events", () => {
    expect(parseVisual(visual)).toEqual(visual);
    expect(parseVisual({ ...visual, neural_expert: null })).not.toBeNull();
    const events = Array.from({ length: 64 }, (_, index) => ({
      ...event,
      id: index + 1,
      timestamp_ms: 1000 + index,
      neural_score: null,
    }));
    expect(parseVisual({ ...visual, events })).not.toBeNull();
    expect(
      parseVisual({
        ...visual,
        events: [...events, { ...event, id: 65, timestamp_ms: 1065 }],
      }),
    ).toBeNull();
    expect(parseVisual({ ...visual, neural_expert: neural })).not.toBeNull();
  });
  it.each([
    { score: 1.01 },
    { score: NaN },
    { inputs: [1, 2, 3] },
    { inputs: [0, 1] },
    { model_version: "" },
    { inference_us: -1 },
    { parameter_count: 0 },
    { parameter_count: 1.5 },
  ])("rejects malformed neural metadata %s", (change) => {
    expect(
      parseVisual({ ...visual, neural_expert: { ...neural, ...change } }),
    ).toBeNull();
  });
  it.each([NaN, 1.01, -1.01, "0.5"])(
    "rejects malformed event neural score %s",
    (score) => {
      expect(
        parseVisual({ ...visual, events: [{ ...event, neural_score: score }] }),
      ).toBeNull();
    },
  );
});
