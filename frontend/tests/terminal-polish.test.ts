import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import { SignalFlow } from "../components/SignalFlow";
import { ContextTip } from "../components/ContextTip";
import { EMPTY_STATE } from "../lib/types";

describe("terminal reading order", () => {
  it("puts incoming data before experts, control and the resulting signal in the DOM", () => {
    const html = renderToStaticMarkup(
      createElement(SignalFlow, {
        state: EMPTY_STATE,
        ready: false,
        nextRefresh: 0,
      }),
    );
    const stages = [
      "Event stream",
      "candle-chart",
      "Three hand-coded experts",
      "MODEL CONTROL",
      "One quote signal",
      "expert-history",
    ];
    const positions = stages.map((stage) => html.indexOf(stage));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });
  it("opens model context on activation and dismisses it with Escape without a request", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const stderr = vi.spyOn(console, "error").mockImplementation(() => {});
    let renderer!: ReactTestRenderer;
    try {
      await act(async () => {
        renderer = create(
          createElement(
            ContextTip,
            { label: "How could the model grow?" },
            "Illustrative future inputs",
          ),
        );
      });
      const button = renderer.root.findByType("button");
      const wrapper = renderer.root.findByProps({ className: "context-tip" });
      const content = renderer.root.findByProps({
        className: "context-tip-content",
      });
      expect(button.props["aria-expanded"]).toBe(false);
      expect(content.props.hidden).toBe(true);
      await act(async () => button.props.onClick());
      expect(button.props["aria-expanded"]).toBe(true);
      expect(button.props["aria-controls"]).toBe(content.props.id);
      expect(content.props.hidden).toBe(false);
      await act(async () =>
        wrapper.props.onKeyDown({ key: "Escape", stopPropagation: vi.fn() }),
      );
      expect(content.props.hidden).toBe(true);
      expect(fetch).not.toHaveBeenCalled();
    } finally {
      if (renderer) await act(async () => renderer.unmount());
      stderr.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});
