import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { GateSection } from "../components/DashboardSections";
import { EMPTY_STATE } from "../lib/types";

describe("runtime model context", () => {
  it("identifies the demo gate as the active runtime policy", () => {
    const html = renderToStaticMarkup(
      createElement(GateSection, {
        state: EMPTY_STATE,
        ready: false,
        nextRefresh: 0,
      }),
    );
    expect(html).toContain("active runtime policy");
    expect(html).toContain("Demo neural gate sets the mix");
    expect(html).toContain("Training Lab runs do not replace the live model");
  });
});
