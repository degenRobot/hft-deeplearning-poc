import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TrainingSection } from "../components/TrainingSection";
import { GateSection } from "../components/DashboardSections";
import { useTrainingReceipt } from "../hooks/useTrainingReceipt";
import { normalizeTrainingReceipt } from "../lib/training";
import { EMPTY_STATE } from "../lib/types";
import archived from "../../artifacts/training-demo.json";

vi.mock("../hooks/useTrainingReceipt", () => ({ useTrainingReceipt: vi.fn() }));
afterEach(() => vi.resetAllMocks());

function renderTraining(loading = false) {
  vi.mocked(useTrainingReceipt).mockReturnValue({
    receipt: loading ? null : normalizeTrainingReceipt(archived),
    loading,
    error: "",
  });
  return renderToStaticMarkup(createElement(TrainingSection));
}

describe("experiment story", () => {
  it("shows the rejected v2 captures without attributing archived training to them", () => {
    const html = renderTraining();
    expect(html).toContain("Three captures, no new model");
    expect(html).toContain("Captures accepted</span><strong>0</strong>");
    expect(html).toContain(
      "Experimental models trained</span><strong>0</strong>",
    );
    expect(html).toContain("<td>0</td><td>16</td>");
    expect(html).toContain("<td>3</td><td>17</td>");
    expect(html).toContain("<td>24</td><td>0</td>");
    expect(html).toContain("30 training and 15 validation");
    expect(html).toContain(
      '<details class="training-archive"><summary>Archived v1',
    );
    expect(html).toContain("21,443 parameters");
    expect(html).not.toContain("RUN IT LOCALLY");
    expect(html).not.toContain("latest");
  });

  it("keeps the v2 outcome visible while the archived receipt loads", () => {
    const html = renderTraining(true);
    expect(html).toContain("Loading the archived v1 receipt");
    expect(html).toContain("Three captures, no new model");
    expect(html).not.toContain("Loss / model");
  });

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
    expect(html).toContain("offline experiments below do not replace it");
  });
});
