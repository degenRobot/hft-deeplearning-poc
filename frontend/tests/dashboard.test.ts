import { describe, expect, it } from "vitest";
import { largestWeight } from "../components/DashboardSections";
import { EMPTY_STATE } from "../lib/types";

describe("dashboard derived values", () => {
  it("uses the largest current expert weight for gate confidence", () => {
    const state = {
      ...EMPTY_STATE,
      gate: {
        ...EMPTY_STATE.gate,
        weights: { microprice: 0.2, flow: 0.65, reversion: 0.15 },
      },
    };
    expect(largestWeight(state)).toBe(0.65);
  });
});
