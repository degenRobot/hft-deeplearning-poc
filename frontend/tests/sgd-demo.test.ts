import { describe, expect, it } from "vitest";
import {
  initialSgdState,
  sgdStep,
  sgdLoss,
  SGD_POINTS,
  SGD_STEPS,
} from "../lib/sgdDemo";

const run = (rate: number) => {
  let state = initialSgdState();
  for (let i = 0; i < SGD_STEPS; i++) state = sgdStep(state, rate);
  return state;
};
describe("SGD illustration", () => {
  it("uses the gradient of the sampled squared error", () => {
    const before = initialSgdState();
    const after = sgdStep(before, 0.1);
    const batchLoss = (w: number, b: number) =>
      after.batch.reduce((sum, index) => {
        const { x, y } = SGD_POINTS[index];
        return sum + (w * x + b - y) ** 2 / after.batch.length;
      }, 0);
    const epsilon = 1e-5;
    const dw =
      (batchLoss(before.weight + epsilon, before.bias) -
        batchLoss(before.weight - epsilon, before.bias)) /
      (2 * epsilon);
    const db =
      (batchLoss(before.weight, before.bias + epsilon) -
        batchLoss(before.weight, before.bias - epsilon)) /
      (2 * epsilon);
    expect(after.weight).toBeCloseTo(before.weight - 0.1 * dw, 8);
    expect(after.bias).toBeCloseTo(before.bias - 0.1 * db, 8);
    expect(after.history.at(-1)?.loss).toBe(sgdLoss(after.weight, after.bias));
    expect(before.history).toHaveLength(1);
  });
  it("shows slow progress, convergence and divergence on the same data", () => {
    const slow = run(0.005);
    const steady = run(0.1);
    const large = run(1.2);
    expect(steady.history.at(-1)!.loss).toBeLessThan(0.02);
    expect(slow.history.at(-1)!.loss).toBeGreaterThan(
      steady.history.at(-1)!.loss * 10,
    );
    expect(large.diverged).toBe(true);
    expect(large.history.at(-1)!.loss).toBeGreaterThan(1e4);
    expect(Number.isFinite(large.weight)).toBe(true);
    expect(sgdStep(large, 0.1)).toBe(large);
    expect(sgdStep(steady, 0.1)).toBe(steady);
    expect(run(0.1)).toEqual(steady);
  });
  it("changes the next update without resetting weights or the sampled batches", () => {
    const state = sgdStep(initialSgdState(), 0.1);
    const slow = sgdStep(state, 0.01);
    const fast = sgdStep(state, 0.1);
    expect(slow.batch).toEqual(fast.batch);
    expect(fast.weight - state.weight).toBeCloseTo(
      10 * (slow.weight - state.weight),
      8,
    );
    expect(slow.step).toBe(2);
    expect(slow.history.map((point) => point.rate)).toEqual([null, 0.1, 0.01]);
  });
  it.each([0, -1, NaN, Infinity, 2])("rejects an invalid rate %s", (rate) => {
    expect(() => sgdStep(initialSgdState(), rate)).toThrow(RangeError);
  });
});
