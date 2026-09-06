/** A reproducible mini-batch SGD illustration, independent of backend training. */
export const SGD_STEPS = 120;
export const SGD_BATCH_SIZE = 8;
export const SGD_POINTS = Array.from({ length: 48 }, (_, i) => {
  const x = -1 + (2 * i) / 47;
  return { x, y: 1.4 * x + 0.5 + 0.14 * Math.sin(i * 7.3) };
});
export type SgdState = {
  weight: number;
  bias: number;
  step: number;
  seed: number;
  batch: number[];
  gradient: [number, number];
  history: { step: number; loss: number; rate: number | null }[];
  diverged: boolean;
};
export function sgdLoss(weight: number, bias: number) {
  return (
    SGD_POINTS.reduce(
      (sum, point) => sum + (weight * point.x + bias - point.y) ** 2,
      0,
    ) / SGD_POINTS.length
  );
}
export function initialSgdState(): SgdState {
  return {
    weight: -0.8,
    bias: -0.5,
    step: 0,
    seed: 42,
    batch: [],
    gradient: [0, 0],
    history: [{ step: 0, loss: sgdLoss(-0.8, -0.5), rate: null }],
    diverged: false,
  };
}
export function sgdStep(state: SgdState, rate: number): SgdState {
  if (state.diverged || state.step >= SGD_STEPS) return state;
  if (!Number.isFinite(rate) || rate <= 0 || rate > 1.5)
    throw new RangeError("Learning rate must be between 0 and 1.5");
  let seed = state.seed;
  const batch = Array.from({ length: SGD_BATCH_SIZE }, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return Math.floor((seed / 2 ** 32) * SGD_POINTS.length);
  });
  const gradient: [number, number] = [0, 0];
  for (const index of batch) {
    const point = SGD_POINTS[index];
    const error = state.weight * point.x + state.bias - point.y;
    gradient[0] += (2 * error * point.x) / SGD_BATCH_SIZE;
    gradient[1] += (2 * error) / SGD_BATCH_SIZE;
  }
  const weight = state.weight - rate * gradient[0];
  const bias = state.bias - rate * gradient[1];
  const loss = sgdLoss(weight, bias);
  return {
    weight,
    bias,
    step: state.step + 1,
    seed,
    batch,
    gradient,
    history: [...state.history, { step: state.step + 1, loss, rate }],
    diverged: loss > 1e4,
  };
}
