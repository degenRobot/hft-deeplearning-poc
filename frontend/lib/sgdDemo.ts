/** Reproducible browser SGD, independent of the backend trading model. */
export const SGD_STEPS = 120;
export const SGD_BATCH_SIZE = 8;
export const SGD_POINTS = Array.from({ length: 48 }, (_, i) => {
  const x = -1 + (2 * i) / 47;
  return { x, y: 1.4 * x + 0.5 + 0.14 * Math.sin(i * 7.3) };
});
export type SgdPoint = { x: number; y: number };
export type SgdDataset = {
  id: string;
  label: string;
  points: SgdPoint[];
  validation: SgdPoint[];
  baselineLoss: number | null;
  capturedAt?: number;
  xScale?: number;
  yScale?: number;
};
export const SYNTHETIC_DATASET: SgdDataset = {
  id: "synthetic",
  label: "Synthetic example",
  points: SGD_POINTS,
  validation: [],
  baselineLoss: null,
};
export type SgdState = {
  weight: number;
  bias: number;
  step: number;
  seed: number;
  batch: number[];
  gradient: [number, number];
  history: {
    step: number;
    loss: number;
    validationLoss: number | null;
    rate: number | null;
  }[];
  diverged: boolean;
};
export function sgdLoss(
  weight: number,
  bias: number,
  points: SgdPoint[] = SGD_POINTS,
) {
  return (
    points.reduce(
      (sum, point) => sum + (weight * point.x + bias - point.y) ** 2,
      0,
    ) / points.length
  );
}
export function initialSgdState(dataset = SYNTHETIC_DATASET): SgdState {
  return {
    weight: -0.8,
    bias: -0.5,
    step: 0,
    seed: 42,
    batch: [],
    gradient: [0, 0],
    history: [
      {
        step: 0,
        loss: sgdLoss(-0.8, -0.5, dataset.points),
        validationLoss: dataset.validation.length
          ? sgdLoss(-0.8, -0.5, dataset.validation)
          : null,
        rate: null,
      },
    ],
    diverged: false,
  };
}
export function sgdStep(
  state: SgdState,
  rate: number,
  dataset = SYNTHETIC_DATASET,
): SgdState {
  if (state.diverged || state.step >= SGD_STEPS) return state;
  if (!Number.isFinite(rate) || rate <= 0 || rate > 1.5)
    throw new RangeError("Learning rate must be between 0 and 1.5");
  let seed = state.seed;
  const batch = Array.from({ length: SGD_BATCH_SIZE }, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return Math.floor((seed / 2 ** 32) * dataset.points.length);
  });
  const gradient: [number, number] = [0, 0];
  for (const index of batch) {
    const point = dataset.points[index];
    const error = state.weight * point.x + state.bias - point.y;
    gradient[0] += (2 * error * point.x) / SGD_BATCH_SIZE;
    gradient[1] += (2 * error) / SGD_BATCH_SIZE;
  }
  const weight = state.weight - rate * gradient[0];
  const bias = state.bias - rate * gradient[1];
  const loss = sgdLoss(weight, bias, dataset.points);
  const validationLoss = dataset.validation.length
    ? sgdLoss(weight, bias, dataset.validation)
    : null;
  return {
    weight,
    bias,
    step: state.step + 1,
    seed,
    batch,
    gradient,
    history: [
      ...state.history,
      { step: state.step + 1, loss, validationLoss, rate },
    ],
    diverged: loss > 1e4,
  };
}
