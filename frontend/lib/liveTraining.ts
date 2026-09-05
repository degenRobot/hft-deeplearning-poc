export interface TrainingStep {
  step: number;
  phase: "supervised" | "rl";
  loss: number;
  reward: number | null;
  action: number | null;
  input_end_ts_ms: number;
  target_ts_ms: number;
  features: number[][];
  activations: { hidden_1: number[]; hidden_2: number[] };
  outputs_before: number[];
  outputs_after: number[];
  layers: {
    name: string;
    gradient_norm: number;
    weight_delta_norm: number;
    weight_norm: number;
  }[];
}
export interface LiveTraining {
  schema_version: 1;
  run_id: string | null;
  status: "idle" | "running" | "completed" | "failed" | "stopped";
  backend: "local" | "modal";
  updated_at: string | null;
  dataset: null | {
    symbol: string;
    event_count: number;
    frame_count: number;
    supervised_examples: number;
    rl_examples: number;
    holdout_examples: number;
    sha256: string;
  };
  latest: TrainingStep | null;
  history: TrainingStep[];
  evaluation: null | {
    supervised: number;
    adapted: number;
    uniform: number;
    microprice: number;
    trade_flow: number;
    reversion: number;
  };
  error: string | null;
}
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const count = (v: unknown): v is number =>
  finite(v) && Number.isSafeInteger(v) && v >= 0;
const vector = (v: unknown, n: number) =>
  Array.isArray(v) && v.length === n && v.every(finite);
const probabilities = (v: unknown) =>
  vector(v, 3) &&
  (v as number[]).every((x) => x >= 0 && x <= 1) &&
  Math.abs((v as number[]).reduce((a, b) => a + b, 0) - 1) < 0.001;
function validStep(v: unknown): v is TrainingStep {
  if (
    !object(v) ||
    !count(v.step) ||
    !["supervised", "rl"].includes(String(v.phase)) ||
    !finite(v.loss)
  )
    return false;
  if (
    !(v.reward === null || finite(v.reward)) ||
    !(v.action === null || (count(v.action) && v.action < 3))
  )
    return false;
  if (
    !finite(v.input_end_ts_ms) ||
    !finite(v.target_ts_ms) ||
    v.input_end_ts_ms <= 0 ||
    v.target_ts_ms < v.input_end_ts_ms ||
    !Number.isFinite(new Date(v.target_ts_ms).getTime())
  )
    return false;
  if (
    !Array.isArray(v.features) ||
    v.features.length !== 30 ||
    !v.features.every((row) => vector(row, 10))
  )
    return false;
  if (
    !object(v.activations) ||
    !vector(v.activations.hidden_1, 64) ||
    !vector(v.activations.hidden_2, 32) ||
    !(v.activations.hidden_1 as number[]).every((x) => x >= 0) ||
    !(v.activations.hidden_2 as number[]).every((x) => x >= 0)
  )
    return false;
  if (!probabilities(v.outputs_before) || !probabilities(v.outputs_after))
    return false;
  return (
    Array.isArray(v.layers) &&
    v.layers.length > 0 &&
    v.layers.length <= 12 &&
    v.layers.every(
      (layer) =>
        object(layer) &&
        typeof layer.name === "string" &&
        [layer.gradient_norm, layer.weight_delta_norm, layer.weight_norm].every(
          (x) => finite(x) && x >= 0,
        ),
    )
  );
}
export function parseLiveTraining(v: unknown): LiveTraining | null {
  if (
    !object(v) ||
    v.schema_version !== 1 ||
    !["idle", "running", "completed", "failed", "stopped"].includes(
      String(v.status),
    ) ||
    !["local", "modal"].includes(String(v.backend))
  )
    return null;
  if (
    !(v.run_id === null || typeof v.run_id === "string") ||
    !(v.error === null || typeof v.error === "string") ||
    !(
      v.updated_at === null ||
      (typeof v.updated_at === "string" &&
        Number.isFinite(Date.parse(v.updated_at)))
    )
  )
    return null;
  if (v.dataset !== null) {
    if (
      !object(v.dataset) ||
      typeof v.dataset.symbol !== "string" ||
      typeof v.dataset.sha256 !== "string" ||
      !/^[a-f\d]{64}$/i.test(v.dataset.sha256) ||
      ![
        v.dataset.event_count,
        v.dataset.frame_count,
        v.dataset.supervised_examples,
        v.dataset.rl_examples,
        v.dataset.holdout_examples,
      ].every(count)
    )
      return null;
  }
  if (
    !(v.latest === null || validStep(v.latest)) ||
    !Array.isArray(v.history) ||
    v.history.length > 400 ||
    !v.history.every(validStep)
  )
    return null;
  if (
    v.evaluation !== null &&
    (!object(v.evaluation) ||
      ![
        "supervised",
        "adapted",
        "uniform",
        "microprice",
        "trade_flow",
        "reversion",
      ].every((k) => finite((v.evaluation as Record<string, unknown>)[k])))
  )
    return null;
  return v as unknown as LiveTraining;
}
export const TRAINING_STALE_MS = 10_000;
export function trainingIsStale(value: LiveTraining, now: number) {
  return (
    value.status === "running" &&
    value.updated_at !== null &&
    now - Date.parse(value.updated_at) > TRAINING_STALE_MS
  );
}
