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
  continuous?: boolean;
  run_id: string | null;
  status: "idle" | "running" | "completed" | "failed" | "stopped";
  backend: "local" | "modal";
  can_stop?: boolean;
  updated_at: string | null;
  dataset: null | {
    symbol: string;
    event_count: number;
    frame_count: number;
    supervised_examples: number;
    rl_examples: number;
    holdout_examples: number;
    sha256: string;
    source_mode?: string;
    limitations?: string[];
    feature_names?: string[];
    hidden_sizes?: [number, number];
    parameter_count?: number;
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
  progress?: {
    stage: string;
    completed_steps: number;
    total_steps: number | null;
    percent: number | null;
    elapsed_seconds: number | null;
    remote_elapsed_seconds: number | null;
    compute_estimate_usd: number | null;
  };
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
function validStep(
  v: unknown,
  sizes: [number, number] = [64, 32],
): v is TrainingStep {
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
    !vector(v.activations.hidden_1, Math.min(sizes[0], 64)) ||
    !vector(v.activations.hidden_2, Math.min(sizes[1], 32)) ||
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
  if (v.continuous !== undefined && typeof v.continuous !== "boolean")
    return null;
  if (v.can_stop !== undefined && typeof v.can_stop !== "boolean") return null;
  if (v.progress !== undefined) {
    const p = v.progress;
    if (
      !object(p) ||
      ![
        "idle",
        "preparing",
        "supervised",
        "rl",
        "finalizing",
        "completed",
        "failed",
        "stopped",
      ].includes(String(p.stage)) ||
      !count(p.completed_steps) ||
      !(
        p.total_steps === null ||
        (count(p.total_steps) && p.total_steps > 0)
      ) ||
      !(
        p.percent === null ||
        (finite(p.percent) && p.percent >= 0 && p.percent <= 100)
      ) ||
      ![
        p.elapsed_seconds,
        p.remote_elapsed_seconds,
        p.compute_estimate_usd,
      ].every((x) => x === null || (finite(x) && x >= 0))
    )
      return null;
    if (
      (p.total_steps !== null &&
        p.completed_steps > (p.total_steps as number)) ||
      (v.status === "running" &&
        (["completed", "failed", "stopped", "idle"].includes(String(p.stage)) ||
          p.percent === 100)) ||
      (v.status !== "running" && p.stage !== v.status)
    )
      return null;
  }
  let sizes: [number, number] = [64, 32];
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
  if (object(v.dataset)) {
    if (
      !(
        v.dataset.source_mode === undefined ||
        typeof v.dataset.source_mode === "string"
      ) ||
      !(
        v.dataset.limitations === undefined ||
        (Array.isArray(v.dataset.limitations) &&
          v.dataset.limitations.every((x) => typeof x === "string"))
      ) ||
      !(
        v.dataset.feature_names === undefined ||
        (Array.isArray(v.dataset.feature_names) &&
          v.dataset.feature_names.length === 10 &&
          v.dataset.feature_names.every(
            (x) => typeof x === "string" && x.length > 0,
          ))
      )
    )
      return null;
    if (v.dataset.hidden_sizes !== undefined) {
      if (
        !Array.isArray(v.dataset.hidden_sizes) ||
        v.dataset.hidden_sizes.length !== 2 ||
        !v.dataset.hidden_sizes.every((x) => count(x) && x >= 8 && x <= 1024)
      )
        return null;
      sizes = v.dataset.hidden_sizes as [number, number];
    }
    if (
      v.dataset.parameter_count !== undefined &&
      (!count(v.dataset.parameter_count) ||
        v.dataset.parameter_count !== trainingParameterCount(...sizes))
    )
      return null;
  }
  if (
    !(v.latest === null || validStep(v.latest, sizes)) ||
    !Array.isArray(v.history) ||
    v.history.length > 400 ||
    !v.history.every((step) => validStep(step, sizes))
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

export interface TrainingOptions {
  backend: "local" | "modal";
  hidden_1: number;
  hidden_2: number;
  epochs: number;
  learning_rate: number;
}
export const DEFAULT_TRAINING_OPTIONS: TrainingOptions = {
  backend: "local",
  hidden_1: 64,
  hidden_2: 32,
  epochs: 12,
  learning_rate: 0.001,
};
export interface TrainingSettings {
  modal: { configured: boolean; available: boolean };
  defaults: TrainingOptions;
  limits: { hidden_min: number; hidden_max: number; epochs_max: number };
  resources: { cpu: number; memory_gib: number; timeout_seconds: number };
  pricing?: {
    checked_on: string;
    source_url: string;
    cpu_core_second_usd: number;
    memory_gib_second_usd: number;
    cpu_cores: number;
    memory_gib: number;
    timeout_seconds: number;
  };
}
export const trainingParameterCount = (first: number, second: number) =>
  301 * first + (first + 1) * second + (second + 1) * 3;
export function validTrainingOptions(
  v: TrainingOptions,
  limits = { hidden_min: 8, hidden_max: 1024, epochs_max: 50 },
) {
  return (
    ["local", "modal"].includes(v.backend) &&
    [v.hidden_1, v.hidden_2].every(
      (x) =>
        Number.isInteger(x) && x >= limits.hidden_min && x <= limits.hidden_max,
    ) &&
    Number.isInteger(v.epochs) &&
    v.epochs >= 1 &&
    v.epochs <= limits.epochs_max &&
    Number.isFinite(v.learning_rate) &&
    v.learning_rate >= 0.00001 &&
    v.learning_rate <= 0.01
  );
}
export function parseTrainingSettings(v: unknown): TrainingSettings | null {
  if (
    !object(v) ||
    !object(v.modal) ||
    typeof v.modal.configured !== "boolean" ||
    typeof v.modal.available !== "boolean" ||
    !object(v.defaults) ||
    !object(v.limits) ||
    !object(v.resources)
  )
    return null;
  const limits = v.limits;
  if (
    ![limits.hidden_min, limits.hidden_max, limits.epochs_max].every(count) ||
    (limits.hidden_min as number) < 8 ||
    (limits.hidden_max as number) > 1024 ||
    (limits.hidden_min as number) > (limits.hidden_max as number) ||
    (limits.epochs_max as number) < 1 ||
    (limits.epochs_max as number) > 50
  )
    return null;
  if (
    !validTrainingOptions(
      v.defaults as unknown as TrainingOptions,
      limits as unknown as TrainingSettings["limits"],
    ) ||
    ![
      v.resources.cpu,
      v.resources.memory_gib,
      v.resources.timeout_seconds,
    ].every((x) => finite(x) && x > 0)
  )
    return null;
  if (
    v.pricing !== undefined &&
    (!object(v.pricing) ||
      typeof v.pricing.checked_on !== "string" ||
      v.pricing.source_url !== "https://modal.com/pricing" ||
      ![
        v.pricing.cpu_core_second_usd,
        v.pricing.memory_gib_second_usd,
        v.pricing.cpu_cores,
        v.pricing.memory_gib,
        v.pricing.timeout_seconds,
      ].every((x) => finite(x) && x > 0))
  )
    return null;
  return {
    modal: { configured: v.modal.configured, available: v.modal.available },
    defaults: {
      backend: v.defaults.backend as TrainingOptions["backend"],
      hidden_1: v.defaults.hidden_1 as number,
      hidden_2: v.defaults.hidden_2 as number,
      epochs: v.defaults.epochs as number,
      learning_rate: v.defaults.learning_rate as number,
    },
    limits: limits as unknown as TrainingSettings["limits"],
    resources: v.resources as unknown as TrainingSettings["resources"],
    pricing: v.pricing as TrainingSettings["pricing"],
  };
}
export function trainingCanStop(data: LiveTraining | null) {
  return (
    data?.status === "running" &&
    (data.can_stop === true ||
      (data.backend === "local" && data.can_stop === undefined))
  );
}
