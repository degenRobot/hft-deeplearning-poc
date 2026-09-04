export interface TrainingEventCounts {
  book: number;
  trade: number;
  total: number;
}

export interface TrainingSource {
  name: string;
  symbol: string;
  venue: string;
  url: string;
  recording_path: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  event_counts: TrainingEventCounts;
}

export interface TrainingDataset {
  feature_names: string[];
  frame_seconds: number;
  lookback_frames: number;
  horizon_frames: number;
  frames: number;
  examples: number;
  train_examples: number;
  validation_examples: number;
}

export interface TrainingRun {
  seed: number;
  epochs: number;
  learning_rate: number;
  parameter_count: number;
  first_train_loss: number;
  last_train_loss: number;
  validation_loss: number;
  model_path: string;
}

export interface TrainingReceipt {
  schema_version: 1;
  generated_at: string;
  source: TrainingSource;
  dataset: TrainingDataset;
  training: TrainingRun;
  limitations: string[];
}

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const stringValue = (value: unknown) =>
  typeof value === "string" ? value : null;

const numberValue = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const integerValue = (value: unknown) => {
  const number = numberValue(value);
  return number !== null && Number.isInteger(number) ? number : null;
};

const stringArray = (value: unknown): string[] | null =>
  Array.isArray(value) &&
  value.every((item): item is string => typeof item === "string")
    ? value
    : null;

const readFields = <T extends readonly string[], V>(
  record: UnknownRecord,
  keys: T,
  read: (value: unknown) => V | null,
  valid: (value: V) => boolean = () => true,
) => {
  const values = keys.map((key) => read(record[key]));
  return values.every((value) => value !== null && valid(value))
    ? (Object.fromEntries(
        keys.map((key, index) => [key, values[index]]),
      ) as Record<T[number], V>)
    : null;
};

function readEventCounts(value: unknown): TrainingEventCounts | null {
  return isRecord(value)
    ? readFields(
        value,
        ["book", "trade", "total"] as const,
        integerValue,
        (number) => number >= 0,
      )
    : null;
}

function readSource(value: unknown): TrainingSource | null {
  if (!isRecord(value)) return null;
  const strings = readFields(
    value,
    [
      "name",
      "symbol",
      "venue",
      "url",
      "recording_path",
      "started_at",
      "ended_at",
    ] as const,
    stringValue,
  );
  const duration = readFields(
    value,
    ["duration_seconds"] as const,
    numberValue,
    (number) => number >= 0,
  );
  const eventCounts = readEventCounts(value.event_counts);
  if (
    !strings ||
    !duration ||
    !/^https?:\/\//i.test(strings.url) ||
    eventCounts === null
  ) {
    return null;
  }
  return {
    ...strings,
    duration_seconds: duration.duration_seconds,
    event_counts: eventCounts,
  };
}

function readDataset(value: unknown): TrainingDataset | null {
  if (!isRecord(value)) return null;
  const featureNames = stringArray(value.feature_names);
  if (!featureNames) return null;
  const dimensions = readFields(
    value,
    ["frame_seconds", "lookback_frames", "horizon_frames"] as const,
    numberValue,
    (number) => number > 0,
  );
  const counts = readFields(
    value,
    ["frames", "examples", "train_examples", "validation_examples"] as const,
    numberValue,
    (number) => number >= 0,
  );
  if (!dimensions || !counts) return null;
  return {
    feature_names: featureNames,
    ...dimensions,
    ...counts,
  };
}

function readTraining(value: unknown): TrainingRun | null {
  if (!isRecord(value)) return null;
  const seed = readFields(value, ["seed"] as const, integerValue);
  const positiveValues = readFields(
    value,
    ["epochs", "learning_rate"] as const,
    numberValue,
    (number) => number > 0,
  );
  const parameterCount = readFields(
    value,
    ["parameter_count"] as const,
    integerValue,
    (number) => number >= 0,
  );
  const losses = readFields(
    value,
    ["first_train_loss", "last_train_loss", "validation_loss"] as const,
    numberValue,
    () => true,
  );
  const modelPath = stringValue(value.model_path);
  if (!seed || !positiveValues || !parameterCount || !losses || !modelPath)
    return null;
  return {
    ...seed,
    ...positiveValues,
    ...parameterCount,
    ...losses,
    model_path: modelPath,
  };
}

/** Normalize the public receipt so rendering never trusts arbitrary API data. */
export function normalizeTrainingReceipt(
  input: unknown,
): TrainingReceipt | null {
  if (!isRecord(input) || input.schema_version !== 1) return null;
  const generatedAt = stringValue(input.generated_at);
  const source = readSource(input.source);
  const dataset = readDataset(input.dataset);
  const training = readTraining(input.training);
  const limitations = stringArray(input.limitations);
  if (
    generatedAt === null ||
    source === null ||
    dataset === null ||
    training === null ||
    limitations === null
  ) {
    return null;
  }
  return {
    schema_version: 1,
    generated_at: generatedAt,
    source,
    dataset,
    training,
    limitations,
  };
}

export const formatTrainingCount = (value: number) =>
  Number.isFinite(value) ? Math.round(value).toLocaleString("en-US") : "—";

export const formatTrainingDuration = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
};

export const formatTrainingLoss = (value: number) =>
  Number.isFinite(value) ? value.toFixed(4) : "—";
