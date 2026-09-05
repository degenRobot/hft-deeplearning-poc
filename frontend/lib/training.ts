export interface TrainingSource {
  symbol: string;
  venue: string;
  url: string;
  duration_seconds: number;
  event_counts: { total: number };
}

export interface TrainingDataset {
  train_examples: number;
  validation_examples: number;
}

export interface TrainingRun {
  parameter_count: number;
  first_train_loss: number;
  last_train_loss: number;
  validation_loss: number;
}

export interface TrainingReceipt {
  schema_version: 1;
  source: TrainingSource;
  dataset: TrainingDataset;
  training: TrainingRun;
  limitations: string[];
}

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const asTyped = <T>(value: UnknownRecord) => value as unknown as T;

/** Check the local API fields used for headline metrics before rendering. */
export function normalizeTrainingReceipt(
  input: unknown,
): TrainingReceipt | null {
  if (
    !isRecord(input) ||
    input.schema_version !== 1 ||
    !Array.isArray(input.limitations)
  )
    return null;
  const sections = [input.source, input.dataset, input.training];
  if (!sections.every(isRecord)) return null;
  const [source, dataset, training] = sections;
  const headlineNumbers = [
    source.duration_seconds,
    isRecord(source.event_counts) ? source.event_counts.total : undefined,
    dataset.train_examples,
    dataset.validation_examples,
    training.first_train_loss,
    training.last_train_loss,
    training.validation_loss,
    training.parameter_count,
  ];
  const headlineText = [source.symbol, source.venue, ...input.limitations];
  if (
    !headlineNumbers.every(isFiniteNumber) ||
    !headlineText.every((value) => typeof value === "string") ||
    typeof source.url !== "string" ||
    !source.url.startsWith("https://")
  )
    return null;
  return {
    schema_version: 1,
    source: asTyped<TrainingSource>(source),
    dataset: asTyped<TrainingDataset>(dataset),
    training: asTyped<TrainingRun>(training),
    limitations: input.limitations as string[],
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
