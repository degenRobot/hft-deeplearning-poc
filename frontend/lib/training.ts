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

function readEventCounts(value: unknown): TrainingEventCounts | null {
  if (!isRecord(value)) return null;
  const book = integerValue(value.book);
  const trade = integerValue(value.trade);
  const total = integerValue(value.total);
  if (
    book === null ||
    trade === null ||
    total === null ||
    book < 0 ||
    trade < 0 ||
    total < 0
  ) {
    return null;
  }
  return { book, trade, total };
}

function readSource(value: unknown): TrainingSource | null {
  if (!isRecord(value)) return null;
  const name = stringValue(value.name);
  const symbol = stringValue(value.symbol);
  const venue = stringValue(value.venue);
  const url = stringValue(value.url);
  const recordingPath = stringValue(value.recording_path);
  const startedAt = stringValue(value.started_at);
  const endedAt = stringValue(value.ended_at);
  const duration = numberValue(value.duration_seconds);
  const eventCounts = readEventCounts(value.event_counts);
  if (
    name === null ||
    symbol === null ||
    venue === null ||
    url === null ||
    !/^https?:\/\//i.test(url) ||
    recordingPath === null ||
    startedAt === null ||
    endedAt === null ||
    duration === null ||
    duration < 0 ||
    eventCounts === null
  ) {
    return null;
  }
  return {
    name,
    symbol,
    venue,
    url,
    recording_path: recordingPath,
    started_at: startedAt,
    ended_at: endedAt,
    duration_seconds: duration,
    event_counts: eventCounts,
  };
}

function readDataset(value: unknown): TrainingDataset | null {
  if (!isRecord(value) || !Array.isArray(value.feature_names)) return null;
  const featureNames = value.feature_names.filter(
    (name): name is string => typeof name === "string",
  );
  if (featureNames.length !== value.feature_names.length) return null;
  const frameSeconds = numberValue(value.frame_seconds);
  const lookbackFrames = integerValue(value.lookback_frames);
  const horizonFrames = integerValue(value.horizon_frames);
  const frames = integerValue(value.frames);
  const examples = integerValue(value.examples);
  const trainExamples = integerValue(value.train_examples);
  const validationExamples = integerValue(value.validation_examples);
  if (
    frameSeconds === null ||
    frameSeconds <= 0 ||
    lookbackFrames === null ||
    lookbackFrames <= 0 ||
    horizonFrames === null ||
    horizonFrames <= 0 ||
    frames === null ||
    frames < 0 ||
    examples === null ||
    examples < 0 ||
    trainExamples === null ||
    trainExamples < 0 ||
    validationExamples === null ||
    validationExamples < 0
  ) {
    return null;
  }
  return {
    feature_names: featureNames,
    frame_seconds: frameSeconds,
    lookback_frames: lookbackFrames,
    horizon_frames: horizonFrames,
    frames,
    examples,
    train_examples: trainExamples,
    validation_examples: validationExamples,
  };
}

function readTraining(value: unknown): TrainingRun | null {
  if (!isRecord(value)) return null;
  const seed = integerValue(value.seed);
  const epochs = integerValue(value.epochs);
  const learningRate = numberValue(value.learning_rate);
  const parameterCount = integerValue(value.parameter_count);
  const firstTrainLoss = numberValue(value.first_train_loss);
  const lastTrainLoss = numberValue(value.last_train_loss);
  const validationLoss = numberValue(value.validation_loss);
  const modelPath = stringValue(value.model_path);
  if (
    seed === null ||
    epochs === null ||
    epochs <= 0 ||
    learningRate === null ||
    learningRate <= 0 ||
    parameterCount === null ||
    parameterCount < 0 ||
    firstTrainLoss === null ||
    lastTrainLoss === null ||
    validationLoss === null ||
    modelPath === null
  ) {
    return null;
  }
  return {
    seed,
    epochs,
    learning_rate: learningRate,
    parameter_count: parameterCount,
    first_train_loss: firstTrainLoss,
    last_train_loss: lastTrainLoss,
    validation_loss: validationLoss,
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
  const limitations = Array.isArray(input.limitations)
    ? input.limitations.filter(
        (limitation): limitation is string => typeof limitation === "string",
      )
    : null;
  if (
    generatedAt === null ||
    source === null ||
    dataset === null ||
    training === null ||
    limitations === null ||
    limitations.length !== (input.limitations as unknown[]).length
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
