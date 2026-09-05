import { DEFAULT_CONFIG, type AppConfig } from "./types";

export const CONFIG_RANGES = {
  higher_level_influence: { min: 0, max: 1 },
  gate_interval_ms: { min: 100, max: 60_000 },
  expert_strength: { min: 0, max: 5 },
  base_spread_bps: { min: 0.01, max: 1_000 },
  max_inventory: { min: 0.001, max: 1_000_000 },
  flow_window_trades: { min: 4, max: 512 },
} as const;

export type PresetId = "neural" | "uniform" | "fast";

export interface Preset {
  id: PresetId;
  label: string;
  description: string;
  patch: Pick<
    AppConfig,
    | "gate_mode"
    | "higher_level_influence"
    | "gate_interval_ms"
    | "expert_strength"
    | "flow_window_trades"
  >;
}

export const PRESETS: Preset[] = [
  {
    id: "neural",
    label: "Neural mix",
    description: "Balanced learned policy",
    patch: {
      gate_mode: "neural",
      higher_level_influence: 0.35,
      gate_interval_ms: 1000,
      expert_strength: 1,
      flow_window_trades: 64,
    },
  },
  {
    id: "uniform",
    label: "Uniform control",
    description: "Equal expert weights",
    patch: {
      gate_mode: "uniform",
      higher_level_influence: 0.35,
      gate_interval_ms: 1000,
      expert_strength: 1,
      flow_window_trades: 64,
    },
  },
  {
    id: "fast",
    label: "Fast reacting",
    description: "Short cadence, recent flow",
    patch: {
      gate_mode: "neural",
      gate_interval_ms: 500,
      higher_level_influence: 0.65,
      flow_window_trades: 16,
      expert_strength: 1.25,
    },
  },
];

const sources = ["replay", "binance"] as const;
const gateModes = ["neural", "uniform", "static"] as const;
const numericFields = [
  "higher_level_influence",
  "gate_interval_ms",
  "expert_strength",
  "base_spread_bps",
  "max_inventory",
  "flow_window_trades",
] as const;
const pick = <T extends string>(
  value: unknown,
  options: readonly T[],
  fallback: T,
) => (options.includes(value as T) ? (value as T) : fallback);

export function normalizeConfig(
  input: unknown,
  fallback: AppConfig = DEFAULT_CONFIG,
): AppConfig {
  const raw =
    input && typeof input === "object" ? (input as Partial<AppConfig>) : {};
  const numeric = Object.fromEntries(
    numericFields.map((key) => [key, numberOr(raw[key], fallback[key])]),
  );
  return {
    ...fallback,
    ...raw,
    ...numeric,
    source: pick(raw.source, sources, fallback.source),
    gate_mode: pick(raw.gate_mode, gateModes, fallback.gate_mode),
    symbol:
      typeof raw.symbol === "string"
        ? raw.symbol.toUpperCase()
        : fallback.symbol,
  } as AppConfig;
}

const numberOr = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

export function configsEqual(left: AppConfig, right: AppConfig) {
  if (!left || !right) return false;
  return (Object.keys(DEFAULT_CONFIG) as (keyof AppConfig)[]).every(
    (key) => left[key] === right[key],
  );
}

export function validateConfig(config: AppConfig): string[] {
  const errors: string[] = [];
  if (!/^[A-Z0-9]{5,20}$/.test(config.symbol))
    errors.push("Symbol must be 5–20 uppercase letters and numbers.");
  if (!Number.isInteger(config.gate_interval_ms))
    errors.push("Gate interval must be a whole number of milliseconds.");
  if (!Number.isInteger(config.flow_window_trades))
    errors.push("Flow window must be a whole number of trades.");
  for (const [key, range] of Object.entries(CONFIG_RANGES)) {
    const value = config[key as keyof typeof CONFIG_RANGES] as number;
    if (!Number.isFinite(value) || value < range.min || value > range.max)
      errors.push(
        `${key.replaceAll("_", " ")} must be between ${range.min} and ${range.max}.`,
      );
  }
  return errors;
}

export function presetMatches(config: AppConfig, preset: Preset) {
  return Object.entries(preset.patch).every(
    ([key, value]) => config[key as keyof AppConfig] === value,
  );
}

export function presetById(id: PresetId) {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0];
}
