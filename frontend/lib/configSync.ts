import { configsEqual } from "./config";
import type { AppConfig } from "./types";

export interface ConfigRequestToken {
  requestGeneration: number;
  mutationGeneration: number;
}

export function isCurrentConfigResponse(
  token: ConfigRequestToken,
  current: ConfigRequestToken,
) {
  return (
    token.requestGeneration === current.requestGeneration &&
    token.mutationGeneration === current.mutationGeneration
  );
}

export function mergeConfigResponse(
  draftConfig: AppConfig,
  appliedConfig: AppConfig,
  canonical: AppConfig,
) {
  return {
    appliedConfig: canonical,
    draftConfig: configsEqual(draftConfig, appliedConfig)
      ? canonical
      : draftConfig,
  };
}
