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

export function shouldSeedDraft(draftWasEdited: boolean) {
  return !draftWasEdited;
}
