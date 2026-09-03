import type { ResetResponse } from "./types";

export function parseResetResponse(input: unknown): ResetResponse | null {
  if (!input || typeof input !== "object") return null;
  const raw = input as Record<string, unknown>;
  const runId = typeof raw.run_id === "string" ? raw.run_id : "";
  const generation =
    typeof raw.feed_generation === "number" &&
    Number.isFinite(raw.feed_generation)
      ? raw.feed_generation
      : -1;
  return runId && generation >= 0
    ? { run_id: runId, feed_generation: generation }
    : null;
}
