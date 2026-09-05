import type { LiveTraining, TrainingSettings } from "../lib/liveTraining";
import { useState } from "react";
import "./trainingExtras.css";

export function TrainingProgress({ data }: { data: LiveTraining | null }) {
  if (!data || data.status === "idle") return null;
  const p = data.progress;
  const stage =
    data.status !== "running" ? data.status : (p?.stage ?? "preparing");
  const labels: Record<string, string> = {
    preparing:
      data.backend === "modal"
        ? "Starting Modal · building image / preparing data"
        : "Preparing dataset",
    supervised: "Supervised warmup",
    rl: "Online RL on replay",
    finalizing: "Evaluating holdout and saving checkpoints",
    completed: "Training complete",
    failed: "Run failed",
    stopped: "Run stopped",
  };
  const ended = data.status !== "running";
  const percent = p?.percent ?? (data.status === "completed" ? 100 : null);
  return (
    <section className="training-progress" aria-label="Training run progress">
      <div className="training-progress-heading">
        <strong>{labels[stage] ?? "Training"}</strong>
        <b>{percent === null ? (ended ? "—" : "Preparing…") : `${percent}%`}</b>
      </div>
      {(!ended || percent !== null) && (
        <progress
          aria-label="Training completion"
          max={100}
          value={percent ?? undefined}
        />
      )}
      <p>
        {p?.total_steps
          ? `${p.completed_steps} / ${p.total_steps} optimizer updates`
          : ended
            ? "Step totals were not recorded for this run."
            : "Progress becomes measurable when the dataset is ready."}
        {p?.elapsed_seconds != null &&
          ` · ${Math.round(p.elapsed_seconds)}s elapsed`}
      </p>
      {data.status === "running" && (
        <p className="flow-footnote">
          Each real optimizer update advances the bar. Completion also waits for
          evaluation and downloaded checkpoints.
        </p>
      )}
      {data.backend === "modal" && p?.compute_estimate_usd != null && (
        <p className="flow-footnote">
          Estimated compute since remote start:{" "}
          <b>${p.compute_estimate_usd.toFixed(5)}</b> of credits. Based on
          requested CPU and memory × observed elapsed time; excludes image build
          and startup. This is not billed usage.
        </p>
      )}
    </section>
  );
}

export function TrainingCost({
  pricing,
}: {
  pricing: TrainingSettings["pricing"];
}) {
  const [minutes, setMinutes] = useState(1);
  if (!pricing) return <p className="flow-footnote">Cost rates unavailable.</p>;
  const rate =
    pricing.cpu_cores * pricing.cpu_core_second_usd +
    pricing.memory_gib * pricing.memory_gib_second_usd;
  const valid = Number.isFinite(minutes) && minutes >= 0.25 && minutes <= 10;
  return (
    <div className="training-cost">
      <div>
        <strong>Estimate your compute credits</strong>
        <p>
          For {pricing.cpu_cores} physical CPU cores + {pricing.memory_gib} GiB
          memory.
        </p>
      </div>
      <label>
        Assumed compute minutes
        <input
          type="number"
          min="0.25"
          max="10"
          step="0.25"
          value={Number.isFinite(minutes) ? minutes : ""}
          onChange={(e) =>
            setMinutes(e.target.value === "" ? NaN : Number(e.target.value))
          }
        />
      </label>
      <b>
        {valid
          ? `≈ $${(rate * minutes * 60).toFixed(5)} of credits`
          : "Enter 0.25–10 minutes"}
      </b>
      <p className="flow-footnote">
        ${(rate * 60).toFixed(5)} per compute minute; about $
        {(rate * pricing.timeout_seconds).toFixed(5)} for the full 10-minute
        function limit. Choose an assumed duration above; model size and data
        affect actual duration. Image build, startup and any other charges are
        additional. Account balance and billed usage are unavailable.{" "}
        <a href={pricing.source_url} target="_blank" rel="noreferrer">
          Rates checked {pricing.checked_on} ↗
        </a>
      </p>
    </div>
  );
}
