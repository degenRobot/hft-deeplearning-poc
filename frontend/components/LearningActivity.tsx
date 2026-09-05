import type { LearningStatus } from "./LiveLearning";
import "./learningActivity.css";

const stageDetails: Record<string, [string, string]> = {
  warming_up: [
    "Collecting inputs",
    "Waiting for 30 consecutive fresh one-second frames before sampling an expert.",
  ],
  observing_reward: [
    "Observing the outcome",
    "An expert has been sampled. Watching the next five seconds of price movement before updating the output head.",
  ],
  waiting_for_interval: [
    "Waiting for the next example",
    "The learner is enabled. The next example starts on the configured sampling interval.",
  ],
  waiting_for_feed: [
    "Waiting for fresh market data",
    "Learning is blocked until the feed is fresh and a continuous input window is available.",
  ],
};

export function LearningActivity({
  data,
  error,
  pending,
  compact = false,
  hideStatus = false,
}: {
  data: LearningStatus | null;
  error: string;
  pending: boolean;
  compact?: boolean;
  hideStatus?: boolean;
}) {
  const failed = Boolean(error || data?.error || data?.stage === "failed");
  const knownStage = data ? stageDetails[data.stage] : undefined;
  const running = Boolean(data?.enabled && knownStage && !failed && !pending);
  const blocked = running && data?.stage === "waiting_for_feed";
  const stage = running ? data?.stage : undefined;
  const headline = failed
    ? "Live RL status needs attention"
    : pending
      ? "Applying Live RL settings"
      : !data
        ? "Connecting to Live RL"
        : blocked
          ? "Live RL waiting for data"
          : running
            ? "Live RL running"
            : data.enabled
              ? "Live RL state unavailable"
              : data.updates > 0
                ? "Live RL paused"
                : "Live RL off";
  const detail =
    error ||
    data?.error ||
    (data?.stage === "failed"
      ? "Learning stopped; the last valid weights are retained."
      : pending
        ? "Waiting for the backend to acknowledge your change."
        : !data
          ? "Waiting for current learning telemetry."
          : running
            ? knownStage![1]
            : data.enabled
              ? "The backend reported an unrecognized learning stage."
              : data.updates > 0
                ? "No new learning updates. Learned weights remain in memory and available to the terminal."
                : "Enable Learn while running to start collecting examples.");
  const tone =
    failed || blocked || (data?.enabled && !knownStage)
      ? "waiting"
      : running
        ? "running"
        : "idle";
  const countdown =
    running && !blocked && stage !== "warming_up"
      ? data?.next_update_in_seconds
      : null;
  const nodes = [
    {
      title: "Collect inputs",
      description: "30 fresh frames × 10 features",
      active: stage === "warming_up" || stage === "waiting_for_feed",
      state:
        stage === "warming_up"
          ? "Collecting"
          : blocked
            ? "Feed blocked"
            : stage === "observing_reward"
              ? "Window sampled"
              : "Input window",
    },
    {
      title: "Sample an expert",
      description: "Neural gate chooses a teaching action",
      active: false,
      state:
        stage === "observing_reward" ? "Expert sampled" : "Once per example",
    },
    {
      title: "Observe the reward",
      description: "Expert signal × next 5s price move",
      active: stage === "observing_reward",
      state:
        stage === "observing_reward" ? "Observing now" : "Five-second outcome",
    },
    {
      title: "Update the gate",
      description: "99 output parameters · hidden layers frozen",
      active: false,
      state:
        data && !failed
          ? `${data.updates} updates completed`
          : "Awaiting telemetry",
    },
  ];
  if (compact)
    return (
      <span
        className="rl-compact-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-tone={tone}
      >
        <span className="rl-status-dot" aria-hidden="true" />
        <span>{running ? knownStage![0] : headline}</span>
        {data && !failed && (
          <span className="rl-compact-updates">
            {data.updates} updates · {data.symbol}
          </span>
        )}
        {failed && <span className="rl-compact-error">{detail}</span>}
      </span>
    );
  return (
    <div className="rl-activity" data-tone={tone}>
      {!hideStatus && (
        <div
          className="rl-statusbar"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <span className="rl-status-dot" aria-hidden="true" />
          <div>
            <strong>{headline}</strong>
            <span>{running ? knownStage![0] : "Learning process"}</span>
          </div>
          {data && !failed && (
            <span className="rl-source">
              {data.source === "binance"
                ? "Binance public feed"
                : data.source === "replay"
                  ? "Replay fixture"
                  : data.source}{" "}
              · {data.symbol} · local CPU
            </span>
          )}
        </div>
      )}
      <p className="rl-detail">{detail}</p>
      <dl className="rl-metrics" aria-label="Live RL activity metrics">
        <div>
          <dt>Completed updates</dt>
          <dd>{data && !failed ? data.updates : "—"}</dd>
        </div>
        <div>
          <dt>Next possible update</dt>
          <dd>
            {countdown == null
              ? "—"
              : countdown === 0
                ? "Awaiting fresh outcome"
                : `~${Math.ceil(countdown)}s`}
          </dd>
        </div>
        <div>
          <dt>Last step duration</dt>
          <dd>
            {data?.step_duration_ms != null && !failed
              ? `${data.step_duration_ms.toFixed(2)} ms`
              : "—"}
          </dd>
        </div>
        <div>
          <dt>Last weight change</dt>
          <dd>
            {data && data.updates > 0 && !failed
              ? data.weight_delta.toExponential(2)
              : "—"}
          </dd>
        </div>
      </dl>
      <figure className="rl-diagram" aria-label="Live learning and quote flow">
        <figcaption>
          Learning loop{" "}
          <span>
            {data && !failed
              ? `Sample every ${data.interval_seconds}s when data is fresh`
              : "Waiting for current settings"}
          </span>
        </figcaption>
        <ol className="rl-steps">
          {nodes.map((node, index) => (
            <li
              key={node.title}
              data-active={node.active}
              aria-current={node.active ? "step" : undefined}
            >
              <span className="rl-step-number">0{index + 1}</span>
              <strong>{node.title}</strong>
              <p>{node.description}</p>
              <span className="rl-step-state">{node.state}</span>
            </li>
          ))}
        </ol>
        <div className="rl-return">
          ↻ Repeat with the next fresh example. Highlights follow backend
          telemetry.
        </div>
        <div className="rl-quote-path">
          <strong>How it reaches the terminal</strong>
          <p>
            Updated gate → smoothed expert weights → three fixed experts → risk
            checks → synthetic quotes
          </p>
          <span>
            The sampled action teaches the gate. Quotes use the weighted
            mixture. No exchange orders.
          </span>
        </div>
      </figure>
    </div>
  );
}
