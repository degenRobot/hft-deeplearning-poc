import type { LiveTraining, TrainingStep } from "../lib/liveTraining";
import { FEATURE_ROWS, FEATURE_HELP } from "../lib/visual";
import { activationOpacity } from "./ActivationNetwork";
export const experts = ["Microprice", "Trade flow", "Reversion"];
const colors = ["#41d6b2", "#62c2ed", "#a99bff"];
const n = (value: number) => value.toLocaleString("en-US");
export const decimal = (value: number) =>
  Math.abs(value) < 0.001 && value !== 0
    ? value.toExponential(2)
    : value.toFixed(4);
export const time = (value: number) =>
  new Date(value).toISOString().slice(11, 23);
export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="training-empty">{children}</div>;
}
export function InputWindow({
  step,
  dataset,
  continuous = false,
}: {
  step: TrainingStep | null;
  dataset: LiveTraining["dataset"];
  continuous?: boolean;
}) {
  return (
    <article className="flow-card">
      <span className="flow-kicker">01 / MARKET INPUT</span>
      <h2>30 frames × 10 features</h2>
      <p className="input-explanation">
        {continuous
          ? "Live features · one column per second."
          : "Recorded features · standardized from supervised data."}
      </p>
      {step ? (
        <>
          <div
            className="heatmap"
            role="img"
            aria-label="Actual 30 frame by 10 feature training input heatmap"
          >
            {FEATURE_ROWS.map(([fallbackLabel, scale], row) => {
              const label = dataset?.feature_names?.[row] ?? fallbackLabel;
              return (
                <div className="heat-row" key={label}>
                  <span
                    title={dataset?.feature_names ? label : FEATURE_HELP[row]}
                  >
                    {label.replaceAll("_", " ")}
                  </span>
                  <div>
                    {step.features.map((frame, column) => (
                      <i
                        className="feature-cell"
                        key={column}
                        style={{
                          background: frame[row] >= 0 ? colors[0] : colors[2],
                          opacity:
                            0.12 +
                            0.88 * Math.min(1, Math.abs(frame[row]) / scale),
                        }}
                        title={`${label} · frame ${column + 1}: ${frame[row].toPrecision(6)}`}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="heat-axis">
            <span>older → newer</span>
            <span>frame 30</span>
          </div>
          <p className="flow-footnote">
            Input ends {time(step.input_end_ts_ms)} UTC. Scales differ by
            feature; hover to read raw values.
          </p>
        </>
      ) : (
        <Empty>Input windows appear after the first optimizer step.</Empty>
      )}
    </article>
  );
}
export function TrainingNetwork({
  step,
  dataset,
}: {
  step: TrainingStep | null;
  dataset: LiveTraining["dataset"];
}) {
  const sizes = dataset?.hidden_sizes ?? [64, 32];
  return (
    <article className="flow-card">
      <span className="flow-kicker">02 / FORWARD PASS</span>
      <h2>Inside the neural gate</h2>
      <p className="input-explanation">
        {dataset
          ? `300 inputs → ${sizes[0]} ReLU → ${sizes[1]} ReLU → 3 expert probabilities.`
          : "300 inputs → 64 ReLU → 32 ReLU → 3 expert probabilities."}
        {dataset?.parameter_count
          ? ` ${n(dataset.parameter_count)} parameters in this run.`
          : ""}
      </p>
      {step ? (
        <>
          <div className="training-neurons">
            {(["hidden_1", "hidden_2"] as const).map((layer, index) => (
              <div key={layer}>
                <span>
                  {step.activations[layer].length < sizes[index]
                    ? `First ${step.activations[layer].length} of ${sizes[index]}`
                    : `${sizes[index]} neurons`}{" "}
                  · hidden {index + 1}
                </span>
                <div className={`neuron-grid layer-${index}`}>
                  {step.activations[layer].map((value, neuron) => (
                    <i
                      key={neuron}
                      style={{
                        background: index === 0 ? colors[0] : colors[2],
                        opacity: activationOpacity(
                          value,
                          Math.max(...step.activations[layer]),
                        ),
                      }}
                      title={`Hidden ${index + 1}, neuron ${neuron + 1}: ${value.toPrecision(6)}`}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="flow-footnote">
            Actual activations · brighter = stronger within this layer.
          </p>
        </>
      ) : (
        <Empty>Waiting for actual forward-pass activations.</Empty>
      )}
    </article>
  );
}
export function Outputs({ step }: { step: TrainingStep | null }) {
  return (
    <article className="flow-card">
      <span className="flow-kicker">03 / ONE OPTIMIZER STEP</span>
      <h2>Watch the output change</h2>
      {step ? (
        <>
          <div className="training-output-head">
            <span>Expert</span>
            <span>Before</span>
            <span>After</span>
          </div>
          {experts.map((name, i) => (
            <div className="training-output" key={name}>
              <div>
                <span>{name}</span>
                <span>{(step.outputs_before[i] * 100).toFixed(2)}%</span>
                <strong style={{ color: colors[i] }}>
                  {(step.outputs_after[i] * 100).toFixed(2)}%
                  <small className="training-output-delta">
                    {step.outputs_after[i] >= step.outputs_before[i] ? "+" : ""}
                    {decimal(
                      (step.outputs_after[i] - step.outputs_before[i]) * 100,
                    )}{" "}
                    pp
                  </small>
                </strong>
              </div>
              <div className="training-prob-track">
                <i
                  style={{
                    width: `${step.outputs_before[i] * 100}%`,
                    background: colors[i],
                    opacity: 0.3,
                  }}
                />
                <i
                  style={{
                    width: `${step.outputs_after[i] * 100}%`,
                    background: colors[i],
                  }}
                />
              </div>
            </div>
          ))}
          <p className="flow-footnote">
            Same input · expert selection before → after learning.
          </p>
        </>
      ) : (
        <Empty>
          Before and after probabilities require a completed optimizer step.
        </Empty>
      )}
    </article>
  );
}
export function LossChart({
  history,
  phase,
}: {
  history: TrainingStep[];
  phase: TrainingStep["phase"];
}) {
  const values = history.filter((x) => x.phase === phase);
  const min = Math.min(...values.map((x) => x.loss));
  const max = Math.max(...values.map((x) => x.loss));
  const points = values
    .map(
      (v, i) =>
        `${8 + (i / Math.max(values.length - 1, 1)) * 444},${90 - ((v.loss - min) / (max - min || 1)) * 70}`,
    )
    .join(" ");
  return (
    <div className="training-loss">
      <div>
        <h3>
          {phase === "supervised"
            ? "Supervised · cross entropy"
            : "RL · policy gradient loss"}
        </h3>
        <strong>
          {values.length ? decimal(values[values.length - 1].loss) : "—"}
        </strong>
      </div>
      {values.length ? (
        <>
          <svg
            viewBox="0 0 460 106"
            role="img"
            aria-label={`${phase} actual loss over ${values.length} optimizer steps`}
          >
            <path d="M8 90H452" stroke="var(--line)" />
            <polyline
              fill="none"
              stroke={phase === "supervised" ? colors[0] : colors[2]}
              strokeWidth="2"
              points={points}
            />
            {values.length === 1 && (
              <circle cx="8" cy="90" r="3" fill={colors[0]} />
            )}
          </svg>
          <div className="training-loss-axis">
            <span>
              step {values[0].step} → {values[values.length - 1].step}
            </span>
            <span>
              range {decimal(min)} … {decimal(max)}
            </span>
          </div>
        </>
      ) : (
        <Empty>
          No {phase === "rl" ? "reinforcement" : "supervised"} updates received.
        </Empty>
      )}
    </div>
  );
}
export function Dataset({ dataset }: { dataset: LiveTraining["dataset"] }) {
  const segments = dataset
    ? ([
        ["Supervised warmup", dataset.supervised_examples],
        ["Online RL replay", dataset.rl_examples],
        ["Frozen holdout", dataset.holdout_examples],
      ] as const)
    : null;
  const total = segments?.reduce((sum, [, count]) => sum + count, 0) || 1;
  return (
    <article className="flow-card training-dataset">
      <span className="flow-kicker">PUBLIC DATA / CHRONOLOGICAL SPLIT</span>
      <h2>Learn from the past. Test on later events.</h2>
      {dataset && segments ? (
        <>
          <div className="training-dataset-numbers">
            <strong>{dataset.symbol}</strong>
            <span>
              {n(dataset.event_count)}{" "}
              {dataset.source_mode === "historical_candles_1s"
                ? "candles"
                : "public events"}
            </span>
            <span>{n(dataset.frame_count)} frames</span>
          </div>
          <div
            className="training-split"
            aria-label="Chronological supervised, reinforcement, and holdout example counts"
          >
            {segments.map(([label, count], i) => (
              <div
                key={label}
                style={{ flexGrow: count / total, borderColor: colors[i] }}
              >
                <span>{label}</span>
                <strong>{n(count)}</strong>
              </div>
            ))}
          </div>
          <details className="feature-values">
            <summary>Dataset fingerprint</summary>
            <code className="training-hash">SHA-256 {dataset.sha256}</code>
          </details>
        </>
      ) : (
        <Empty>
          The backend will report the selected public dataset and accepted
          example counts.
        </Empty>
      )}
      <p className="flow-footnote">
        Temporal boundaries include an embargo. Holdout examples never update
        the weights. This experiment stays separate from the model in the live
        terminal.
      </p>
    </article>
  );
}
export function Evaluation({
  evaluation,
}: {
  evaluation: LiveTraining["evaluation"];
}) {
  const rows: [string, keyof NonNullable<LiveTraining["evaluation"]>][] = [
    ["Supervised checkpoint", "supervised"],
    ["After RL adaptation", "adapted"],
    ["Equal expert weights", "uniform"],
    ["Microprice only", "microprice"],
    ["Trade flow only", "trade_flow"],
    ["Reversion only", "reversion"],
  ];
  return (
    <article className="flow-card">
      <span className="flow-kicker">FROZEN HOLDOUT / SAME LATER EXAMPLES</span>
      <h2>Does learning beat simple baselines?</h2>
      {evaluation && new Set(Object.values(evaluation)).size === 1 && (
        <p className="input-explanation">
          Every policy has the same mean reward in this holdout. This sample
          provides no evidence that either trained model beats the baselines.
        </p>
      )}
      {evaluation ? (
        <table className="training-table">
          <caption>Mean delayed proxy reward in bps · higher is better</caption>
          <thead>
            <tr>
              <th scope="col">Policy</th>
              <th scope="col">Reward</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, key]) => (
              <tr key={key}>
                <th scope="row">{name}</th>
                <td>{decimal(evaluation[key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty>
          Evaluation appears when the run finishes. No baseline result is
          assumed.
        </Empty>
      )}
      <p className="flow-footnote">
        Proxy rewards measure this learning task. They are not trading P&amp;L,
        and a lower training loss does not prove better holdout performance.
      </p>
    </article>
  );
}
