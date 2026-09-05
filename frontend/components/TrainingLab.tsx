"use client";
import Link from "next/link";
import { useLiveTraining } from "../hooks/useLiveTraining";
import type { LiveTraining, TrainingStep } from "../lib/liveTraining";
import { FEATURE_ROWS, FEATURE_HELP } from "../lib/visual";
import { activationOpacity } from "./ActivationNetwork";
const experts = ["Microprice", "Trade flow", "Reversion"];
const colors = ["#41d6b2", "#62c2ed", "#a99bff"];
const n = (value: number) => value.toLocaleString("en-US");
const decimal = (value: number) =>
  Math.abs(value) < 0.001 && value !== 0
    ? value.toExponential(2)
    : value.toFixed(4);
const time = (value: number) => new Date(value).toISOString().slice(11, 23);
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="training-empty">{children}</div>;
}
function InputWindow({ step }: { step: TrainingStep | null }) {
  return (
    <article className="flow-card">
      <span className="flow-kicker">01 / RECORDED MARKET INPUT</span>
      <h2>30 frames × 10 features</h2>
      <p className="input-explanation">
        Raw features for this update. Each column is an observed frame. The
        model standardizes inputs using only the supervised data.
      </p>
      {step ? (
        <>
          <div
            className="heatmap"
            role="img"
            aria-label="Actual 30 frame by 10 feature training input heatmap"
          >
            {FEATURE_ROWS.map(([label, scale], row) => (
              <div className="heat-row" key={label}>
                <span title={FEATURE_HELP[row]}>{label}</span>
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
            ))}
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
function TrainingNetwork({ step }: { step: TrainingStep | null }) {
  return (
    <article className="flow-card">
      <span className="flow-kicker">02 / FORWARD PASS</span>
      <h2>Inside the neural gate</h2>
      <p className="input-explanation">
        300 inputs → 64 ReLU → 32 ReLU → 3 expert probabilities.
      </p>
      {step ? (
        <>
          <div className="training-neurons">
            {(["hidden_1", "hidden_2"] as const).map((layer, index) => (
              <div key={layer}>
                <span>
                  {index === 0 ? "64" : "32"} neurons · hidden {index + 1}
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
            Actual activations before this update. Brightness is scaled within
            each layer. Hover to inspect a neuron.
          </p>
        </>
      ) : (
        <Empty>Waiting for actual forward-pass activations.</Empty>
      )}
    </article>
  );
}
function Outputs({ step }: { step: TrainingStep | null }) {
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
            Same input, before and after the weight update. These are
            expert-selection probabilities, not a forecast probability.
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
function LossChart({
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
function Dataset({ dataset }: { dataset: LiveTraining["dataset"] }) {
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
            <span>{n(dataset.event_count)} public events</span>
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
function Evaluation({
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
      {evaluation ? (
        <table className="training-table">
          <caption>Mean delayed proxy reward · higher is better</caption>
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
export function TrainingLab() {
  const { data, error, loading, pending, start, stop } = useLiveTraining();
  const step = data?.latest ?? null;
  const running = data?.status === "running";
  return (
    <main className="shell training-lab signal-lab">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">MG</span>
          <div>
            <h1>Training Lab</h1>
            <span className="brand-caption">
              From public events to model weights
            </span>
          </div>
        </div>
        <Link className="terminal-nav" href="/">
          Live terminal ↗
        </Link>
      </header>
      <div className="training-intro">
        <div>
          <span className="flow-kicker">EXPERIMENTAL / LEARNING IN VIEW</span>
          <h2>A small model, trained in front of you.</h2>
          <p>
            Supervised warmup learns which proxy expert fits the next move. Then
            REINFORCE updates the gate one example at a time, using delayed
            rewards from recorded public market events.
          </p>
        </div>
        <div className="training-controls">
          <button
            className="button primary"
            disabled={loading || pending || !data || running}
            onClick={() => void start()}
          >
            {pending ? "Updating run…" : "Start local training"}
          </button>
          <button
            className="button ghost"
            disabled={pending || !running || data?.backend !== "local"}
            onClick={() => void stop()}
          >
            Stop run
          </button>
          <span>Bounded local run · no cloud launch</span>
        </div>
      </div>
      <div
        className={`training-statusbar ${error ? "training-error" : ""}`}
        role="status"
      >
        <span className={`live-indicator ${running ? "active" : ""}`} />
        <strong>
          {error
            ? "Telemetry unavailable"
            : loading
              ? "Connecting to training backend"
              : pending
                ? "Waiting for run acknowledgement"
                : data?.status.toUpperCase()}
        </strong>
        <span>
          {error ||
            (data?.run_id
              ? `${data.backend} · ${data.run_id} · ${step ? `step ${step.step} / ${step.phase === "rl" ? "online RL replay" : "supervised"}` : "Preparing dataset"}`
              : "No training run yet")}
        </span>
        {data?.updated_at && (
          <time dateTime={data.updated_at}>
            {new Date(data.updated_at).toISOString().slice(11, 19)} UTC
          </time>
        )}
      </div>
      {data?.error && (
        <p className="training-run-error" role="alert">
          Run failed: {data.error}
        </p>
      )}
      <div className="training-explainer">
        <span>01 · recorded features</span>
        <b>→</b>
        <span>02 · forward pass</span>
        <b>→</b>
        <span>03 · loss + gradient</span>
        <b>→</b>
        <span>04 · updated weights</span>
      </div>
      <div className="training-forward">
        <InputWindow step={step} />
        <TrainingNetwork step={step} />
        <Outputs step={step} />
      </div>
      <div className="training-secondary">
        <article className="flow-card">
          <span className="flow-kicker">
            OPTIMIZATION / ACTUAL STEP HISTORY
          </span>
          <h2>Two stages, two learning signals</h2>
          <LossChart history={data?.history ?? []} phase="supervised" />
          <LossChart history={data?.history ?? []} phase="rl" />
          <p className="flow-footnote">
            Latest {data?.history.length ?? 0} of at most 400 telemetry steps.
            Each chart uses its own loss scale; policy-gradient loss can be
            negative.
          </p>
        </article>
        <article className="flow-card">
          <span className="flow-kicker">
            BACKPROPAGATION / STEP {step?.step ?? "—"}
          </span>
          <h2>What changed in the weights?</h2>
          {step ? (
            <table className="training-table training-layer-table">
              <caption>L2 norms from this actual optimizer step</caption>
              <thead>
                <tr>
                  <th scope="col">Layer</th>
                  <th scope="col">Gradient</th>
                  <th scope="col">Weight Δ</th>
                  <th scope="col">Weights</th>
                </tr>
              </thead>
              <tbody>
                {step.layers.map((layer) => (
                  <tr key={layer.name}>
                    <th scope="row">{layer.name}</th>
                    <td>
                      {decimal(layer.gradient_norm)}
                      <i
                        className="training-norm-bar"
                        style={{
                          width: `${(100 * layer.gradient_norm) / Math.max(...step.layers.map((x) => x.gradient_norm), 1e-12)}%`,
                        }}
                      />
                    </td>
                    <td className="teal">{decimal(layer.weight_delta_norm)}</td>
                    <td>{decimal(layer.weight_norm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>
              No optimizer update yet. Gradient and weight-change norms will
              appear here.
            </Empty>
          )}
          <p className="flow-footnote">
            The gradient describes loss sensitivity. The optimizer turns it into
            a weight change; the next forward pass produces new expert
            probabilities.
          </p>
          <div className="training-reward">
            <span className="flow-kicker">DELAYED REWARD / RECORDED TIME</span>
            <h3>
              {step?.phase === "rl" && step.action !== null
                ? `${experts[step.action]} selected`
                : "Choose an expert, then observe the outcome"}
            </h3>
            <div>
              <span>{step ? time(step.input_end_ts_ms) : "input time"}</span>
              <b>
                → +
                {step
                  ? ((step.target_ts_ms - step.input_end_ts_ms) / 1000).toFixed(
                      1,
                    )
                  : "5"}
                s →
              </b>
              <strong>
                {step?.reward != null ? decimal(step.reward) : "reward pending"}
              </strong>
            </div>
            <p className="flow-footnote">
              Online contextual-bandit updates on replay. Recorded outcomes
              arrive after the input window; this is not reinforcement from a
              live exchange.
            </p>
          </div>
        </article>
      </div>
      <div className="training-bottom">
        <Dataset dataset={data?.dataset ?? null} />
        <Evaluation evaluation={data?.evaluation ?? null} />
      </div>
      <footer className="footer">
        Educational experiment · actual optimizer telemetry · no model promotion
        or trading execution
      </footer>
    </main>
  );
}
