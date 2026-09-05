"use client";
import { useState } from "react";
import { LiveLearningPanel, useLiveLearning } from "./LiveLearning";
import { TrainingProgress } from "./TrainingProgress";
import { TrainingDataset } from "./TrainingDataset";
import Link from "next/link";
import { SiteNav } from "./SiteNav";
import { TrainingControls } from "./TrainingControls";
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
function InputWindow({
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
function TrainingNetwork({
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
export function TrainingLab() {
  const learning = useLiveLearning();
  const [view, setView] = useState<"manual" | "live">("manual");
  const { data, error, loading, pending, start, stop } = useLiveTraining(view);
  const continuous = view === "live";
  const step = data?.latest ?? null;
  const running = data?.status === "running";
  const historical = data?.dataset?.source_mode === "historical_candles_1s";
  return (
    <main className="shell training-lab signal-lab">
      <header className="topbar lab-topbar">
        <Link
          className="brand brand-home"
          href="/"
          aria-label="Market Gate Lab home"
        >
          <span className="brand-mark">MG</span>
          <div>
            <h1>Training Lab</h1>
            <span className="brand-caption">Market Gate Lab</span>
          </div>
        </Link>
        <SiteNav current="training" />
      </header>
      <section className="training-workspace" aria-label="Training mode">
        <nav className="training-view-tabs" aria-label="Training view">
          <button aria-pressed={!continuous} onClick={() => setView("manual")}>
            <strong>Train on history</strong>
            <span>Prepare data, run an experiment, compare results.</span>
          </button>
          <button aria-pressed={continuous} onClick={() => setView("live")}>
            <strong>Live RL {learning.data?.enabled ? "· running" : ""}</strong>
            <span>Watch the terminal model adapt to incoming data.</span>
          </button>
        </nav>
      </section>
      <section
        className="training-results"
        aria-labelledby="training-results-title"
      >
        <div className="training-results-heading">
          <div>
            <span className="flow-kicker">
              {continuous ? "LIVE TELEMETRY" : "EXPERIMENT RESULTS"}
            </span>
            <h2 id="training-results-title">
              {continuous ? "Inside each live update" : "Follow the learning"}
            </h2>
          </div>
          <p>
            {continuous
              ? "Actual updates from the terminal model."
              : "Run progress, model activations and measured outcomes."}
          </p>
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
                ? `${continuous ? "live feed" : data.backend} · ${step ? `step ${step.step} / ${step.phase === "rl" ? (continuous ? "live RL" : "RL replay") : "supervised"}` : continuous ? "Waiting for first live update" : "Preparing dataset"}`
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
          <span>01 · features</span>
          <b>→</b>
          <span>02 · forward pass</span>
          <b>→</b>
          <span>03 · loss + gradient</span>
          <b>→</b>
          <span>04 · updated weights</span>
        </div>
        <div className="training-forward">
          <InputWindow
            step={step}
            dataset={data?.dataset ?? null}
            continuous={continuous}
          />
          <TrainingNetwork step={step} dataset={data?.dataset ?? null} />
          <Outputs step={step} />
        </div>
        <div className="training-secondary">
          <article className="flow-card">
            <span className="flow-kicker">
              OPTIMIZATION / ACTUAL STEP HISTORY
            </span>
            <h2>
              {continuous ? "Learning as data arrives" : "Learning curve"}
            </h2>
            {!continuous && (
              <LossChart history={data?.history ?? []} phase="supervised" />
            )}
            <LossChart history={data?.history ?? []} phase="rl" />
            <p className="flow-footnote">
              {data?.history.length ?? 0} recent updates · each curve has its
              own scale.
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
                      <td className="teal">
                        {decimal(layer.weight_delta_norm)}
                      </td>
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
              {continuous
                ? "Frozen layers: Δ = 0. Output head: one small gradient step."
                : "Gradient → weight change → new expert probabilities."}
            </p>
            <div className="training-reward">
              <span className="flow-kicker">
                DELAYED REWARD / OBSERVED OUTCOME
              </span>
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
                    ? (
                        (step.target_ts_ms - step.input_end_ts_ms) /
                        1000
                      ).toFixed(1)
                    : "5"}
                  s →
                </b>
                <strong>
                  {step?.reward != null
                    ? decimal(step.reward)
                    : "reward pending"}
                </strong>
              </div>
              <p className="flow-footnote">
                {continuous
                  ? "Live observed move × sampled expert signal. Clipped proxy reward, not P&L."
                  : "Delayed proxy reward on recorded data · not live exchange feedback."}
              </p>
            </div>
          </article>
        </div>
        {!continuous && <TrainingProgress data={data} />}
        {historical && (
          <aside
            className="training-mode-banner"
            aria-label="Historical training limitations"
          >
            <strong>Historical candle proxy training</strong>
            <p>
              1s candles → 30s input → five-second close-price move. Four book
              features unavailable; not compatible with the live terminal.
            </p>
            <details>
              <summary>Data assumptions</summary>
              {data?.dataset?.limitations?.map((limitation, index) => (
                <p key={index}>{limitation}</p>
              ))}
            </details>
          </aside>
        )}
        {!continuous && (
          <details className="training-evaluation-details">
            <summary>Dataset split &amp; holdout comparison</summary>
            <div className="training-bottom">
              <Dataset dataset={data?.dataset ?? null} />
              <Evaluation evaluation={data?.evaluation ?? null} />
            </div>
          </details>
        )}
      </section>
      <details className="training-settings">
        <summary>
          <strong>Training Settings</strong>
          <span>
            {continuous
              ? "Live learning controls"
              : "Data preparation & run configuration"}
          </span>
        </summary>
        <div className="training-settings-body">
          {continuous ? (
            <LiveLearningPanel learning={learning} lab />
          ) : (
            <div className="training-setup-grid">
              <TrainingDataset />
              <TrainingControls
                data={data}
                loading={loading}
                pending={pending}
                start={start}
                stop={stop}
              />
            </div>
          )}
        </div>
      </details>
      <details className="flow-card transfer-note">
        <summary>How does transfer learning fit?</summary>
        <div className="learning-cycle">
          <span>Pretrain on history</span>
          <b>→</b>
          <span>Keep useful layers</span>
          <b>→</b>
          <span>Fine-tune on new data</span>
        </div>
        <p>
          A longer download is more examples, not a slower model: our candles
          are still 1 second, the input is 30 frames, and the target is a
          5-second move. Changing those time scales changes what features mean.
        </p>
        <p>
          The live demo reuses the loaded gate, freezes its hidden layers and
          adapts its output head. Historical candle models stay separate:
          candles cannot supply missing order-book features. Connecting those
          models would require a compatible feature schema and a fresh
          evaluation.
        </p>
        <p>
          <a
            href="https://docs.fast.ai/callback.schedule.html#Learner.fine_tune"
            target="_blank"
            rel="noreferrer"
          >
            fast.ai: freeze, then fine-tune ↗
          </a>{" "}
          ·{" "}
          <a
            href="https://forums.fast.ai/t/transfer-learning-in-fast-ai-how-does-the-magic-work/55620"
            target="_blank"
            rel="noreferrer"
          >
            Your reference discussion ↗
          </a>
        </p>
      </details>
      <footer className="footer">
        Educational demo · actual weight updates · synthetic quotes only
      </footer>
    </main>
  );
}
