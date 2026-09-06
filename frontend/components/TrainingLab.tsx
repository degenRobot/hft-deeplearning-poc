"use client";
import { useRef, useState } from "react";
import { LiveLearningPanel, useLiveLearning } from "./LiveLearning";
import { TrainingProgress } from "./TrainingProgress";
import { TrainingDataset } from "./TrainingDataset";
import { PageHeader } from "./PageHeader";
import { SgdPlayground } from "./SgdPlayground";
import { TrainingControls } from "./TrainingControls";
import { useLiveTraining } from "../hooks/useLiveTraining";
import { trainingCanStop } from "../lib/liveTraining";
import {
  InputWindow,
  TrainingNetwork,
  Outputs,
  LossChart,
  Dataset,
  Evaluation,
  Empty,
  experts,
  decimal,
  time,
} from "./TrainingCharts";
const runLabels = {
  running: "Training in progress",
  completed: "Run complete",
  stopped: "Run stopped",
  failed: "Run failed",
};

export function TrainingLab() {
  const learning = useLiveLearning();
  const [view, setView] = useState<"manual" | "live">("manual");
  const { data, error, loading, pending, start, stop } = useLiveTraining(view);
  const settingsRef = useRef<HTMLDetailsElement>(null);
  const openSettings = () => {
    const settings = settingsRef.current;
    if (!settings) return;
    settings.open = true;
    settings.scrollIntoView({ block: "start" });
    settings.querySelector("summary")?.focus({ preventScroll: true });
  };
  const continuous = view === "live";
  const step = data?.latest ?? null;
  const running = data?.status === "running";
  const historical = data?.dataset?.source_mode === "historical_candles_1s";
  return (
    <main className="shell training-lab signal-lab">
      <PageHeader title="Training Lab" current="training" />
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
                  : data?.status === "idle"
                    ? continuous
                      ? "Live RL paused"
                      : "Ready to train"
                    : data
                      ? runLabels[data.status]
                      : "Connecting…"}
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
        <div className="training-next-action">
          <p>
            {loading
              ? "Connecting to the local backend…"
              : error
                ? "Check that the local backend is running. This page reconnects automatically."
                : continuous
                  ? learning.data?.enabled
                    ? "Following the terminal model. Live controls are in Training Settings."
                    : "Enable Live RL in Training Settings to watch the terminal model learn."
                  : running
                    ? "Charts update as the run progresses. You can stop this run at any time."
                    : !step
                      ? "Start with the bundled dataset: open Training Settings, then choose Train locally."
                      : "Review the charts and holdout comparison, or configure another run."}
          </p>
          <div>
            <button
              type="button"
              className="button ghost"
              onClick={openSettings}
            >
              Open Training Settings
            </button>
            <a className="button ghost" href="#sgd-playground">
              Try SGD
            </a>
            {!continuous && trainingCanStop(data) && (
              <button
                type="button"
                className="button ghost"
                disabled={pending}
                onClick={() => void stop()}
              >
                {pending ? "Stopping…" : "Stop run"}
              </button>
            )}
          </div>
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
      <SgdPlayground />
      <details className="training-settings" ref={settingsRef}>
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
