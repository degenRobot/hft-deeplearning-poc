"use client";

import { useEffect, useId, useState, type ReactNode } from "react";
import {
  initialSgdState,
  sgdStep,
  SYNTHETIC_DATASET,
  type SgdDataset,
  SGD_STEPS,
  SGD_BATCH_SIZE,
} from "../lib/sgdDemo";
import "./sgd-playground.css";
import { useBrowserMarketData } from "../hooks/useBrowserMarketData";
import {
  capturedReturns,
  liveSgdDataset,
  MIN_CAPTURE_ROWS,
} from "../lib/browserMarketData";

const presets = [
  ["Small", 0.005],
  ["Steady", 0.1],
  ["Too large", 1.2],
] as const;
const lossY = (loss: number) =>
  190 - ((Math.max(-3, Math.min(4, Math.log10(loss))) + 3) / 7) * 166;
const rateLabel = (rate: number) => Number(rate.toPrecision(3)).toString();

export function SgdPlayground() {
  const { capture, status } = useBrowserMarketData();
  const [selected, setSelected] = useState<{
    dataset: SgdDataset;
    revision: number;
  } | null>(null);
  const available = capture ? liveSgdDataset(capture) : null;
  const rows = capture ? capturedReturns(capture).length : 0;
  const choose = (dataset: SgdDataset) =>
    setSelected((previous) => ({
      dataset,
      revision: (previous?.revision ?? 0) + 1,
    }));
  const controls = (
    <div className="sgd-source">
      <div>
        <strong>{status}</strong>
        <span>
          {rows} usable return pairs · {capture?.candles.length ?? 0} completed
          candles
        </span>
      </div>
      <div className="sgd-actions">
        <button
          className="button"
          type="button"
          disabled={!available}
          onClick={() => available && choose(available)}
        >
          {selected ? "Train on latest capture" : "Train on live capture"}
        </button>
        <button
          className="button ghost"
          type="button"
          onClick={() => choose(SYNTHETIC_DATASET)}
        >
          Synthetic example
        </button>
      </div>
    </div>
  );
  return selected ? (
    <SgdExperiment
      key={selected.revision}
      dataset={selected.dataset}
      dataControls={controls}
    />
  ) : (
    <section
      id="sgd-playground"
      className="flow-card sgd-playground"
      aria-label="Browser training on live data"
    >
      <span className="flow-kicker">REAL DATA / BROWSER SGD</span>
      <h2>See how a model learns</h2>
      <p className="flow-footnote">
        Fit next-second returns from live trade candles. Training happens in
        this browser.
      </p>
      {controls}
      <p className="sgd-help">
        {available
          ? "Your capture is ready. Start a run to watch the fit and loss update at every SGD step."
          : `Waiting for ${MIN_CAPTURE_ROWS} consecutive-second return pairs. Open Live Terminal with Binance connected; gaps are skipped.`}
      </p>
    </section>
  );
}

function SgdExperiment({
  dataset,
  dataControls,
}: {
  dataset: SgdDataset;
  dataControls: ReactNode;
}) {
  const id = useId();
  const [state, setState] = useState(() => initialSgdState(dataset));
  const [rate, setRate] = useState(0.1);
  const [playing, setPlaying] = useState(Boolean(dataset.capturedAt));
  const done = state.diverged || state.step >= SGD_STEPS;
  const running = playing && !done;
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      if (!document.hidden)
        setState((previous) => sgdStep(previous, rate, dataset));
    }, 140);
    return () => window.clearInterval(timer);
  }, [running, rate, dataset]);
  const latest = state.history.at(-1)!;
  const live = Boolean(dataset.capturedAt);
  const allPoints = [...dataset.points, ...dataset.validation];
  const xBound = Math.max(1, ...allPoints.map((point) => Math.abs(point.x)));
  const yBound =
    Math.max(live ? 1.5 : 2.5, ...allPoints.map((point) => Math.abs(point.y))) *
    1.05;
  const px = (x: number) => 44 + ((x + xBound) / (2 * xBound)) * 370;
  const py = (y: number) => 190 - ((y + yBound) / (2 * yBound)) * 166;
  const status = state.diverged
    ? "Diverging · reset and lower the rate"
    : done
      ? "120 steps complete"
      : running
        ? "Running"
        : state.step
          ? "Paused"
          : "Ready · press Run";
  return (
    <section
      id="sgd-playground"
      className="flow-card sgd-playground"
      aria-labelledby={`${id}-title`}
    >
      <div className="sgd-heading">
        <div>
          <span className="flow-kicker">
            {live ? "REAL DATA / BROWSER SGD" : "SYNTHETIC SGD EXAMPLE"}
          </span>
          <h2 id={`${id}-title`}>See how a model learns</h2>
          <p>
            {dataset.label} · 2-parameter model · separate from the trading gate
          </p>
          {live && (
            <p>
              Frozen at{" "}
              {new Date(dataset.capturedAt!).toISOString().slice(11, 19)} UTC ·{" "}
              {dataset.points.length} training / {dataset.validation.length}{" "}
              held out
            </p>
          )}
        </div>
        <span
          className={state.diverged ? "sgd-diverged" : "sgd-status"}
          role="status"
        >
          {status}
        </span>
      </div>
      {dataControls}
      <div className="sgd-controls">
        <label className="sgd-rate" htmlFor={`${id}-rate`}>
          <span>
            Learning rate <output>{rateLabel(rate)}</output>
          </span>
          <input
            id={`${id}-rate`}
            type="range"
            min="0"
            max="100"
            step="1"
            value={(Math.log10(rate / 0.001) / Math.log10(1500)) * 100}
            aria-valuetext={rateLabel(rate)}
            onChange={(event) =>
              setRate(0.001 * 1500 ** (Number(event.target.value) / 100))
            }
          />
        </label>
        <div className="sgd-presets" aria-label="Example learning rates">
          {presets.map(([label, value]) => (
            <button
              type="button"
              key={label}
              aria-pressed={rate === value}
              onClick={() => setRate(value)}
            >
              {live && label === "Too large" ? "High" : label}
            </button>
          ))}
        </div>
        <div className="sgd-actions">
          <button
            className="button"
            type="button"
            disabled={done}
            onClick={() => setPlaying(!playing)}
          >
            {running ? "Pause" : "Run"}
          </button>
          <button
            className="button ghost"
            type="button"
            disabled={running || done}
            onClick={() =>
              setState((previous) => sgdStep(previous, rate, dataset))
            }
          >
            Step
          </button>
          <button
            className="button ghost"
            type="button"
            onClick={() => {
              setPlaying(false);
              setState(initialSgdState(dataset));
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <div className="sgd-charts">
        <figure>
          <figcaption>
            {live ? "Fit next-second returns" : "Fit the points"}{" "}
            <span>
              step {state.step} / {SGD_STEPS}
            </span>
          </figcaption>
          <svg
            viewBox="0 0 440 220"
            role="img"
            aria-label={`Linear model fit after ${state.step} SGD steps. Weight ${state.weight.toFixed(3)}, bias ${state.bias.toFixed(3)}.`}
          >
            <defs>
              <clipPath id={`${id}-clip`}>
                <rect x="44" y="24" width="370" height="166" />
              </clipPath>
            </defs>
            {[-yBound / 2, 0, yBound / 2].map((y) => (
              <g key={y}>
                <line
                  x1="44"
                  x2="414"
                  y1={py(y)}
                  y2={py(y)}
                  className="sgd-grid"
                />
                <text x="33" y={py(y) + 4} textAnchor="end">
                  {Number(y.toPrecision(2))}
                </text>
              </g>
            ))}
            <g clipPath={`url(#${id}-clip)`}>
              {dataset.validation.map((point, i) => (
                <circle
                  key={`validation-${i}`}
                  cx={px(point.x)}
                  cy={py(point.y)}
                  r="3"
                  className="sgd-heldout-point"
                />
              ))}
              {dataset.points.map((point, i) => (
                <circle
                  key={i}
                  cx={px(point.x)}
                  cy={py(point.y)}
                  r={state.batch.includes(i) ? 4 : 3}
                  className={
                    state.batch.includes(i) ? "sgd-batch" : "sgd-point"
                  }
                />
              ))}
              <line
                x1={px(-xBound)}
                x2={px(xBound)}
                y1={py(-state.weight * xBound + state.bias)}
                y2={py(state.weight * xBound + state.bias)}
                className="sgd-fit"
              />
            </g>
            {[-xBound, 0, xBound].map((x) => (
              <text key={x} x={px(x)} y="211" textAnchor="middle">
                {Number(x.toPrecision(2))}
              </text>
            ))}
          </svg>
          <div className="sgd-legend">
            <span>● training</span>
            {live && <span className="sgd-heldout-key">● held out</span>}
            <span className="sgd-batch-key">● current batch</span>
            <span className="sgd-fit-key">― model</span>
          </div>
        </figure>
        <figure>
          <figcaption>
            Loss <span>mean squared error · log scale</span>
          </figcaption>
          <svg
            viewBox="0 0 440 220"
            role="img"
            aria-label={`Training loss across ${dataset.points.length} points: ${latest.loss.toPrecision(3)}, after ${state.step} steps. Lower is better.`}
          >
            {[0.001, 0.1, 10, 1000].map((loss) => (
              <g key={loss}>
                <line
                  x1="44"
                  x2="414"
                  y1={lossY(loss)}
                  y2={lossY(loss)}
                  className="sgd-grid"
                />
                <text x="37" y={lossY(loss) + 4} textAnchor="end">
                  {loss}
                </text>
              </g>
            ))}
            {state.history
              .filter(
                (point, i) => i > 0 && point.rate !== state.history[i - 1].rate,
              )
              .map((point) => (
                <line
                  key={point.step}
                  x1={44 + (point.step / SGD_STEPS) * 370}
                  x2={44 + (point.step / SGD_STEPS) * 370}
                  y1="24"
                  y2="190"
                  className="sgd-rate-change"
                >
                  <title>
                    Learning rate {rateLabel(point.rate!)} at step {point.step}
                  </title>
                </line>
              ))}
            {live && (
              <polyline
                points={state.history
                  .map(
                    (point) =>
                      `${44 + (point.step / SGD_STEPS) * 370},${lossY(point.validationLoss!)}`,
                  )
                  .join(" ")}
                className="sgd-validation"
              />
            )}
            <polyline
              points={state.history
                .map(
                  (point) =>
                    `${44 + (point.step / SGD_STEPS) * 370},${lossY(point.loss)}`,
                )
                .join(" ")}
              className="sgd-loss"
            />
            <circle
              cx={44 + (state.step / SGD_STEPS) * 370}
              cy={lossY(latest.loss)}
              r="4"
              className="sgd-batch"
            />
            {[0, 40, 80, 120].map((step) => (
              <text
                key={step}
                x={44 + (step / SGD_STEPS) * 370}
                y="211"
                textAnchor="middle"
              >
                {step}
              </text>
            ))}
          </svg>
          <div className="sgd-legend">
            <span>Train {latest.loss.toPrecision(3)}</span>
            {latest.validationLoss !== null && (
              <span className="sgd-heldout-key">
                Held out {latest.validationLoss.toPrecision(3)}
              </span>
            )}
            <span>┆ learning-rate change</span>
          </div>
        </figure>
      </div>
      {live && (
        <p className="sgd-metrics">
          Previous 1s return → next 1s return · normalized axes / MSE
          <br />
          Held-out mean-prediction baseline:{" "}
          {dataset.baselineLoss!.toPrecision(3)} · lower is better
        </p>
      )}
      <div className="sgd-cycle" aria-label="SGD update cycle">
        <span>Sample {SGD_BATCH_SIZE} points</span>
        <b>→</b>
        <span>Predict + measure error</span>
        <b>→</b>
        <span>Compute gradient</span>
        <b>→</b>
        <span>Update weights</span>
      </div>
      <details className="sgd-help">
        <summary>What does the learning rate change?</summary>
        <p>
          <code>new weight = old weight − learning rate × gradient</code>
        </p>
        <p>
          Smaller steps can learn slowly; oversized steps can overshoot. Change
          the rate while running, or reset to compare the same points and
          mini-batches. Batch gradients can be noisy, so loss need not fall at
          every step.
        </p>
        <p>
          This model fits a line with two parameters. Neural networks apply
          gradients to many weights. The real Training Lab uses Adam; this
          slider only controls this SGD illustration. Runs stop at 120 steps or
          loss above 10,000; the fit and loss charts use fixed display bounds.
        </p>
        {live && (
          <p>
            Only completed, consecutive trade candles are used. The earliest 80%
            train the model; one boundary pair is skipped and the later pairs
            are held out. Input and target scales use training data only (1
            input unit = {dataset.xScale!.toPrecision(3)} bps; 1 target unit ={" "}
            {dataset.yScale!.toPrecision(3)} bps). This frozen sample is small
            and noisy: lower training loss does not establish predictive skill
            or profitability. “Train on latest capture” starts a new run; Reset
            keeps this sample.
          </p>
        )}
        <a
          href="https://www.kaggle.com/code/init27/fastai-v3-lesson-2-sgd"
          target="_blank"
          rel="noreferrer"
        >
          Explore the lesson ↗
        </a>
      </details>
    </section>
  );
}
