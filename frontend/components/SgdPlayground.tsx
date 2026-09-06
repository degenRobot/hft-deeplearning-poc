"use client";

import { useEffect, useId, useState } from "react";
import {
  initialSgdState,
  sgdStep,
  SGD_POINTS,
  SGD_STEPS,
  SGD_BATCH_SIZE,
} from "../lib/sgdDemo";
import "./sgd-playground.css";

const presets = [
  ["Small", 0.005],
  ["Steady", 0.1],
  ["Too large", 1.2],
] as const;
const px = (x: number) => 44 + ((x + 1) / 2) * 370;
const py = (y: number) => 190 - ((y + 2.5) / 5) * 166;
const lossY = (loss: number) =>
  190 - ((Math.max(-3, Math.min(4, Math.log10(loss))) + 3) / 7) * 166;
const rateLabel = (rate: number) => Number(rate.toPrecision(3)).toString();

export function SgdPlayground() {
  const id = useId();
  const [state, setState] = useState(initialSgdState);
  const [rate, setRate] = useState(0.1);
  const [playing, setPlaying] = useState(false);
  const done = state.diverged || state.step >= SGD_STEPS;
  const running = playing && !done;
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setState((previous) => sgdStep(previous, rate));
    }, 140);
    return () => window.clearInterval(timer);
  }, [running, rate]);
  const latest = state.history.at(-1)!;
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
          <span className="flow-kicker">INTERACTIVE SGD ILLUSTRATION</span>
          <h2 id={`${id}-title`}>See how a model learns</h2>
          <p>Synthetic data · separate from your training run</p>
        </div>
        <span
          className={state.diverged ? "sgd-diverged" : "sgd-status"}
          role="status"
        >
          {status}
        </span>
      </div>
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
              {label}
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
            onClick={() => setState((previous) => sgdStep(previous, rate))}
          >
            Step
          </button>
          <button
            className="button ghost"
            type="button"
            onClick={() => {
              setPlaying(false);
              setState(initialSgdState());
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <div className="sgd-charts">
        <figure>
          <figcaption>
            Fit the points{" "}
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
            {[-2, 0, 2].map((y) => (
              <g key={y}>
                <line
                  x1="44"
                  x2="414"
                  y1={py(y)}
                  y2={py(y)}
                  className="sgd-grid"
                />
                <text x="33" y={py(y) + 4} textAnchor="end">
                  {y}
                </text>
              </g>
            ))}
            <g clipPath={`url(#${id}-clip)`}>
              {SGD_POINTS.map((point, i) => (
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
                x1={px(-1)}
                x2={px(1)}
                y1={py(-state.weight + state.bias)}
                y2={py(state.weight + state.bias)}
                className="sgd-fit"
              />
            </g>
            {[-1, 0, 1].map((x) => (
              <text key={x} x={px(x)} y="211" textAnchor="middle">
                {x}
              </text>
            ))}
          </svg>
          <div className="sgd-legend">
            <span>● data</span>
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
            aria-label={`Loss across all 48 points: ${latest.loss.toPrecision(3)}, after ${state.step} steps. Lower is better.`}
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
            <span>Loss {latest.loss.toPrecision(3)}</span>
            <span>┆ learning-rate change</span>
          </div>
        </figure>
      </div>
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
          This example fits a line with two parameters. Neural networks apply
          gradients to many weights. The real Training Lab uses Adam; this
          slider only controls this SGD illustration. Runs stop at 120 steps or
          loss above 10,000; the fit and loss charts use fixed display bounds.
        </p>
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
