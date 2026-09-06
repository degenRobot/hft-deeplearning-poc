"use client";

import { useState } from "react";
import { ContextTip } from "./ContextTip";
import "./terminalPolish.css";
import { ActivationNetwork } from "./ActivationNetwork";
import { ExpertSignalHeatmap, signalDirection } from "./ExpertSignalHeatmap";
import { CandleChart } from "./CandleChart";
import type { CSSProperties } from "react";
import type { MarketGateState } from "../lib/types";
import {
  EXPERT_KEYS,
  FEATURE_ROWS,
  FEATURE_HELP,
  eventTime,
  signed,
  type VisualEvent,
  type VisualTelemetry,
} from "../lib/visual";

const labels = ["Microprice", "Trade flow", "Reversion"];
const inputs = [
  "bid / ask pressure",
  "recent buy / sell volume",
  "price vs recent fair value",
];
const descriptions = [
  "Combines bid/ask depth imbalance with the depth-weighted microprice’s distance from the mid. More bid pressure leans buy.",
  "Compares recent buyer- and seller-initiated volume. More buyer volume leans buy, with strength increasing over recent trade arrivals.",
  "Compares the current mid with the recent mean mid. Below that mean leans buy; above it leans sell.",
];
const colors = ["#41d6b2", "#62c2ed", "#a99bff"];
const price = (value: number) =>
  value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

function Spark({ values, color }: { values: number[]; color: string }) {
  const points = values
    .map(
      (v, i) =>
        `${((i / Math.max(1, values.length - 1)) * 220).toFixed(1)},${(22 - Math.max(-1, Math.min(1, v)) * 19).toFixed(1)}`,
    )
    .join(" ");
  return (
    <svg
      className="signal-spark"
      viewBox="0 0 220 44"
      role="img"
      aria-label="Recent event-by-event signal, fixed scale minus one to plus one"
    >
      <path d="M0 22H220" className="spark-zero" />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  );
}

function EventTape({ events }: { events: VisualEvent[] }) {
  const [tradesOnly, setTradesOnly] = useState(false);
  const rows = (tradesOnly ? events.filter((e) => e.kind !== "book") : events)
    .slice(-12)
    .reverse();
  return (
    <article className="flow-card tape-card">
      <div className="flow-card-heading">
        <span className="flow-kicker">MARKET / EVENT TAPE</span>
        <button
          className="flow-toggle"
          aria-pressed={tradesOnly}
          onClick={() => setTradesOnly(!tradesOnly)}
        >
          {tradesOnly ? "Trades only" : "Book + trades"}
        </button>
      </div>
      <h2>Event stream</h2>
      <div className="tape-columns">
        <span>TIME · UTC</span>
        <span>TYPE</span>
        <span>PRICE</span>
      </div>
      <div className="tape-rows" aria-label="Recent processed market events">
        {rows.length ? (
          rows.map((e) => (
            <div
              className={`tape-row ${e.kind}`}
              key={e.id}
              tabIndex={0}
              title={`Event ${e.id} · ${e.kind === "book" ? "book mid; combined bid + ask depth" : "trade price and quantity"} · size ${e.size} · mixed signal ${signed(e.signal, 4)} · gate revision ${e.gate_revision}`}
            >
              <time>{eventTime(e.timestamp_ms)}</time>
              <b>{e.kind.toUpperCase()}</b>
              <span>{price(e.price)}</span>
            </div>
          ))
        ) : (
          <p className="flow-empty">Waiting for processed events…</p>
        )}
      </div>
      <div className="tape-legend">
        <span>
          <i className="buy-dot" />
          BUY
        </span>
        <span>
          <i className="sell-dot" />
          SELL
        </span>
        <span>
          <i className="book-dot" />
          BOOK MID
        </span>
      </div>
      <p className="flow-footnote">
        Latest processed events · 10 Hz screen. Live timestamps use the local
        receive clock.
      </p>
    </article>
  );
}

function FeatureWindow({
  visual,
  neural,
}: {
  visual: VisualTelemetry | null;
  neural: boolean;
}) {
  const frames = visual?.window || [];
  const padding = 30 - frames.length;
  const duration =
    frames.length > 1
      ? ((frames.at(-1)!.timestamp_ms - frames[0].timestamp_ms) / 1000).toFixed(
          1,
        )
      : "0";
  return (
    <div className="feature-window">
      <div className="flow-card-heading">
        <span className="flow-kicker">
          {!visual
            ? "WAITING FOR FEATURE DATA"
            : neural
              ? "01 / INPUT TO LAST INFERENCE"
              : "FEATURE BUFFER / BASELINE IGNORES IT"}
        </span>
        <span className="flow-small">{frames.length}/30 frames</span>
      </div>
      <h3>What the model sees</h3>
      <p className="input-explanation">
        Each column is one observed second; each row describes price, liquidity
        or trading activity. The last 30 frames become 300 model inputs.
      </p>
      <div
        className="heatmap"
        role="img"
        aria-label={`${frames.length} observed feature frames, ${padding} zero-padded frames; ten features per frame`}
      >
        {FEATURE_ROWS.map(([label, scale], r) => (
          <div className="heat-row" key={label}>
            <span title={FEATURE_HELP[r]}>{label}</span>
            <div>
              {Array.from({ length: 30 }, (_, c) => {
                const frame = frames[c - padding];
                const v = frame?.values[r];
                return (
                  <i
                    key={c}
                    className={frame ? "feature-cell" : "feature-cell padded"}
                    style={
                      frame
                        ? {
                            background: v! < 0 ? "#a99bff" : "#41d6b2",
                            opacity:
                              0.15 + Math.min(1, Math.abs(v!) / scale) * 0.85,
                          }
                        : undefined
                    }
                    title={
                      frame
                        ? `${label}: ${v!.toPrecision(5)} · ${eventTime(frame.timestamp_ms)} UTC`
                        : "Zero padding: no observed frame"
                    }
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="heat-axis">
        <span>older</span>
        <span>newer →</span>
      </div>
      <p className="flow-footnote">
        {duration}s between first and last frame · {padding} padded. Each row
        has its own color scale. Green is positive; violet is negative.
      </p>
      <details className="feature-values">
        <summary>Input guide &amp; latest values</summary>
        {frames.length ? (
          <dl>
            {FEATURE_ROWS.map(([label], i) => (
              <div key={label}>
                <dt>
                  {label}
                  <small>{FEATURE_HELP[i]}</small>
                </dt>
                <dd>{frames.at(-1)!.values[i].toPrecision(5)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="flow-footnote">No observed frame yet.</p>
        )}
      </details>
    </div>
  );
}

export function SignalFlow({
  state,
  ready,
  nextRefresh,
}: {
  state: MarketGateState;
  ready: boolean;
  nextRefresh: number;
}) {
  const [motion, setMotion] = useState(true);
  const visual = ready ? state.visual : null;
  const events = visual?.events || [];
  const latest = events.at(-1);
  const spanMs =
    events.length > 1 ? latest!.timestamp_ms - events[0].timestamp_ms : 0;
  const recentPace =
    spanMs > 0 ? (((events.length - 1) * 1000) / spanMs).toFixed(1) : null;
  const neural = ready && state.gate.mode === "neural";
  const active = ready && !!visual;
  const total = ready
    ? state.experts.reduce((n, e) => n + e.contribution, 0)
    : 0;
  const progress = active
    ? Math.max(
        0,
        Math.min(100, 100 - (nextRefresh / state.gate.cadence_ms) * 100),
      )
    : 0;
  return (
    <section
      className={`signal-lab ${motion && active ? "motion-on" : "motion-off"}`}
      aria-label="Live two-speed model flow"
    >
      <div className="flow-toolbar">
        <div>
          <span className={`live-indicator ${active ? "active" : ""}`} />
          <strong>
            {active
              ? state.source === "binance"
                ? "LIVE PUBLIC FEED"
                : "REPLAY STREAM"
              : "AWAITING FEED"}
          </strong>
          <span>{state.symbol}</span>
          <span title="Average spacing over retained processed events; not exchange throughput">
            {recentPace
              ? `${recentPace} events/s · recent window`
              : "measuring event pace"}
          </span>
        </div>
        <button
          className="flow-toggle"
          aria-pressed={!motion}
          onClick={() => setMotion(!motion)}
        >
          {motion ? "Reduce motion" : "Enable motion"}
        </button>
      </div>
      <div className="market-lane">
        <EventTape events={events} />
        <CandleChart candles={visual?.candles || []} symbol={state.symbol} />
      </div>
      <div className="fast-lane">
        <article className="flow-card experts-card">
          <div className="flow-card-heading">
            <span className="flow-kicker">FAST PATH / EVERY EVENT</span>
            <span className="speed-badge">EVERY EVENT</span>
          </div>
          <h2 className="heading-with-tip">
            Three hand-coded experts{" "}
            <ContextTip label="Can there be more experts?">
              Simple examples for this demo. A larger neural model could weight
              decisions from N specialist HFT algorithms, each with its own
              latency budget. Here, only these three rules feed the mix.
            </ContextTip>
          </h2>
          <p className="flow-footnote">
            Fixed rules score every event. The gate sets their weights.
          </p>
          <div className="expert-cards">
            {EXPERT_KEYS.map((id, i) => {
              const expert = state.experts.find((e) => e.id === id);
              return (
                <div
                  className="visual-expert"
                  key={id}
                  style={{ "--expert-color": colors[i] } as CSSProperties}
                >
                  <span className="expert-kind">HAND-CODED RULE</span>
                  <div className="expert-line">
                    <span>
                      <i />
                      {labels[i]}
                    </span>
                    <strong>
                      {ready && expert ? signed(expert.score) : "—"}
                      {ready && expert && (
                        <small className="expert-rule-direction">
                          {signalDirection(expert.score)}
                        </small>
                      )}
                    </strong>
                  </div>
                  <div className="expert-subline">
                    <span>{inputs[i]}</span>
                    <span>
                      ×{" "}
                      {ready && expert
                        ? `${(expert.weight * 100).toFixed(1)}%`
                        : "—"}
                    </span>
                  </div>
                  <p className="expert-rule-description">{descriptions[i]}</p>
                  <Spark
                    values={events.map((e) => e.scores[i])}
                    color={colors[i]}
                  />
                  <div className="contribution-line">
                    <span>weighted contribution</span>
                    <b>
                      {ready && expert ? signed(expert.contribution, 3) : "—"}
                    </b>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="flow-footnote">
            − sell bias · + buy bias · strength, not calibrated confidence.
          </p>
          {visual?.neural_expert && (
            <div className="neural-shadow-card">
              <span className="expert-kind">
                TRAINED MODEL / EXPERIMENTAL SHADOW
              </span>
              <div className="neural-shadow-title">
                <h3 className="heading-with-tip">
                  Tiny neural expert{" "}
                  <ContextTip label="What does the shadow model do?">
                    A learned combination of the three rule scores. Its output
                    is excluded from the expert allocation, mixed signal and
                    synthetic quote. The local forward-pass timing excludes
                    market and network latency.
                  </ContextTip>
                </h3>
                <strong
                  className={
                    visual.neural_expert.score < 0
                      ? "expert-history-negative"
                      : "expert-history-positive"
                  }
                >
                  {signed(visual.neural_expert.score, 3)}
                </strong>
              </div>
              <p className="flow-footnote">
                {signalDirection(visual.neural_expert.score)} · shadow only,
                excluded from the quote mix.
              </p>
              <div className="neural-shadow-metadata">
                <span>
                  {visual.neural_expert.parameter_count.toLocaleString("en-US")}{" "}
                  parameters
                </span>
                <span>
                  {visual.neural_expert.inference_us.toFixed(1)} μs inference
                </span>
                <code>{visual.neural_expert.model_version}</code>
              </div>
              <div className="neural-shadow-inputs">
                Rule score inputs:{" "}
                {visual.neural_expert.inputs
                  .map((value, index) => `${labels[index]} ${signed(value, 4)}`)
                  .join(" · ")}
              </div>
              <p className="flow-footnote">
                3 inputs → 8 neurons → 1 score ·{" "}
                <a
                  href="https://github.com/degenRobot/hft-deeplearning-poc/blob/main/scripts/train_tiny_expert.py"
                  target="_blank"
                  rel="noreferrer"
                >
                  How this tiny model was trained ↗
                </a>
              </p>
            </div>
          )}
        </article>
      </div>
      <div className="terminal-section-label">
        <span>MODEL CONTROL</span>
        <span>
          30 × 10 inputs → neural gate → expert allocation ·{" "}
          {ready ? `${state.gate.cadence_ms} ms cadence` : "waiting"}
        </span>
      </div>
      <div className="slow-lane">
        <FeatureWindow visual={visual} neural={neural} />
        <ActivationNetwork
          activations={neural ? visual?.activations || null : null}
          outputs={EXPERT_KEYS.map((id) => visual?.proposed[id] || 0)}
          revision={active ? state.gate.revision : 0}
          mode={active ? state.gate.mode : undefined}
        />
        <div className="weight-output">
          <span className="flow-kicker">03 / WEIGHT ALLOCATION</span>
          <h3>Control the fast experts</h3>
          <div className="weight-head">
            <span>Expert</span>
            <span>Proposed</span>
            <span>Applied</span>
          </div>
          {EXPERT_KEYS.map((id, i) => (
            <div className="output-weight" key={id}>
              <div className="weight-inline-head">
                <span>Expert</span>
                <span>Proposed</span>
                <span>Applied</span>
              </div>
              <div className="weight-values">
                <span style={{ color: colors[i] }}>{labels[i]}</span>
                <span>
                  {active ? `${(visual!.proposed[id] * 100).toFixed(1)}%` : "—"}
                </span>
                <b>
                  {active
                    ? `${(state.gate.weights[id] * 100).toFixed(1)}%`
                    : "—"}
                </b>
              </div>
              <div className="weight-track">
                <i
                  style={{
                    width: `${active ? state.gate.weights[id] * 100 : 0}%`,
                    background: colors[i],
                  }}
                />
              </div>
            </div>
          ))}
          <p className="flow-footnote">
            {active
              ? `${Math.round(visual!.influence * 100)}% policy influence + smoothing`
              : "Waiting for model telemetry"}
          </p>
          <div className="refresh-track">
            <i style={{ width: `${progress}%` }} />
          </div>
          <div className="refresh-label">
            <span>Update #{active ? state.gate.revision : "—"}</span>
            <span>
              {active ? `${Math.round(nextRefresh)} ms to cadence` : "—"}
            </span>
          </div>
          <p className="flow-footnote">
            Applied weights scale each expert’s contribution. Refresh waits for
            the next market event.
          </p>
        </div>
      </div>
      <div className="model-to-experts">
        <span>Applied allocation → weighted signal</span>
        <div>
          {EXPERT_KEYS.map((id, i) => (
            <span key={id} style={{ color: colors[i] }}>
              {labels[i]}{" "}
              <b>
                {active ? `${(state.gate.weights[id] * 100).toFixed(1)}%` : "—"}
              </b>
            </span>
          ))}
        </div>
      </div>

      <div
        className="route-map"
        aria-label="Market events feed fast experts and one-second feature frames. The neural gate supplies expert weights. Mixed signals pass risk checks before a synthetic quote."
      >
        <span>Market events</span>
        <b>→</b>
        <span>Fast experts</span>
        <b>→</b>
        <span>Weighted mix</span>
        <b>→</b>
        <span>Risk → quote</span>
      </div>
      <div className="event-transit" key={latest?.id || 0} aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      <div className="signal-lane">
        <article className="flow-card mix-card">
          <span className="flow-kicker">SIGNAL / RISK</span>
          <h2 className="heading-with-tip">
            One quote signal{" "}
            <ContextTip label="Could the model control more than direction?">
              This demo uses one weighted directional signal. A richer design
              could use multiple gates for strategy selection, quote size and
              spread, with every output subject to fixed risk limits.
            </ContextTip>
          </h2>
          <div
            className={`mixed-number ${total < 0 ? "negative" : "positive"}`}
          >
            {ready ? signed(total, 3) : "—"}
          </div>
          <div className="mixed-meter">
            <i
              style={{ left: `${50 + Math.min(1, Math.max(-1, total)) * 46}%` }}
            />
          </div>
          <div className="meter-labels">
            <span>lean sell</span>
            <span>lean buy</span>
          </div>
          <div className="contribution-stack">
            {state.experts.map((e, i) => (
              <div key={e.id}>
                <span style={{ color: colors[i] }}>{labels[i]}</span>
                <b>{ready ? signed(e.contribution, 3) : "—"}</b>
              </div>
            ))}
          </div>
          <div className="risk-node">
            <span>{ready && state.quote ? "✓" : "○"}</span>
            <div>
              <strong>Risk controller</strong>
              <small>
                {ready
                  ? state.health.risk_reason ||
                    "Book fresh · inventory within limit"
                  : "Waiting for fresh data"}
              </small>
            </div>
          </div>
          <div className="quote-node">
            <span>SYNTHETIC BID / ASK</span>
            <strong>
              {ready && state.quote
                ? `${price(state.quote.bid)} / ${price(state.quote.ask)}`
                : "Quote withheld"}
            </strong>
          </div>
          <p className="flow-footnote">
            Expert score × weight × strength → sum.
            <br />
            Signal + inventory → guarded price skew.
          </p>
        </article>
      </div>
      <ExpertSignalHeatmap visual={visual} />
      {!visual && ready ? (
        <p className="flow-empty">
          Visual telemetry unavailable. Restart the backend with the latest
          code.
        </p>
      ) : null}
    </section>
  );
}
