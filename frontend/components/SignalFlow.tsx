"use client";

import { useState } from "react";
import type { CSSProperties } from "react";
import type { MarketGateState } from "../lib/types";
import {
  EXPERT_KEYS,
  FEATURE_ROWS,
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
const colors = ["#bade79", "#74cfca", "#bba3ec"];
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
        <span className="flow-kicker">01 / MARKET INPUT</span>
        <button
          className="flow-toggle"
          aria-pressed={tradesOnly}
          onClick={() => setTradesOnly(!tradesOnly)}
        >
          {tradesOnly ? "Trades only" : "Book + trades"}
        </button>
      </div>
      <h2>The event stream</h2>
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
              ? "INPUT TO LAST INFERENCE"
              : "FEATURE BUFFER / BASELINE IGNORES IT"}
        </span>
        <span className="flow-small">{frames.length}/30 frames</span>
      </div>
      <h3>Ticks → one-second features</h3>
      <div
        className="heatmap"
        role="img"
        aria-label={`${frames.length} observed feature frames, ${padding} zero-padded frames; ten features per frame`}
      >
        {FEATURE_ROWS.map(([label, scale], r) => (
          <div className="heat-row" key={label}>
            <span>{label}</span>
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
                            background: v! < 0 ? "#bba3ec" : "#bade79",
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
        <summary>Inspect latest frame values</summary>
        {frames.length ? (
          <dl>
            {FEATURE_ROWS.map(([label], i) => (
              <div key={label}>
                <dt>{label}</dt>
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

function Network({
  revision,
  mode,
}: {
  revision: number;
  mode?: MarketGateState["gate"]["mode"];
}) {
  const neural = mode === "neural";
  const layers = [6, 8, 5, 3];
  return (
    <div className={`network-model ${neural ? "" : "network-bypassed"}`}>
      <span className="flow-kicker">
        {!mode
          ? "WAITING FOR POLICY"
          : neural
            ? "SLOW NEURAL GATE"
            : `${mode.toUpperCase()} POLICY`}
      </span>
      <h3>{!mode || neural ? "The bigger model" : "Neural gate bypassed"}</h3>
      <div className="network-pulse" key={revision}>
        <svg
          viewBox="0 0 270 150"
          role="img"
          aria-label="Schematic of the 300 input, 64 hidden, 32 hidden, 3 output neural network; dots are representative, not neuron activations"
        >
          {layers
            .slice(0, -1)
            .flatMap((n, l) =>
              Array.from({ length: n }, (_, a) =>
                Array.from({ length: layers[l + 1] }, (_, b) => (
                  <line
                    key={`${l}-${a}-${b}`}
                    x1={24 + l * 74}
                    y1={22 + (a * 104) / (n - 1)}
                    x2={24 + (l + 1) * 74}
                    y2={22 + (b * 104) / (layers[l + 1] - 1)}
                  />
                )),
              ),
            )}
          {layers.flatMap((n, l) =>
            Array.from({ length: n }, (_, i) => (
              <circle
                key={`${l}-${i}`}
                cx={24 + l * 74}
                cy={22 + (i * 104) / (n - 1)}
                r={l === 3 ? 6 : 4}
                style={{ fill: l === 3 ? colors[i] : undefined }}
              />
            )),
          )}
        </svg>
      </div>
      <div className="network-sizes">
        <span>300</span>
        <span>64</span>
        <span>32</span>
        <span>3</span>
      </div>
      <p className="flow-footnote">
        21,443 parameters · network schematic.{" "}
        {neural
          ? "Inference, not live retraining."
          : mode
            ? "Selected baseline supplies the weights."
            : "Waiting for a fresh snapshot."}
      </p>
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
      <div className="fast-lane">
        <EventTape events={events} />
        <article className="flow-card experts-card">
          <div className="flow-card-heading">
            <span className="flow-kicker">02 / FAST SIGNALS</span>
            <span className="speed-badge">EVERY EVENT</span>
          </div>
          <h2>Three small experts</h2>
          <p className="flow-footnote">
            Deterministic signals · traces show recent events on a −1 to +1
            scale
          </p>
          {EXPERT_KEYS.map((id, i) => {
            const expert = state.experts.find((e) => e.id === id);
            return (
              <div
                className="visual-expert"
                key={id}
                style={{ "--expert-color": colors[i] } as CSSProperties}
              >
                <div className="expert-line">
                  <span>
                    <i />
                    {labels[i]}
                  </span>
                  <strong>
                    {ready && expert ? signed(expert.score) : "—"}
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
        </article>
        <article className="flow-card mix-card">
          <span className="flow-kicker">03 / COMBINE + GUARD</span>
          <h2>One quote signal</h2>
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
      <div className="slow-divider">
        <span>↓ aggregate events</span>
        <span className="slow-clock">
          SLOW PATH ·{" "}
          {ready ? `${state.gate.cadence_ms} ms configured cadence` : "waiting"}
        </span>
        <span>↑ weights feed the fast mix</span>
      </div>
      <div className="slow-lane">
        <FeatureWindow visual={visual} neural={neural} />
        <Network
          revision={active ? state.gate.revision : 0}
          mode={active ? state.gate.mode : undefined}
        />
        <div className="weight-output">
          <span className="flow-kicker">OUTPUT / EXPERT WEIGHTS</span>
          <h3>Change the mix</h3>
          <div className="weight-head">
            <span>Expert</span>
            <span>Proposed</span>
            <span>Applied</span>
          </div>
          {EXPERT_KEYS.map((id, i) => (
            <div className="output-weight" key={id}>
              <div>
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
            Refresh occurs on the next market event after the cadence is due.
          </p>
        </div>
      </div>
      {!visual && ready ? (
        <p className="flow-empty">
          Visual telemetry unavailable. Restart the backend with the latest
          code.
        </p>
      ) : null}
    </section>
  );
}
