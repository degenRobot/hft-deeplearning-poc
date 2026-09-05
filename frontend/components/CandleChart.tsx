"use client";

import { useState } from "react";
import type { Candle } from "../lib/visual";
import { eventTime } from "../lib/visual";

const price = (n: number) =>
  n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
export function CandleChart({
  candles,
  symbol,
}: {
  candles: Candle[];
  symbol: string;
}) {
  const [range, setRange] = useState(60);
  const [selected, setSelected] = useState<number | null>(null);
  const last = candles.at(-1);
  const end = last?.timestamp_ms || 0;
  const start = end - (range - 1) * 1000;
  const visible = candles.filter((c) => c.timestamp_ms >= start);
  const current = visible.find((c) => c.timestamp_ms === selected) || last;
  const low = visible.length ? Math.min(...visible.map((c) => c.low)) : 0;
  const high = visible.length ? Math.max(...visible.map((c) => c.high)) : 1;
  const pad = Math.max((high - low) * 0.15, high * 0.000001, 0.01);
  const floor = low - pad,
    ceiling = high + pad;
  const y = (n: number) => 22 + ((ceiling - n) / (ceiling - floor)) * 210;
  const x = (t: number) => 12 + ((t - start) / 1000) * (558 / range);
  const width = Math.max(2, (558 / range) * 0.64);
  const maxVolume = Math.max(1e-9, ...visible.map((c) => c.volume));
  return (
    <article className="flow-card candle-card">
      <div className="flow-card-heading">
        <span className="flow-kicker">MARKET / TRADE OHLC</span>
        <div className="chart-ranges" aria-label="Candle chart time range">
          {[30, 60, 90].map((n) => (
            <button
              key={n}
              className="flow-toggle"
              aria-pressed={range === n}
              onClick={() => {
                setRange(n);
                setSelected(null);
              }}
            >
              {n}s
            </button>
          ))}
        </div>
      </div>
      <div className="chart-title">
        <h2>
          {symbol} <span>1s candles</span>
        </h2>
        <strong
          className={last && last.close < last.open ? "negative" : "positive"}
        >
          {last ? price(last.close) : "—"}
        </strong>
      </div>
      <div className="ohlc-readout" aria-label="Selected candle OHLC">
        {current ? (
          <>
            <time>{eventTime(current.timestamp_ms).slice(0, 8)} UTC</time>
            <span>
              O <b>{price(current.open)}</b>
            </span>
            <span>
              H <b>{price(current.high)}</b>
            </span>
            <span>
              L <b>{price(current.low)}</b>
            </span>
            <span>
              C <b>{price(current.close)}</b>
            </span>
          </>
        ) : (
          <span>Waiting for trades to build candles…</span>
        )}
      </div>
      <svg
        className="candle-chart"
        viewBox="0 0 700 300"
        role="img"
        aria-label={`${visible.length} observed one-second trade candles. Empty seconds are gaps. Latest candle may still be forming.`}
      >
        {Array.from({ length: 5 }, (_, i) => {
          const value = ceiling - (i * (ceiling - floor)) / 4;
          return (
            <g key={i}>
              <line
                x1="8"
                x2="570"
                y1={y(value)}
                y2={y(value)}
                className="chart-grid"
              />
              <text x="690" textAnchor="end" y={y(value) + 3}>
                {visible.length ? price(value) : "—"}
              </text>
            </g>
          );
        })}
        {visible.map((c) => {
          const cx = x(c.timestamp_ms) + width / 2;
          const color = c.close >= c.open ? "#41d6b2" : "#f17b8e";
          return (
            <g
              key={c.timestamp_ms}
              onMouseEnter={() => setSelected(c.timestamp_ms)}
            >
              <title>{`${eventTime(c.timestamp_ms)} UTC · O ${price(c.open)} H ${price(c.high)} L ${price(c.low)} C ${price(c.close)} · volume ${c.volume}`}</title>
              <line
                x1={cx}
                x2={cx}
                y1={y(c.high)}
                y2={y(c.low)}
                stroke={color}
              />
              <rect
                x={cx - width / 2}
                y={Math.min(y(c.open), y(c.close))}
                width={width}
                height={Math.max(1.5, Math.abs(y(c.open) - y(c.close)))}
                fill={color}
              />
              <rect
                x={cx - width / 2}
                y={275 - (c.volume / maxVolume) * 30}
                width={width}
                height={Math.max(1, (c.volume / maxVolume) * 30)}
                fill={color}
                opacity=".4"
              />
            </g>
          );
        })}
        {last && (
          <line
            x1="8"
            x2="570"
            y1={y(last.close)}
            y2={y(last.close)}
            className="chart-last"
          />
        )}
        {[0, 0.5, 1].map((part, i) => (
          <text
            key={i}
            x={12 + part * 548}
            y="296"
            textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
          >
            {last
              ? eventTime(start + (range - 1) * 1000 * part).slice(0, 8)
              : "—"}
          </text>
        ))}
      </svg>
      <div className="chart-caption">
        <span>
          Observed trades · UTC receive clock on live feed · volume below
        </span>
        <span>Latest candle may be forming</span>
      </div>
      <details className="feature-values candle-inspector">
        <summary>Inspect candle values</summary>
        <label>
          Candle
          <select
            aria-label="Candle timestamp"
            value={current?.timestamp_ms || ""}
            onChange={(e) => setSelected(Number(e.target.value))}
          >
            {!visible.length && <option value="">Waiting for trades</option>}
            {[...visible].reverse().map((c) => (
              <option key={c.timestamp_ms} value={c.timestamp_ms}>
                {eventTime(c.timestamp_ms).slice(0, 8)} UTC
              </option>
            ))}
          </select>
        </label>
        <p>
          {current
            ? `O ${price(current.open)} · H ${price(current.high)} · L ${price(current.low)} · C ${price(current.close)} · Volume ${current.volume.toPrecision(5)}`
            : "No observed candles yet."}{" "}
          Empty seconds stay empty; history starts when this run starts.
        </p>
      </details>
    </article>
  );
}
