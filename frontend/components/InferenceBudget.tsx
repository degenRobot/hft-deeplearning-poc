"use client";

import { useState } from "react";

const intervals = [1, 10, 100, 1000];

function numberLabel(value: number) {
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export function InferenceBudget() {
  const [latencyMs, setLatencyMs] = useState(2);
  const [intervalMs, setIntervalMs] = useState(1000);
  const dutyPercent = (latencyMs / intervalMs) * 100;
  const overloaded = latencyMs > intervalMs;

  return (
    <div className="inference-budget">
      <div className="inference-controls">
        <label htmlFor="inference-time">
          Inference time{" "}
          <output htmlFor="inference-time">{numberLabel(latencyMs)} ms</output>
        </label>
        <input
          id="inference-time"
          type="range"
          min="0.1"
          max="100"
          step="0.1"
          value={latencyMs}
          onChange={(event) => setLatencyMs(Number(event.target.value))}
        />
        <label htmlFor="controller-interval">
          Controller interval{" "}
          <output htmlFor="controller-interval">
            {numberLabel(intervalMs)} ms
          </output>
        </label>
        <input
          id="controller-interval"
          type="range"
          min="1"
          max="1000"
          step="1"
          value={intervalMs}
          onChange={(event) => setIntervalMs(Number(event.target.value))}
        />
        <div
          className={`inference-result${overloaded ? " overloaded" : ""}`}
          role="status"
          aria-live="polite"
        >
          <strong>
            {numberLabel(dutyPercent)}% <span>of the interval</span>
          </strong>
          <span>
            {overloaded
              ? "Over budget: one serial worker cannot keep this cadence."
              : latencyMs === intervalMs
                ? "Fully occupied: no time remains for other work."
                : "Inference fits within this interval."}
          </span>
          <span>
            Serial ceiling:{" "}
            <b>{numberLabel(1000 / latencyMs)} updates / second</b>
          </span>
        </div>
      </div>
      <figure className="inference-chart">
        <figcaption>Same model, different controller intervals</figcaption>
        <div className="inference-axis" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100% occupied</span>
        </div>
        <ul aria-label="Inference time as a percentage of each controller interval">
          {intervals.map((interval) => {
            const percent = (latencyMs / interval) * 100;
            return (
              <li key={interval}>
                <span>{interval === 1000 ? "1 s" : `${interval} ms`}</span>
                <span
                  className={`inference-track${percent > 100 ? " over-budget" : ""}`}
                  aria-hidden="true"
                >
                  <span style={{ width: `${Math.min(100, percent)}%` }} />
                  {percent > 100 && <b>→</b>}
                </span>
                <strong>
                  {numberLabel(percent)}%
                  {percent > 100 && <small>over budget</small>}
                </strong>
              </li>
            );
          })}
        </ul>
        <p>
          Bars stop at 100%; arrows mark overflow. Duty cycle = inference time ÷
          interval. One serial worker, without other processing or I/O.
        </p>
        <span className="inference-assumption">
          Illustrative time budget · not measured latency or a price estimate
        </span>
      </figure>
    </div>
  );
}
