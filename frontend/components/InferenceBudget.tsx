"use client";

import { useState } from "react";

const intervals = [1, 10, 100, 1000];

function numberLabel(value: number) {
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export function InferenceBudget() {
  const [hardware, setHardware] = useState("CPU");
  const [workers, setWorkers] = useState(1);
  const [latencyMs, setLatencyMs] = useState(2);
  const [intervalMs, setIntervalMs] = useState(1000);
  const dutyPercent = (latencyMs / intervalMs) * 100;
  const overloaded = latencyMs > intervalMs;
  const poolCapacity = (workers * 1000) / latencyMs;
  const requestedRate = 1000 / intervalMs;

  return (
    <div className="inference-budget">
      <div className="inference-controls">
        <div className="inference-hardware">
          <label htmlFor="compute-hardware">
            Hardware
            <select
              id="compute-hardware"
              value={hardware}
              onChange={(event) => setHardware(event.target.value)}
            >
              <option value="CPU">CPU</option>
              <option value="GPU">GPU / accelerator</option>
            </select>
          </label>
          <label htmlFor="inference-workers">
            Independent workers
            <select
              id="inference-workers"
              value={workers}
              onChange={(event) => setWorkers(Number(event.target.value))}
            >
              {[1, 2, 4, 8, 16].map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="inference-hardware-note">
          Enter measured or assumed time for your model on {hardware}. Selecting
          GPU does not assume a speedup.
        </p>
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
              ? "Over latency budget: each result takes longer than the update interval."
              : latencyMs === intervalMs
                ? "Fully occupied: no time remains for other work."
                : "Inference fits within this interval."}
          </span>
          <span>
            Per-worker ceiling:{" "}
            <b>{numberLabel(1000 / latencyMs)} updates / second</b>
          </span>
          <div className="inference-pool">
            <strong>
              {workers} {hardware} {workers === 1 ? "worker" : "workers"} ·
              ideal parallel capacity
            </strong>
            <span>
              <b>{numberLabel(poolCapacity)} predictions / second</b> for a
              requested {numberLabel(requestedRate)} / second
            </span>
            <span>
              {poolCapacity < requestedRate
                ? "Not enough capacity for this request rate."
                : poolCapacity === requestedRate
                  ? "At capacity: no throughput headroom."
                  : "Capacity fits this request rate."}
            </span>
            <span>
              More workers allow overlapping requests; each result still takes{" "}
              {numberLabel(latencyMs)} ms.
            </span>
          </div>
        </div>
      </div>
      <figure className="inference-chart">
        <figcaption>
          Per-result latency on {hardware}, different intervals
        </figcaption>
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
          interval. Extra workers do not shorten this per-result latency.
        </p>
        <span className="inference-assumption">
          Illustrative budget: ideal independent workers, no batching,
          contention, transfer or queue overhead. A worker is not a GPU core;
          several workers sharing one GPU may not scale linearly.
        </span>
        <a
          className="inference-benchmark-link"
          href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html"
        >
          Benchmark your actual model and hardware ↗
        </a>
      </figure>
    </div>
  );
}
