"use client";

import { memo, useState } from "react";
import type { Activations } from "../lib/visual";
import type { MarketGateState } from "../lib/types";

const hues = ["#41d6b2", "#a99bff", "#62c2ed"];
export function activationOpacity(value: number, maximum: number) {
  return value > 0 && maximum > 0
    ? 0.22 + (0.78 * Math.log1p(value)) / Math.log1p(maximum)
    : 0.08;
}

export const ActivationNetwork = memo(function ActivationNetwork({
  mode,
  activations,
  outputs,
  revision,
}: {
  mode?: MarketGateState["gate"]["mode"];
  activations: Activations | null;
  outputs: number[];
  revision: number;
}) {
  const [layer, setLayer] = useState<"hidden_1" | "hidden_2">("hidden_1");
  const [neuron, setNeuron] = useState(0);
  const neural = mode === "neural";
  const live = neural && !!activations;
  const groups = [
    activations?.hidden_1 || Array(64).fill(0),
    activations?.hidden_2 || Array(32).fill(0),
  ];
  return (
    <div className={`network-model ${neural ? "" : "network-bypassed"}`}>
      <div className="flow-card-heading">
        <span className="flow-kicker">
          {!mode
            ? "WAITING FOR POLICY"
            : neural
              ? "02 / NEURAL INFERENCE"
              : `${mode.toUpperCase()} POLICY`}
        </span>
        <span className="flow-small">#{mode ? revision : "—"}</span>
      </div>
      <h3>{!mode || neural ? "The bigger model" : "Neural gate bypassed"}</h3>
      <div className="network-labels">
        <span>64 · ReLU</span>
        <span>32 · ReLU</span>
        <span>3 · weights</span>
      </div>
      <svg
        className="activation-network"
        viewBox="0 0 340 180"
        role="img"
        aria-label={
          live
            ? "Actual activations: 64 neurons in hidden layer one, 32 in hidden layer two, and three output weights"
            : "Neural activation data unavailable"
        }
      >
        <path
          d="M106 90 C140 90 145 90 181 90 M231 90 C265 90 270 30 306 30 M231 90H306 M231 90 C265 90 270 150 306 150"
          className="layer-connections"
        />
        {groups.map((values, l) => {
          const maximum = Math.max(0, ...values);
          const columns = l === 0 ? 8 : 4;
          return values.map((value, i) => (
            <circle
              key={`${l}-${i}`}
              cx={(l === 0 ? 10 : 186) + (i % columns) * 13}
              cy={25 + Math.floor(i / columns) * 18}
              r="4.5"
              fill={hues[l]}
              fillOpacity={live ? activationOpacity(value, maximum) : 0.08}
              data-activation={live ? value : undefined}
            >
              <title>
                {live
                  ? `Hidden ${l + 1} · neuron ${i + 1}: ${value.toPrecision(5)}`
                  : "No activation from a current neural pass"}
              </title>
            </circle>
          ));
        })}
        {outputs.map((value, i) => (
          <g key={i}>
            <circle
              cx="315"
              cy={30 + i * 60}
              r="8"
              fill={["#41d6b2", "#62c2ed", "#a99bff"][i]}
              fillOpacity={live ? 0.1 + value * 0.9 : 0.08}
            >
              <title>
                {live
                  ? `${["Microprice", "Trade flow", "Reversion"][i]}: ${(value * 100).toFixed(2)}%`
                  : "Waiting for neural output"}
              </title>
            </circle>
            <text x="315" y={49 + i * 60} textAnchor="middle">
              {live ? `${(value * 100).toFixed(0)}%` : "—"}
            </text>
          </g>
        ))}
      </svg>
      <div className="activation-legend">
        <span>
          <i />0 / inactive
        </span>
        <span>
          <i />
          stronger activity
        </span>
      </div>
      <p className="flow-footnote">
        {live
          ? "Actual ReLU activations · brightness uses a log scale within each layer and pass."
          : neural
            ? "Waiting for activation telemetry from this pass."
            : mode
              ? "Selected baseline supplies the weights."
              : "Waiting for a fresh snapshot."}
      </p>
      <details className="feature-values activation-inspector">
        <summary>Inspect a neuron &amp; architecture</summary>
        <p>
          300 inputs → 64 ReLU → 32 ReLU → 3 weights. 21,443 parameters. Lines
          show layer flow, not individual connection strengths. Inference, not
          live retraining.
        </p>
        <div className="neuron-controls">
          <label>
            Layer
            <select
              aria-label="Activation layer"
              value={layer}
              onChange={(e) => {
                setLayer(e.target.value as typeof layer);
                setNeuron(0);
              }}
            >
              <option value="hidden_1">Hidden 1 · teal</option>
              <option value="hidden_2">Hidden 2 · violet</option>
            </select>
          </label>
          <label>
            Neuron
            <select
              aria-label="Neuron index"
              value={neuron}
              onChange={(e) => setNeuron(Number(e.target.value))}
            >
              {Array.from(
                { length: layer === "hidden_1" ? 64 : 32 },
                (_, i) => (
                  <option key={i} value={i}>
                    {i + 1}
                  </option>
                ),
              )}
            </select>
          </label>
        </div>
        <output>
          {live
            ? `Activation ${activations![layer][neuron].toPrecision(6)} · pass #${revision}`
            : "No current neural activation"}
        </output>
      </details>
    </div>
  );
});
