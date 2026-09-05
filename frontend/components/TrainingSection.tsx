"use client";

import { useTrainingReceipt } from "../hooks/useTrainingReceipt";
import {
  formatTrainingCount,
  formatTrainingDuration,
  formatTrainingLoss,
  type TrainingReceipt,
} from "../lib/training";
import { SectionTitle } from "./Primitives";

const steps = [
  ["01", "Record", "Keep a small public book + trade sample."],
  ["02", "Frame", "Aggregate events into causal one-second features."],
  ["03", "Label", "Turn 30 frames × 10 features into short-horizon utilities."],
  ["04", "Train", "Fit the slow gate, then check later validation examples."],
] as const;

function ReceiptMetrics({ receipt }: { receipt: TrainingReceipt }) {
  const { source, dataset, training } = receipt;
  const metrics = [
    [
      "Sample",
      source.symbol,
      `${source.venue} · ${formatTrainingDuration(source.duration_seconds)}`,
    ],
    [
      "Recorded events",
      formatTrainingCount(source.event_counts.total),
      "total public book + trade events",
    ],
    [
      "Training examples",
      formatTrainingCount(dataset.train_examples),
      `${formatTrainingCount(dataset.validation_examples)} held-out validation`,
    ],
    [
      "Loss / model",
      `${formatTrainingLoss(training.first_train_loss)} → ${formatTrainingLoss(training.last_train_loss)}`,
      `validation ${formatTrainingLoss(training.validation_loss)} · ${formatTrainingCount(training.parameter_count)} parameters`,
    ],
  ];
  return (
    <div className="training-metrics">
      {metrics.map(([label, value, detail]) => (
        <div className="metric" key={label}>
          <span className="metric-label">{label}</span>
          <strong>{value}</strong>
          <span className="metric-detail">{detail}</span>
        </div>
      ))}
    </div>
  );
}

export function TrainingSection() {
  const { receipt, loading, error } = useTrainingReceipt();
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="05"
        eyebrow="training example"
        title="From public ticks to a tiny model"
      />
      <div className="training-pipeline" aria-label="Training pipeline">
        {steps.map(([number, title, text]) => (
          <article className="training-step" key={number}>
            <span className="training-step-number">{number}</span>
            <h3>{title}</h3>
            <p>{text}</p>
          </article>
        ))}
      </div>
      {loading ? (
        <div className="panel training-status" role="status">
          Looking for the latest local training receipt…
        </div>
      ) : receipt ? (
        <div className="training-receipt">
          <div className="training-receipt-heading">
            <h3>{receipt.source.symbol} sample</h3>
            <a href={receipt.source.url} target="_blank" rel="noreferrer">
              public source ↗
            </a>
          </div>
          <ReceiptMetrics receipt={receipt} />
          <p className="training-limitations">
            <strong>Local CPU example only.</strong>{" "}
            {receipt.limitations.join(" ")} Not a trading claim.
          </p>
        </div>
      ) : (
        <div className="panel training-status training-unavailable">
          <strong>No training receipt yet.</strong>
          <span>
            {error || "Start the backend after creating a local sample."}
          </span>
        </div>
      )}
      <div className="training-commands panel">
        <div>
          <span className="panel-kicker teal">RUN IT LOCALLY</span>
          <p>The receipt is from local CPU. Modal remote smoke is optional.</p>
        </div>
        <code>
          <span>uv run --extra training python scripts/record_binance.py</span>
          <span>uv run --extra training python scripts/train_gate.py</span>
          <span>make modal-plan</span>
        </code>
      </div>
    </section>
  );
}
