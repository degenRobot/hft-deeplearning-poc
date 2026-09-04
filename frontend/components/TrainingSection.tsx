"use client";

import {
  formatTrainingCount,
  formatTrainingDuration,
  formatTrainingLoss,
} from "../lib/training";
import { useTrainingReceipt } from "../hooks/useTrainingReceipt";
import type { TrainingReceipt } from "../lib/training";
import { SectionTitle } from "./Primitives";

const steps = [
  {
    number: "01",
    title: "Record",
    text: "Keep a small public book + trade sample.",
  },
  {
    number: "02",
    title: "Frame",
    text: "Aggregate events into causal one-second features.",
  },
  {
    number: "03",
    title: "Label",
    text: "Turn 30 frames × 10 features into short-horizon utilities.",
  },
  {
    number: "04",
    title: "Train",
    text: "Fit the slow gate, then check later validation examples.",
  },
];

function ReceiptMetrics({ receipt }: { receipt: TrainingReceipt }) {
  const { source, dataset, training } = receipt;
  return (
    <>
      <div className="training-metrics">
        <div className="training-metric">
          <span className="metric-label">Sample</span>
          <strong>{source.symbol}</strong>
          <span className="metric-detail">
            {source.venue} · {formatTrainingDuration(source.duration_seconds)}
          </span>
        </div>
        <div className="training-metric">
          <span className="metric-label">Recorded events</span>
          <strong>{formatTrainingCount(source.event_counts.total)}</strong>
          <span className="metric-detail">
            {formatTrainingCount(source.event_counts.book)} book ·{" "}
            {formatTrainingCount(source.event_counts.trade)} trade
          </span>
        </div>
        <div className="training-metric">
          <span className="metric-label">Dataset</span>
          <strong>{formatTrainingCount(dataset.examples)} examples</strong>
          <span className="metric-detail">
            {formatTrainingCount(dataset.frames)} frames ·{" "}
            {dataset.lookback_frames} × {dataset.feature_names.length} window
          </span>
        </div>
        <div className="training-metric">
          <span className="metric-label">Chronological split</span>
          <strong>
            {formatTrainingCount(dataset.train_examples)} /{" "}
            {formatTrainingCount(dataset.validation_examples)}
          </strong>
          <span className="metric-detail">train / validation examples</span>
        </div>
      </div>
      <div className="training-detail-grid">
        <dl className="panel training-detail-panel">
          <div>
            <dt>Run</dt>
            <dd>
              {training.epochs} epochs · seed {training.seed}
            </dd>
          </div>
          <div>
            <dt>Train loss</dt>
            <dd>
              {formatTrainingLoss(training.first_train_loss)} →{" "}
              {formatTrainingLoss(training.last_train_loss)}
            </dd>
          </div>
          <div>
            <dt>Validation loss</dt>
            <dd>{formatTrainingLoss(training.validation_loss)}</dd>
          </div>
        </dl>
        <dl className="panel training-detail-panel">
          <div>
            <dt>Parameters</dt>
            <dd>{formatTrainingCount(training.parameter_count)}</dd>
          </div>
          <div>
            <dt>Learning rate</dt>
            <dd>{training.learning_rate}</dd>
          </div>
          <div>
            <dt>Model artifact</dt>
            <dd className="training-path">{training.model_path}</dd>
          </div>
        </dl>
      </div>
    </>
  );
}

function TrainingCommands() {
  return (
    <div className="training-commands panel">
      <div>
        <span className="panel-kicker teal">RUN IT LOCALLY</span>
        <p>
          The browser only shows the receipt. The committed receipt is from
          local CPU; the optional Modal remote smoke has not run. Use these
          commands to repeat the local path or inspect its parity plan.
        </p>
      </div>
      <code>
        <span>uv run --extra training python scripts/record_binance.py</span>
        <span>uv run --extra training python scripts/train_gate.py</span>
        <span>make modal-plan</span>
      </code>
    </div>
  );
}

export function TrainingSection() {
  const { receipt, loading, error } = useTrainingReceipt();
  return (
    <section className="dashboard-section training-section">
      <SectionTitle
        index="05"
        eyebrow="training example"
        title="From public ticks to a tiny model"
      >
        <span>
          A short, inspectable path from Binance events to a higher-level gate.
          It is a wiring demo, not a trading claim.
        </span>
      </SectionTitle>
      <div className="training-pipeline" aria-label="Training pipeline">
        {steps.map((step) => (
          <article className="training-step" key={step.number}>
            <span className="training-step-number">{step.number}</span>
            <h3>{step.title}</h3>
            <p>{step.text}</p>
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
            <div>
              <span className="panel-kicker violet">LATEST RECEIPT</span>
              <h3>
                {receipt.source.name} · {receipt.source.symbol}
              </h3>
            </div>
            <div className="training-receipt-meta">
              <a
                className="training-source"
                href={receipt.source.url}
                target="_blank"
                rel="noreferrer"
              >
                public source ↗
              </a>
              <span className="training-generated">
                generated {new Date(receipt.generated_at).toLocaleString()}
              </span>
            </div>
          </div>
          <ReceiptMetrics receipt={receipt} />
          <p className="training-limitations">
            <strong>Keep the claim small.</strong>{" "}
            {receipt.limitations.join(" ")}
          </p>
        </div>
      ) : (
        <div className="panel training-status training-unavailable">
          <strong>No training receipt yet.</strong>
          <span>
            {error ||
              "Start the backend after creating a local sample to show its metrics here."}
          </span>
        </div>
      )}
      <TrainingCommands />
    </section>
  );
}
