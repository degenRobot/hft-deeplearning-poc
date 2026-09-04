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
      `${formatTrainingCount(source.event_counts.book)} book · ${formatTrainingCount(source.event_counts.trade)} trade`,
    ],
    [
      "Dataset",
      `${formatTrainingCount(dataset.examples)} examples`,
      `${formatTrainingCount(dataset.frames)} frames · ${dataset.lookback_frames} × ${dataset.feature_names.length} window`,
    ],
    [
      "Chronological split",
      `${formatTrainingCount(dataset.train_examples)} / ${formatTrainingCount(dataset.validation_examples)}`,
      "train / validation examples",
    ],
  ];
  const details = [
    ["Run", `${training.epochs} epochs · seed ${training.seed}`],
    [
      "Train loss",
      `${formatTrainingLoss(training.first_train_loss)} → ${formatTrainingLoss(training.last_train_loss)}`,
    ],
    ["Validation loss", formatTrainingLoss(training.validation_loss)],
    ["Parameters", formatTrainingCount(training.parameter_count)],
    ["Learning rate", training.learning_rate],
    ["Model artifact", training.model_path],
  ];
  return (
    <>
      <div className="training-metrics">
        {metrics.map(([label, value, detail]) => (
          <div className="training-metric" key={label}>
            <span className="metric-label">{label}</span>
            <strong>{value}</strong>
            <span className="metric-detail">{detail}</span>
          </div>
        ))}
      </div>
      <div className="training-detail-grid">
        <dl className="panel training-detail-panel">
          {details.slice(0, 3).map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <dl className="panel training-detail-panel">
          {details.slice(3).map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd
                className={
                  label === "Model artifact" ? "training-path" : undefined
                }
              >
                {value}
              </dd>
            </div>
          ))}
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
