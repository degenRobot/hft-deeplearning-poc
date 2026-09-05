"use client";

import { useTrainingReceipt } from "../hooks/useTrainingReceipt";
import experiment from "../lib/experiment-summary.json";
import {
  formatTrainingCount,
  formatTrainingDuration,
  formatTrainingLoss,
  type TrainingReceipt,
} from "../lib/training";
import { SectionTitle } from "./Primitives";

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
        eyebrow="offline experiments"
        title="Three captures, no new model"
        description="The v2 public-data experiment stopped at its data checks. The active demo gate above remains separate."
      />
      <div className="experiment-summary panel">
        <span className="panel-kicker teal">
          V2 · PUBLIC DATA · 5 SEPTEMBER 2026
        </span>
        <div className="training-metrics experiment-metrics">
          {[
            ["Captures attempted", experiment.attempted_samples],
            ["Captures accepted", experiment.accepted_samples],
            [
              "Experimental models trained",
              experiment.experimental_models_trained,
            ],
          ].map(([label, value]) => (
            <div className="metric" key={label}>
              <span className="metric-label">{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
        <table className="experiment-samples">
          <caption>
            Usable examples after data checks. Each capture needed at least{" "}
            {experiment.minimum_train_examples} training and{" "}
            {experiment.minimum_validation_examples} validation examples.
          </caption>
          <thead>
            <tr>
              <th scope="col">Capture</th>
              <th scope="col">Train</th>
              <th scope="col">Validation</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {experiment.samples.map((sample, index) => (
              <tr key={sample.id}>
                <th scope="row">{index + 1}</th>
                <td>{sample.train_examples}</td>
                <td>{sample.validation_examples}</td>
                <td>{sample.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="panel-caption">
          All three captures fell short. No replacement captures were taken;
          there is no new model comparison or performance result to report.
        </p>
      </div>
      <details className="training-archive">
        <summary>Archived v1 · offline training walkthrough</summary>
        <p className="panel-caption">
          This earlier local CPU example illustrates the training steps. Its
          model and losses are separate from the active demo gate and the v2
          experiment above.
        </p>
        {loading ? (
          <div className="panel training-status" role="status">
            Loading the archived v1 receipt…
          </div>
        ) : receipt ? (
          <div className="training-receipt">
            <div className="training-receipt-heading">
              <h3>{receipt.source.symbol} · archived v1 sample</h3>
              <a href={receipt.source.url} target="_blank" rel="noreferrer">
                public source ↗
              </a>
            </div>
            <ReceiptMetrics receipt={receipt} />
            <p className="training-limitations">
              <strong>Archived teaching example.</strong>{" "}
              {receipt.limitations.join(" ")} Not a trading claim.
            </p>
          </div>
        ) : (
          <div className="panel training-status training-unavailable">
            <strong>Archived receipt unavailable.</strong>
            <span>
              {error ||
                "The backend could not provide the archived v1 receipt."}
            </span>
          </div>
        )}
      </details>
    </section>
  );
}
