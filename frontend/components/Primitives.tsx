import { percent } from "../lib/format";

export function SectionTitle({
  index,
  eyebrow,
  title,
  description,
}: {
  index: string;
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="section-title">
      <div>
        <span className="eyebrow">
          {index} / {eyebrow}
        </span>
        <h2>{title}</h2>
      </div>
      {description && <p>{description}</p>}
    </div>
  );
}

export function Metric({
  label,
  value,
  detail,
  tone = "default",
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: string;
}) {
  return (
    <div className={`metric ${tone}`}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </div>
  );
}

export function WeightRow({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const width = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="weight-row">
      <div className="weight-label">
        <span>{label}</span>
        <b>{percent(value)}</b>
      </div>
      <div className="weight-track">
        <span style={{ width: `${width}%`, background: color }} />
      </div>
    </div>
  );
}
