import { EMPTY_STATE, type MarketGateState } from "../lib/types";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export function HistoryChart({ history }: { history: MarketGateState[] }) {
  const width = 760;
  const height = 214;
  const padding = { left: 34, right: 16, top: 16, bottom: 28 };
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  const points = history.length > 1 ? history : [EMPTY_STATE, EMPTY_STATE];
  const toPoint = (value: number, index: number) =>
    `${padding.left + (index / Math.max(points.length - 1, 1)) * usableWidth},${padding.top + (1 - clamp(value, 0, 1)) * usableHeight}`;
  const series = [
    { key: "microprice", color: "#0b8f7c", label: "Microprice" },
    { key: "flow", color: "#7568b3", label: "Flow" },
    { key: "reversion", color: "#b56a3c", label: "Reversion" },
  ] as const;

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <svg
        className="history-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Expert weight history chart"
      >
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={padding.top + (1 - tick) * usableHeight}
              y2={padding.top + (1 - tick) * usableHeight}
              className="chart-grid"
            />
            <text
              x="4"
              y={padding.top + (1 - tick) * usableHeight + 4}
              className="chart-axis"
            >
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        {series.map((item) => (
          <polyline
            key={item.key}
            points={points
              .map((snapshot, index) =>
                toPoint(snapshot.gate.weights[item.key], index),
              )
              .join(" ")}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        <text x={padding.left} y={height - 6} className="chart-axis">
          {history.length ? "earliest" : "waiting for snapshots"}
        </text>
        <text
          x={width - padding.right}
          y={height - 6}
          textAnchor="end"
          className="chart-axis"
        >
          now
        </text>
      </svg>
    </div>
  );
}
