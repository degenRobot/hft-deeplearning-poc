import {
  age,
  formatTimestamp,
  number,
  percent,
  price,
  signed,
} from "../lib/format";
import type { Expert, MarketGateState } from "../lib/types";
import { Metric, SectionTitle, WeightRow } from "./Primitives";

const modeTitle = (mode: MarketGateState["gate"]["mode"]) =>
  ({
    neural: "Demo neural gate",
    uniform: "Uniform baseline",
    static: "Static baseline",
    "uniform-fallback": "Uniform fallback",
  })[mode];
export const largestWeight = (state: MarketGateState) =>
  Math.max(...Object.values(state.gate.weights));
type MarketMetric = {
  label: string;
  value: (state: MarketGateState) => string;
  detail: (state: MarketGateState) => string;
  tone?: (state: MarketGateState) => string;
};
const marketMetrics: MarketMetric[] = [
  {
    label: "Mid price",
    value: (s) => price(s.market.mid),
    detail: (s) => s.symbol,
    tone: () => "accent",
  },
  {
    label: "Spread",
    value: (s) => `${number(s.market.spread_bps, 2)} bps`,
    detail: () => "top of book",
  },
  {
    label: "Imbalance",
    value: (s) => signed(s.market.imbalance),
    detail: () => "bid pressure",
    tone: (s) => (s.market.imbalance >= 0 ? "positive" : "negative"),
  },
  {
    label: "Trade flow",
    value: (s) => signed(s.market.trade_flow),
    detail: () => "signed impulse",
    tone: (s) => (s.market.trade_flow >= 0 ? "positive" : "negative"),
  },
];
const weightRows = [
  ["Microprice pressure", "microprice", "#0b8f7c"],
  ["Trade-flow impulse", "flow", "#7568b3"],
  ["Short reversion", "reversion", "#b56a3c"],
] as const;

function ExpertCard({
  expert,
  index,
  ready,
}: {
  expert: Expert;
  index: number;
  ready: boolean;
}) {
  const values = [
    ["score", signed(expert.score), expert.score >= 0],
    ["weight", percent(expert.weight), true],
    ["contribution", signed(expert.contribution), expert.contribution >= 0],
  ] as const;
  return (
    <article className="expert-card">
      <div className="expert-index">0{index + 1}</div>
      <h3>{expert.label}</h3>
      <div className="expert-values">
        {values.map(([label, value, positive]) => (
          <div key={label}>
            <span>{label}</span>
            <b
              className={
                label === "weight" ? undefined : positive ? "up" : "down"
              }
            >
              {ready ? value : "—"}
            </b>
          </div>
        ))}
      </div>
    </article>
  );
}

export function MarketSection({
  state,
  ready,
}: {
  state: MarketGateState;
  ready: boolean;
}) {
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="01"
        eyebrow="market state"
        title="What the feed says"
        description="Normalized snapshots from the backend. Values remain blank until a ready snapshot arrives."
      />
      <div className="metric-grid market-metrics">
        {marketMetrics.map((metric) => (
          <Metric
            key={metric.label}
            label={metric.label}
            value={ready ? metric.value(state) : "—"}
            detail={metric.detail(state)}
            tone={metric.tone?.(state)}
          />
        ))}
      </div>
    </section>
  );
}

export function GateSection({
  state,
  ready,
  nextRefresh,
  mode = state.gate.mode,
}: {
  state: MarketGateState;
  ready: boolean;
  nextRefresh: number;
  mode?: MarketGateState["gate"]["mode"];
}) {
  const largest = ready ? largestWeight(state) : 0;
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="02"
        eyebrow="active runtime policy"
        title={`${modeTitle(mode)} sets the mix`}
        description="The bundled demo model or a selected baseline sets bounded expert weights. The offline experiments below do not replace it."
      />
      <div className="split-grid">
        <article className="panel gate-panel">
          <div className="panel-heading">
            <span className="panel-kicker violet">WEIGHT POLICY</span>
            <span className="model-version">
              {ready ? state.gate.model_version : "—"}
            </span>
          </div>
          <div className="regime">
            {ready ? modeTitle(mode) : "Awaiting feed"}
          </div>
          <div className="confidence">
            <span>Largest weight</span>
            <b>{ready ? percent(largest) : "—"}</b>
            <div className="confidence-track">
              <span style={{ width: `${largest * 100}%` }} />
            </div>
          </div>
          <div className="gate-footer">
            <span>
              {ready ? `Next refresh ${age(nextRefresh)}` : "Next refresh —"}
            </span>
            <span>
              {ready ? `${state.gate.mode} · ${state.gate.cadence_ms} ms` : "—"}
            </span>
          </div>
        </article>
        <article className="panel weights-panel">
          <div className="panel-heading">
            <span className="panel-kicker teal">CURRENT WEIGHTS</span>
            <span className="weight-total">Σ 1.00</span>
          </div>
          {weightRows.map(([label, key, color]) => (
            <WeightRow
              key={key}
              label={label}
              value={ready ? state.gate.weights[key] : 0}
              color={color}
            />
          ))}
          <p className="panel-caption">
            {ready
              ? "Bounded and smoothed before reaching the fast mixer."
              : "Waiting for a ready policy snapshot."}
          </p>
        </article>
      </div>
    </section>
  );
}

export function ExpertsSection({
  state,
  ready,
}: {
  state: MarketGateState;
  ready: boolean;
}) {
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="03"
        eyebrow="fast plane"
        title="Three experts, one accountable quote"
        description="Scores are event-speed signals, not trade recommendations. Contributions stay legible."
      />
      <div className="expert-grid">
        {state.experts.map((expert, index) => (
          <ExpertCard
            key={expert.id}
            expert={expert}
            index={index}
            ready={ready}
          />
        ))}
      </div>
    </section>
  );
}

const healthRows = [
  [
    "Risk reason",
    (s: MarketGateState) => s.health.risk_reason || "none reported",
  ],
  ["Feed error", (s: MarketGateState) => s.health.feed_error],
  [
    "Events processed",
    (s: MarketGateState) => number(s.health.events_processed, 0),
  ],
  [
    "Late events dropped",
    (s: MarketGateState) => number(s.health.late_events_dropped, 0),
  ],
  ["Feed generation", (s: MarketGateState) => s.health.feed_generation],
  ["Message age", (s: MarketGateState) => age(s.health.message_age_ms)],
  ["Reconnects", (s: MarketGateState) => s.health.reconnects],
  ["Last snapshot", (s: MarketGateState) => formatTimestamp(s.timestamp)],
] as const;

export function PaperSection({
  state,
  ready,
}: {
  state: MarketGateState;
  ready: boolean;
}) {
  const hasBackendRun = Boolean(state.health.run_id);
  const healthStatus = hasBackendRun ? state.health.feed_status : "waiting";
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="04"
        eyebrow="paper state"
        title="Synthetic quote, no execution"
        description="A toy quote and simulation ledger make the effect inspectable without an order endpoint."
      />
      <div className="lower-grid">
        <article className="quote-panel">
          <span className="panel-kicker lime">SYNTHETIC QUOTE</span>
          {state.quote && ready ? (
            <>
              <div className="quote-values">
                <div>
                  <span>bid</span>
                  <strong>{price(state.quote.bid)}</strong>
                </div>
                <i>/</i>
                <div>
                  <span>ask</span>
                  <strong>{price(state.quote.ask)}</strong>
                </div>
              </div>
              <div className="quote-sub">
                mid {price(state.market.mid)} · spread{" "}
                {number(state.market.spread_bps)} bps
              </div>
            </>
          ) : (
            <div className="quote-empty">
              No quote until a healthy snapshot arrives.
            </div>
          )}
        </article>
        <article className="panel paper-panel">
          <span className="panel-kicker teal">SIMULATION STATE</span>
          <div className="paper-values">
            <Metric
              label="Inventory"
              value={ready ? signed(state.paper.inventory, 4) : "—"}
              detail="BTC · paper only"
            />
            <Metric
              label="P&amp;L"
              value={ready ? signed(state.paper.pnl) : "—"}
              detail="simulation units"
              tone={state.paper.pnl >= 0 ? "positive" : "negative"}
            />
          </div>
        </article>
        <article className="panel health-panel">
          <span className="panel-kicker teal">FEED / RISK HEALTH</span>
          <div className="health-status">
            <span
              className={`health-dot ${healthStatus === "running" ? "ok" : healthStatus === "waiting" ? "waiting" : "warn"}`}
            />
            <strong>{healthStatus}</strong>
          </div>
          <dl>
            {healthRows.map(([label, read]) =>
              label === "Feed error" && !state.health.feed_error ? null : (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{hasBackendRun ? read(state) : "—"}</dd>
                </div>
              ),
            )}
          </dl>
        </article>
      </div>
    </section>
  );
}
