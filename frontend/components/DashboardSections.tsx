import {
  age,
  formatTimestamp,
  number,
  percent,
  price,
  signed,
} from "../lib/format";
import type { MarketGateState } from "../lib/types";
import { HistoryChart } from "./HistoryChart";
import { Metric, SectionTitle, WeightRow } from "./Primitives";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));
const modeTitle = (mode: MarketGateState["gate"]["mode"]) =>
  ({
    neural: "Neural model",
    uniform: "Uniform baseline",
    static: "Static baseline",
    "uniform-fallback": "Uniform fallback",
  })[mode];
const largestWeight = (state: MarketGateState) =>
  Math.max(
    state.gate.weights.microprice,
    state.gate.weights.flow,
    state.gate.weights.reversion,
  );

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
      >
        <span>
          Normalized snapshots from the backend. Values remain blank until a
          ready snapshot arrives.
        </span>
      </SectionTitle>
      <div className="metric-grid market-metrics">
        <Metric
          label="Mid price"
          value={ready ? price(state.market.mid) : "—"}
          detail={state.symbol}
          tone="accent"
        />
        <Metric
          label="Spread"
          value={ready ? `${number(state.market.spread_bps, 2)} bps` : "—"}
          detail="top of book"
        />
        <Metric
          label="Imbalance"
          value={ready ? signed(state.market.imbalance) : "—"}
          detail="bid pressure"
          tone={state.market.imbalance >= 0 ? "positive" : "negative"}
        />
        <Metric
          label="Trade flow"
          value={ready ? signed(state.market.trade_flow) : "—"}
          detail="signed impulse"
          tone={state.market.trade_flow >= 0 ? "positive" : "negative"}
        />
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
        eyebrow="slow plane"
        title={`${modeTitle(mode)} sets the mix`}
      >
        <span>
          Refreshes at its configured cadence. It only changes bounded expert
          weights.
        </span>
      </SectionTitle>
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
          <WeightRow
            label="Microprice pressure"
            value={ready ? state.gate.weights.microprice : 0}
            color="#0b8f7c"
          />
          <WeightRow
            label="Trade-flow impulse"
            value={ready ? state.gate.weights.flow : 0}
            color="#7568b3"
          />
          <WeightRow
            label="Short reversion"
            value={ready ? state.gate.weights.reversion : 0}
            color="#b56a3c"
          />
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
  history,
  ready,
}: {
  state: MarketGateState;
  history: MarketGateState[];
  ready: boolean;
}) {
  return (
    <section className="dashboard-section">
      <SectionTitle
        index="03"
        eyebrow="fast plane"
        title="Three experts, one accountable quote"
      >
        <span>
          Scores are event-speed signals, not trade recommendations.
          Contributions stay legible.
        </span>
      </SectionTitle>
      <div className="expert-grid">
        {state.experts.map((expert, index) => (
          <article className="expert-card" key={expert.id}>
            <div className="expert-index">0{index + 1}</div>
            <h3>{expert.label}</h3>
            <div className="expert-values">
              <div>
                <span>score</span>
                <b className={expert.score >= 0 ? "up" : "down"}>
                  {ready ? signed(expert.score) : "—"}
                </b>
              </div>
              <div>
                <span>weight</span>
                <b>{ready ? percent(expert.weight) : "—"}</b>
              </div>
              <div>
                <span>contribution</span>
                <b className={expert.contribution >= 0 ? "up" : "down"}>
                  {ready ? signed(expert.contribution) : "—"}
                </b>
              </div>
            </div>
          </article>
        ))}
      </div>
      <div className="panel chart-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker teal">
              WEIGHT / CONTRIBUTION HISTORY
            </span>
            <h3>How the mix has moved</h3>
          </div>
          <span className="chart-window">last {history.length} revisions</span>
        </div>
        <HistoryChart history={history} />
      </div>
    </section>
  );
}

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
      >
        <span>
          A toy quote and simulation ledger make the effect inspectable without
          an order endpoint.
        </span>
      </SectionTitle>
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
            <div>
              <dt>Risk reason</dt>
              <dd>
                {hasBackendRun
                  ? state.health.risk_reason || "none reported"
                  : "—"}
              </dd>
            </div>
            {state.health.feed_error && (
              <div>
                <dt>Feed error</dt>
                <dd>{hasBackendRun ? state.health.feed_error : "—"}</dd>
              </div>
            )}
            <div>
              <dt>Events processed</dt>
              <dd>
                {hasBackendRun ? number(state.health.events_processed, 0) : "—"}
              </dd>
            </div>
            <div>
              <dt>Late events dropped</dt>
              <dd>
                {hasBackendRun
                  ? number(state.health.late_events_dropped, 0)
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>Feed generation</dt>
              <dd>{hasBackendRun ? state.health.feed_generation : "—"}</dd>
            </div>
            <div>
              <dt>Message age</dt>
              <dd>
                {hasBackendRun && state.timestamp
                  ? age(state.health.message_age_ms)
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>Reconnects</dt>
              <dd>{hasBackendRun ? state.health.reconnects : "—"}</dd>
            </div>
            <div>
              <dt>Last snapshot</dt>
              <dd>{formatTimestamp(hasBackendRun ? state.timestamp : "")}</dd>
            </div>
          </dl>
        </article>
      </div>
    </section>
  );
}
