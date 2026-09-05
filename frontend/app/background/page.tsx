import Link from "next/link";
import { InferenceBudget } from "../../components/InferenceBudget";
import { PageHeader } from "../../components/PageHeader";
import "./background.css";

export const metadata = {
  title: "Background · Market Gate Lab",
  description:
    "A visual experiment: a slower neural controller guides fast market rules.",
};

const fastPath = [
  ["Market events", "Public books + trades, or replay"],
  ["3 fixed experts", "Pressure · flow · reversion"],
  ["Weighted mix", "Scores × applied weights"],
  ["Risk checks", "Fresh data · paper inventory"],
  ["Synthetic quotes", "Illustrative bid / ask"],
];

const learningRoles = [
  ["Neural gate", "Offline", "Proposes weights for three experts"],
  [
    "Live RL output head",
    "Optional · 99 parameters",
    "Adapts the gate’s output; hidden layers stay frozen",
  ],
  ["Pressure, flow, reversion", "No", "Produce bounded directional scores"],
  ["Mixer", "No", "Sums score × applied weight × expert strength"],
  ["Risk controller", "No", "Decides whether a synthetic quote is allowed"],
  ["Exchange execution", "Not present", "No orders are sent"],
];
const featureGroups = [
  ["Price", "1-second and 5-second return"],
  ["Risk / regime", "Recent volatility and spread"],
  ["Order book", "Imbalance and microprice displacement"],
  ["Flow / activity", "Signed flow, trade arrivals and quote updates"],
  ["Reference value", "Distance from recent fair value"],
];
const expertMeanings = [
  ["Pressure", "Top-of-book imbalance and microprice displacement"],
  ["Flow", "Direction and intensity of recent aggressive trades"],
  ["Reversion", "Distance from the recent local mean"],
];
const example = [
  { name: "Microprice", score: 0.6, weight: 0.5 },
  { name: "Flow", score: 0.2, weight: 0.35 },
  { name: "Reversion", score: -0.4, weight: 0.15 },
];
const signed = (value: number) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;

export default function BackgroundPage() {
  return (
    <main className="shell background-page">
      <PageHeader title="Background" current="background" />

      <section className="background-section" aria-labelledby="poc-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker">01 / Running today</span>
            <h2 id="poc-title">Fast rules. A slower guide.</h2>
          </div>
          <p>
            The network chooses the mix. Every event still passes through
            deterministic rules.
          </p>
        </div>
        <figure className="background-diagram">
          <div className="background-gate-row">
            <div className="background-gate">
              <span className="background-lane-label">Slower control loop</span>
              <div className="background-gate-flow">
                <span>
                  30 × 10
                  <br />
                  <small>Recent features</small>
                </span>
                <b aria-hidden="true">→</b>
                <span>
                  Small neural gate
                  <br />
                  <small>21,443 parameters</small>
                </span>
                <b aria-hidden="true">→</b>
                <span>
                  3 weights
                  <br />
                  <small>Blended + smoothed</small>
                </span>
              </div>
              <span className="background-gate-note">
                Optional Live RL adapts the 99-parameter output head; hidden
                layers stay frozen.
              </span>
            </div>
          </div>
          <div className="background-weight-link">
            <span aria-hidden="true">↓</span>
            <span>Applied weights → weighted mix</span>
          </div>
          <ol
            className="background-route"
            aria-label="Fast event and quote path"
          >
            {fastPath.map(([title, description], index) => (
              <li
                key={title}
                className={index === 2 ? "background-mixer" : undefined}
              >
                <span className="background-step">0{index + 1}</span>
                <strong>{title}</strong>
                <small>{description}</small>
              </li>
            ))}
          </ol>
          <figcaption>
            Small for illustration: three hand-written experts and one neural
            gate. No exchange orders or demonstrated profitability.
          </figcaption>
        </figure>
        <aside className="background-clocks" aria-labelledby="clocks-title">
          <h3 id="clocks-title">Why two clocks?</h3>
          <div className="background-clock-pair">
            <div>
              <strong>Every accepted event</strong>
              <p>
                Books and trades arrive irregularly. With a valid book, the
                three small rules recalculate their scores.
              </p>
            </div>
            <div>
              <strong>About once per second</strong>
              <p>
                The controller reads one-second summaries and refreshes the
                allocation on the next accepted event after its 1,000 ms target
                is due.
              </p>
            </div>
          </div>
          <p className="background-note">
            “Fast” means per event here, not production microseconds. This demo
            still runs due gate inference inline; an independent controller
            worker would keep that wait off the event path.
          </p>
        </aside>
        <details className="background-inputs">
          <summary>What goes into the model and the three rules?</summary>
          <div className="background-reference-grid">
            <div>
              <h3>30 frames × 10 features</h3>
              <p>Roughly 30 seconds of recent context.</p>
              <dl>
                {featureGroups.map(([name, detail]) => (
                  <div key={name}>
                    <dt>{name}</dt>
                    <dd>{detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div>
              <h3>Three specialist rules</h3>
              <p>Fixed formulas, each returning a score from −1 to +1.</p>
              <dl>
                {expertMeanings.map(([name, detail]) => (
                  <div key={name}>
                    <dt>{name}</dt>
                    <dd>{detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </details>
      </section>

      <section className="background-section" aria-labelledby="authority-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker">02 / Learning & authority</span>
            <h2 id="authority-title">What learns? What stays in control?</h2>
          </div>
          <p>
            The model proposes an allocation. Fixed mixing and risk rules retain
            quote authority.
          </p>
        </div>
        <div className="background-authority">
          <table aria-label="Learning scope and control in the demo">
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Learns?</th>
                <th scope="col">What it controls</th>
              </tr>
            </thead>
            <tbody>
              {learningRoles.map(([name, learns, controls]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{learns}</td>
                  <td>{controls}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div
          className="background-neural-roles"
          role="group"
          aria-label="Two neural roles"
        >
          <article>
            <span className="background-lane-label">
              ALLOCATOR · NEURAL GATE
            </span>
            <h3>300 → 64 → 32 → 3</h3>
            <strong>21,443 parameters</strong>
            <p>
              Allocates expert weights and indirectly shapes the synthetic
              quote. Offline training does not automatically replace the live
              checkpoint.
            </p>
          </article>
          <article>
            <span className="background-lane-label">
              OBSERVER · TINY NEURAL EXPERT
            </span>
            <h3>3 → 8 → 1</h3>
            <strong>41 parameters · shadow only</strong>
            <p>
              Reads the three rule scores on fresh accepted events. Excluded
              from the mixer, risk decision and quote; it does not learn live.
            </p>
          </article>
        </div>
        <figure className="background-worked-event">
          <div>
            <span className="panel-kicker">One illustrative event</span>
            <h3>Three contributions. One quote bias.</h3>
            <p>
              Illustrative applied weights, after blending and smoothing. Expert
              strength is 1×, the demo default.
            </p>
          </div>
          <table aria-label="Expert score multiplied by applied weight">
            <thead>
              <tr>
                <th scope="col">Expert</th>
                <th scope="col">Score</th>
                <th scope="col">Weight</th>
                <th scope="col">Contribution</th>
              </tr>
            </thead>
            <tbody>
              {example.map(({ name, score, weight }) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{signed(score)}</td>
                  <td>{(weight * 100).toFixed(0)}%</td>
                  <td>{signed(score * weight)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row" colSpan={3}>
                  Mixed signal · buy bias
                </th>
                <td>
                  {signed(
                    example.reduce(
                      (total, row) => total + row.score * row.weight,
                      0,
                    ),
                  )}
                </td>
              </tr>
            </tfoot>
          </table>
          <figcaption>
            The +0.31 signal can lean the synthetic quote. Stale-data and
            inventory checks can still suppress it.
          </figcaption>
        </figure>
      </section>

      <section className="background-section" aria-labelledby="future-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker violet">
              03 / A possible extension
            </span>
            <h2 id="future-title">More context. More specialists.</h2>
          </div>
          <p>A design to test, not a feature of this POC.</p>
        </div>
        <figure className="background-diagram background-future">
          <div className="background-future-control">
            <div className="background-flow-node">
              <span className="background-lane-label">Richer inputs</span>
              <strong>Venues · cross-market data · inventory</strong>
            </div>
            <span className="background-arrow" aria-hidden="true">
              →
            </span>
            <div className="background-flow-node">
              <span className="background-lane-label">Modular model</span>
              <strong>Encoders · transformers · other components</strong>
            </div>
            <span className="background-arrow" aria-hidden="true">
              →
            </span>
            <div className="background-flow-node">
              <span className="background-lane-label">Slow controller</span>
              <strong>Context → allocations</strong>
            </div>
          </div>
          <div className="background-future-link">
            <span>Proposed weights → weighted decisions</span>
            <span aria-hidden="true">↓</span>
          </div>
          <ol
            className="background-route background-future-route"
            aria-label="Possible future evaluation path"
          >
            <li>
              <span className="background-step">EVENTS</span>
              <strong>N experts / models</strong>
              <small>Specialists with explicit latency budgets</small>
            </li>
            <li className="background-mixer">
              <span className="background-step">MIX</span>
              <strong>Weighted decisions</strong>
              <small>Controller proposes the allocation</small>
            </li>
            <li>
              <span className="background-step">CONSTRAIN</span>
              <strong>Shared risk checks</strong>
              <small>Deterministic limits remain in control</small>
            </li>
            <li>
              <span className="background-step">VALIDATE</span>
              <strong>Shadow evaluation</strong>
              <small>Costs · latency · holdout periods</small>
            </li>
          </ol>
          <figcaption>
            More parameters do not guarantee a better policy. Compare against
            simple baselines before promoting a candidate.
          </figcaption>
        </figure>
      </section>

      <section className="background-section" aria-labelledby="proof-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker">04 / The proof boundary</span>
            <h2 id="proof-title">A signal is only the beginning.</h2>
          </div>
        </div>
        <figure className="background-diagram">
          <ol
            className="background-route background-proof"
            aria-label="From market data to the limit of the demonstration"
          >
            {[
              "Market data",
              "Directional scores",
              "Weighted quote bias",
              "Synthetic quote",
              "Simplified paper fills",
              "Live profitability not demonstrated",
            ].map((step, index) => (
              <li key={step}>
                <span className="background-step">
                  {index === 5 ? "NOT PROVEN" : `0${index + 1}`}
                </span>
                <strong>{step}</strong>
              </li>
            ))}
          </ol>
          <figcaption>
            Fills, queue position, fees, latency, impact and inventory exposure
            can erase forecast skill. The paper simulation does not establish
            live returns.
          </figcaption>
        </figure>
      </section>

      <section className="background-section" aria-labelledby="budget-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker">05 / Try the tradeoff</span>
            <h2 id="budget-title">How often can the model run?</h2>
          </div>
          <p>
            Change the hypothetical inference time. See how much of each update
            interval it consumes.
          </p>
        </div>
        <p className="background-demo-default">
          <strong>Demo default</strong> One-second feature frames · 1,000 ms
          gate target · refresh on the next accepted market event.
        </p>
        <InferenceBudget />
      </section>
      <footer className="background-footer">
        <span>
          Educational simulation · Python + public feeds · no production HFT
          latency claim
        </span>
        <Link href="/">Open Live Terminal →</Link>
      </footer>
    </main>
  );
}
