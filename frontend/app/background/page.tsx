import Link from "next/link";
import { SiteNav } from "../../components/SiteNav";
import "./background.css";

export const metadata = {
  title: "Background · Market Gate Lab",
  description:
    "An interactive experiment in a slower neural controller guiding fast, deterministic market rules.",
};

const experts = [
  {
    name: "Microprice pressure",
    detail: "Reads top-of-book size imbalance and the size-weighted price.",
    className: "pressure",
  },
  {
    name: "Trade-flow impulse",
    detail: "Measures the direction and intensity of recent aggregate trades.",
    className: "flow",
  },
  {
    name: "Short reversion",
    detail: "Responds to price moving away from a recent fair-value estimate.",
    className: "reversion",
  },
];

const questions = [
  {
    title: "Does it hold up on richer data?",
    detail:
      "Capture longer, varied books and trades. Compare against fixed policies on later periods, then add fees, latency and fill assumptions to evaluation.",
  },
  {
    title: "Could context improve the allocation?",
    detail:
      "Test volatility, liquidity and flow-regime features. Keep them only if they improve results across held-out periods, rather than one convenient sample.",
  },
  {
    title: "When should a new model take over?",
    detail:
      "Run candidates in shadow beside the current model. Require data checks, baseline comparisons and a rollback path before allowing promotion.",
  },
  {
    title: "What changes with the full order book?",
    detail:
      "Build a depth adapter with snapshot recovery, sequence checks and venue-specific validation. Then test whether deeper liquidity adds useful information.",
  },
  {
    title: "Can the two time scales stay separate?",
    detail:
      "Move the event and quote path into compiled code, keep the slower controller separate, and measure latency while delayed or missing model updates are injected.",
  },
];

export default function BackgroundPage() {
  return (
    <main className="shell background-page">
      <header className="background-header">
        <Link className="background-brand" href="/">
          <span className="brand-mark" aria-hidden="true">
            MG
          </span>
          Market Gate Lab
        </Link>
        <SiteNav current="background" />
      </header>

      <section className="background-hero" aria-labelledby="background-title">
        <p className="background-kicker">The idea behind the terminal</p>
        <h1 id="background-title">
          Fast rules.
          <br />A slower guide.
        </h1>
        <p className="background-lede">
          Could a neural network help decide which simple market algorithm to
          trust, and by how much? This lab makes that idea visible, from
          incoming market data to a synthetic quote.
        </p>
        <p className="background-intent">
          A fun illustration of potential model control over HFT-style rules.
          Educational simulation; no exchange orders, production-HFT claim or
          proven alpha.
        </p>
        <div className="background-links">
          <Link className="background-primary" href="/">
            Explore the live terminal <span aria-hidden="true">↗</span>
          </Link>
          <Link href="/training">
            Open the training lab <span aria-hidden="true">→</span>
          </Link>
        </div>
      </section>

      <section className="background-section" aria-labelledby="route-title">
        <div className="background-section-heading">
          <p className="background-kicker">01 / The route</p>
          <h2 id="route-title">The model sets the mix.</h2>
          <p>
            Three fixed algorithms react to accepted book and trade events. The
            neural gate refreshes their weights once per second by default.
            Deterministic code checks the result before producing a quote.
          </p>
        </div>
        <figure className="background-diagram">
          <div className="background-controller">
            <span className="background-kicker">Slower control loop</span>
            <strong>
              Recent features → neural gate → allocation + smoothing
            </strong>
            <span>
              30 one-second frames × 10 features · 21,443 model parameters
            </span>
          </div>
          <div className="background-control-link">
            <span aria-hidden="true">↓</span> Applied weights feed the mixer
          </div>
          <ol className="background-route" aria-label="Event and quote path">
            <li>
              <span>01</span>
              <strong>Public data</strong>
              <small>
                Books + trades
                <br />
                or replay
              </small>
            </li>
            <li>
              <span>02</span>
              <strong>Fixed experts</strong>
              <small>
                Three bounded
                <br />
                directional signals
              </small>
            </li>
            <li className="background-mixer">
              <span>03</span>
              <strong>Weighted mix</strong>
              <small>
                Expert scores ×<br />
                applied weights
              </small>
            </li>
            <li>
              <span>04</span>
              <strong>Risk checks</strong>
              <small>
                Data freshness +<br />
                paper inventory
              </small>
            </li>
            <li>
              <span>05</span>
              <strong>Synthetic quote</strong>
              <small>
                Illustrative bid/ask
                <br />
                No order route
              </small>
            </li>
          </ol>
          <figcaption>
            The gate proposes an allocation. Its influence is blended with
            uniform weights and smoothed over time. Invalid or stale books and
            the paper inventory cap can suppress the quote regardless of the
            network&apos;s output.
          </figcaption>
        </figure>
        <div className="background-experts">
          {experts.map((expert) => (
            <article
              className={`background-expert ${expert.className}`}
              key={expert.name}
            >
              <h3>{expert.name}</h3>
              <p>{expert.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="background-section" aria-labelledby="learning-title">
        <div className="background-section-heading">
          <p className="background-kicker">02 / What changes</p>
          <h2 id="learning-title">
            Inference and learning are different views.
          </h2>
        </div>
        <div className="background-learning-grid">
          <article className="background-card">
            <span className="background-tag">Live terminal</span>
            <h3>A small adaptive head</h3>
            <p>
              The terminal normally runs inference. Optional Live RL updates 99
              output weights and biases from a delayed directional reward. Both
              hidden layers and the three expert rules stay frozen.
            </p>
            <p>
              Updates stay in memory. Pause keeps them; reset or a runtime
              settings change restores the checkpoint. The network still acts
              through the same smoothed mixture and deterministic checks.
            </p>
          </article>
          <article className="background-card">
            <span className="background-tag">Training lab</span>
            <h3>Recorded data, visible updates</h3>
            <p>
              Supervised training and replay policy updates change real model
              parameters. A later holdout measures expert-proxy utility without
              updating the model. Training does not automatically replace the
              running gate.
            </p>
            <p>
              Historical candles use an offline approximation: candle closes
              stand in for midprices and missing book features are zero. These
              teaching models cannot be loaded by the live book-based gate.
            </p>
          </article>
        </div>
      </section>

      <aside className="background-boundary" aria-labelledby="boundary-title">
        <p className="background-kicker">What the screen can tell you</p>
        <h2 id="boundary-title">A working mechanism is the starting point.</h2>
        <p>
          You can inspect the inputs, activations, weights and quote checks.
          That does not establish an edge. Proxy rewards and paper results omit
          realistic queue position, partial fills, fees, latency and market
          impact. Python and public internet feeds make the experiment easy to
          follow; they do not demonstrate colocated HFT performance.
        </p>
      </aside>

      <section className="background-section" aria-labelledby="next-title">
        <div className="background-section-heading">
          <p className="background-kicker">03 / Open questions</p>
          <h2 id="next-title">What would be worth testing next?</h2>
          <p>
            These are future experiments, not capabilities the demo already has.
          </p>
        </div>
        <ol className="background-questions">
          {questions.map((question) => (
            <li key={question.title}>
              <h3>{question.title}</h3>
              <p>{question.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <footer className="background-footer">
        <span>Follow a signal, then inspect how its weight changes.</span>
        <div className="background-links">
          <Link href="/">
            Live terminal <span aria-hidden="true">↗</span>
          </Link>
          <Link href="/training">
            Training lab <span aria-hidden="true">↗</span>
          </Link>
        </div>
      </footer>
    </main>
  );
}
