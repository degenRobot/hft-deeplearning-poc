import Image from "next/image";
import Link from "next/link";
import { InferenceBudget } from "../../components/InferenceBudget";
import { SiteNav } from "../../components/SiteNav";
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

export default function BackgroundPage() {
  return (
    <main className="shell background-page">
      <header className="topbar background-topbar">
        <Link
          className="brand brand-home"
          href="/"
          aria-label="Market Gate Lab home"
        >
          <span className="brand-mark" aria-hidden="true">
            MG
          </span>
          <div>
            <h1>Background</h1>
            <span className="brand-caption">Market Gate Lab</span>
          </div>
        </Link>
        <SiteNav current="background" />
      </header>

      <figure className="background-banner">
        <div className="background-banner-image">
          <Image
            src="/images/honse.jpg"
            alt="Anime character in side profile"
            fill
            sizes="(max-width: 1600px) 100vw, 1556px"
            priority
          />
        </div>
        <figcaption>
          <span>
            A small experiment in neural control of fast market rules.
          </span>
          <a href="https://blog.sakugabooru.com/wp-content/uploads/2025/04/honse.jpg">
            Image source ↗
          </a>
        </figcaption>
      </figure>

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
      </section>

      <section className="background-section" aria-labelledby="future-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker violet">
              02 / A possible extension
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

      <section className="background-section" aria-labelledby="budget-title">
        <div className="background-section-heading">
          <div>
            <span className="panel-kicker">03 / Try the tradeoff</span>
            <h2 id="budget-title">How often can the model run?</h2>
          </div>
          <p>
            Change the hypothetical inference time. See how much of each update
            interval it consumes.
          </p>
        </div>
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
