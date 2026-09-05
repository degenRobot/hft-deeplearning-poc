"use client";
import { useEffect, useRef } from "react";
import { eventTime, signed, type VisualTelemetry } from "../lib/visual";
import "./expert-signals.css";

const expertNames = [
  "Microprice",
  "Trade flow",
  "Reversion",
  "Tiny NN · shadow",
];
export const signalDirection = (score: number) =>
  score > 0 ? "Buy bias" : score < 0 ? "Sell bias" : "Neutral";
export function signalCellColor(score: number | null) {
  if (score === null) return "transparent";
  // One shared scale. Zero stays dark; intensity follows absolute signal strength.
  return score < 0
    ? `rgba(235, 111, 126, ${0.08 + Math.abs(score) * 0.92})`
    : `rgba(65, 214, 178, ${0.08 + Math.abs(score) * 0.92})`;
}

export function ExpertSignalHeatmap({
  visual,
}: {
  visual: VisualTelemetry | null;
}) {
  const scroller = useRef<HTMLDivElement | null>(null);
  const events = visual?.events ?? [];
  const newest = events.at(-1);
  const latestId = newest?.id;
  useEffect(() => {
    if (scroller.current)
      scroller.current.scrollLeft = scroller.current.scrollWidth;
  }, [latestId]);
  const hasNeural =
    Boolean(visual?.neural_expert) ||
    events.some((event) => event.neural_score != null);
  const labels = expertNames.slice(0, hasNeural ? 4 : 3);
  const span =
    newest && events.length > 1
      ? newest.timestamp_ms - events[0].timestamp_ms
      : 0;
  const rate =
    span > 0 ? (((events.length - 1) * 1000) / span).toFixed(1) : null;
  return (
    <article
      className="flow-card expert-history"
      aria-label="Expert signal history"
    >
      <div className="flow-card-heading">
        <span className="flow-kicker">
          EVERY PROCESSED EVENT / RECENT SIGNALS
        </span>
        <span className="flow-small">{events.length} / 64 retained events</span>
      </div>
      <div className="expert-history-heading">
        <h2>Expert signal history</h2>
        <span>
          {rate
            ? `${rate} events/s · retained window`
            : "Waiting to measure event rate"}
        </span>
      </div>
      {hasNeural && (
        <p className="flow-footnote">Tiny NN stays outside the quote mix.</p>
      )}
      <div className="expert-history-legend" aria-label="Shared signal scale">
        <span>
          <i className="expert-history-sell" />
          −1 · Sell bias
        </span>
        <span>
          <i className="expert-history-zero" />0 · Neutral
        </span>
        <span>
          <i className="expert-history-buy" />
          +1 · Buy bias
        </span>
        <span>Brighter = stronger signal, not calibrated confidence</span>
      </div>
      {events.length ? (
        <>
          <div
            className="expert-history-scroll"
            ref={scroller}
            tabIndex={0}
            aria-label="Scrollable event history. Oldest events on the left, newest on the right."
          >
            <table className="expert-history-grid">
              <caption className="expert-history-sr">
                Scores per distinct market event. Each column is an actual
                event, with every row sharing the same minus one to plus one
                scale.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Expert / event</th>
                  {events.map((event) => (
                    <th key={event.id} scope="col">
                      <span className="expert-history-sr">
                        Event {event.id}, {eventTime(event.timestamp_ms)} UTC
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {labels.map((label, row) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    {events.map((event) => {
                      const score =
                        row < 3
                          ? event.scores[row]
                          : (event.neural_score ?? null);
                      const description = `${label} · event ${event.id} · ${eventTime(event.timestamp_ms)} UTC · ${score === null ? "No neural score recorded" : `${signalDirection(score)} ${signed(score, 4)}`}`;
                      return (
                        <td
                          key={event.id}
                          className={`expert-history-cell ${score === null ? "expert-history-missing" : ""}`}
                          style={{ backgroundColor: signalCellColor(score) }}
                          title={description}
                          data-event-id={event.id}
                        >
                          <span className="expert-history-sr">
                            {score === null
                              ? "Not recorded"
                              : `${signalDirection(score)} ${signed(score, 4)}`}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="expert-history-axis">
            <span>Oldest · {eventTime(events[0].timestamp_ms)} UTC</span>
            <span>Newer → {eventTime(newest!.timestamp_ms)} UTC</span>
          </div>
          <div
            className="expert-history-latest"
            aria-label="Latest event scores"
          >
            {labels.map((label, row) => {
              const score =
                row < 3 ? newest!.scores[row] : (newest!.neural_score ?? null);
              return (
                <div key={label}>
                  <span>{label}</span>
                  <strong
                    className={
                      score === null || score === 0
                        ? ""
                        : score > 0
                          ? "expert-history-positive"
                          : "expert-history-negative"
                    }
                  >
                    {score === null
                      ? "Not recorded"
                      : `${signed(score, 3)} · ${signalDirection(score)}`}
                  </strong>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <p className="flow-empty">
          Waiting for processed market events. New columns appear only when real
          event IDs arrive.
        </p>
      )}
      <p className="flow-footnote">
        Sampled history from the backend’s latest 64 processed events.{" "}
        {visual
          ? `${visual.snapshot_interval_ms} ms UI snapshots can carry several events`
          : "UI snapshots can carry several events"}
        ; this is not exchange throughput. Event spacing is by sequence, not
        elapsed time. No events means no new columns.
      </p>
    </article>
  );
}
