"use client";
import { useEffect, useRef, useState } from "react";
import { API_DEFAULT } from "../lib/connection";
import {
  parsePublicTrainingData,
  captureIsStale,
  ACTIVE_CAPTURE_STATES,
  type PublicTrainingData,
} from "../lib/trainingData";
import "./trainingExtras.css";
const API = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(/\/$/, "");
const time = (ms: number | null | undefined) =>
  ms
    ? new Date(ms).toISOString().replace("T", " ").replace(".000Z", " UTC")
    : "Unavailable";

export function TrainingDataset() {
  const [data, setData] = useState<PublicTrainingData | null>(null);
  const [seconds, setSeconds] = useState(900);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [stale, setStale] = useState(false);
  const command = useRef<
    ((action: "start" | "stop", seconds: number) => Promise<void>) | null
  >(null);
  useEffect(() => {
    let disposed = false,
      version = 0,
      busy = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    const request = async (action?: "start" | "stop", duration?: number) => {
      const epoch = ++version;
      controller?.abort();
      const current = new AbortController();
      controller = current;
      const timeout = setTimeout(() => {
        current.abort();
        if (!disposed && epoch === version) {
          setData(null);
          setStale(false);
          setError(
            "Public dataset status timed out. Capture measurements cleared.",
          );
          busy = false;
          setPending(false);
          timer = setTimeout(() => void request(), 1000);
        }
      }, 10000);
      try {
        if (action) {
          const r = await fetch(`${API}/training/data/${action}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
              action === "start" ? { seconds: duration } : {},
            ),
            signal: current.signal,
          });
          if (!r.ok) throw new Error(`Capture ${action} returned ${r.status}`);
        }
        if (disposed || epoch !== version || current.signal.aborted) return;
        const r = await fetch(`${API}/training/data`, {
          cache: "no-store",
          signal: current.signal,
        });
        if (!r.ok) throw new Error("Public dataset status unavailable");
        const parsed = parsePublicTrainingData(await r.json());
        if (disposed || epoch !== version || current.signal.aborted) return;
        if (!parsed) throw new Error("Public dataset status is invalid");
        if (!disposed && epoch === version) {
          const outdated = captureIsStale(parsed.capture, Date.now());
          setData(parsed);
          setStale(outdated);
          setError(
            outdated
              ? "Capture heartbeat is stale. Current progress is unknown; stop remains available when the backend confirms it owns the capture."
              : "",
          );
        }
      } catch (cause) {
        if (!disposed && epoch === version) {
          setData(null);
          setStale(false);
          setError(
            current.signal.aborted
              ? "Public dataset status timed out. Capture measurements cleared."
              : cause instanceof Error
                ? cause.message
                : "Capture status unavailable",
          );
        }
      } finally {
        clearTimeout(timeout);
        if (!disposed && epoch === version) {
          busy = false;
          setPending(false);
          clearTimeout(timer);
          timer = setTimeout(() => void request(), 1000);
        }
      }
    };
    command.current = async (action, duration) => {
      if (disposed || busy) return;
      busy = true;
      clearTimeout(timer);
      setPending(true);
      await request(action, duration);
    };
    void request();
    return () => {
      disposed = true;
      version++;
      clearTimeout(timer);
      controller?.abort();
      command.current = null;
    };
  }, []);
  const selected = data?.selected,
    capture = data?.capture;
  const active = !!capture && ACTIVE_CAPTURE_STATES.includes(capture.status);
  const valid = Number.isInteger(seconds) && seconds >= 30 && seconds <= 1800;
  return (
    <section className="training-dataset" aria-label="Public training dataset">
      <span className="flow-kicker">PUBLIC DATA / YOUR NEXT TRAINING RUN</span>
      <h2>Learn from real books and trades.</h2>
      <p>
        Binance public BTCUSDT best bid / ask prices and sizes, plus aggregate
        trades. We retain at most one book update per 100 ms and every trade
        within the capture limits. These become 10 features per observed second;
        training takes 30-frame windows and a delayed 5-second learning target.
      </p>
      {selected && (
        <>
          <div className="dataset-facts">
            <span>
              <strong>{selected.event_count.toLocaleString()}</strong>public
              events
            </span>
            <span>
              <strong>{selected.book_count.toLocaleString()}</strong>book
              updates
            </span>
            <span>
              <strong>{selected.trade_count.toLocaleString()}</strong>aggregate
              trades
            </span>
            <span>
              <strong>{(selected.bytes / 1e6).toFixed(2)} MB</strong>
              {selected.training_ready
                ? "Ready for training"
                : "Needs more contiguous data"}
            </span>
          </div>
          <p>
            {selected.label} · {selected.symbol}
            <br />
            {time(selected.first_event_ts_ms)} →{" "}
            {time(selected.last_event_ts_ms)}
          </p>
          {selected.error && <p role="status">{selected.error}</p>}
          <details>
            <summary>Dataset fingerprint and source</summary>
            <p>
              {selected.path}
              <br />
              SHA-256: {selected.sha256 ?? "Unavailable"}
            </p>
            <a
              href="https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md"
              target="_blank"
              rel="noreferrer"
            >
              Binance public market-data documentation ↗
            </a>
          </details>
        </>
      )}
      <p>
        <b>Capture a new time range, starting now.</b> Choose 30–1,800 seconds;
        900 seconds is the working example. This records a live public stream,
        not a historical download. Older trades and candles alone cannot
        reconstruct the required bid/ask features.
      </p>
      <div className="training-capture-controls">
        <label>
          Capture duration (seconds)
          <input
            type="number"
            min={30}
            max={1800}
            step={30}
            value={Number.isFinite(seconds) ? seconds : ""}
            disabled={pending || active}
            onChange={(e) =>
              setSeconds(e.target.value === "" ? NaN : Number(e.target.value))
            }
          />
        </label>
        <button
          className="button primary"
          disabled={!data || !valid || pending || active}
          onClick={() => void command.current?.("start", seconds)}
        >
          {pending ? "Updating capture…" : "Capture more public data"}
        </button>
        <button
          className="button ghost"
          disabled={pending || !capture?.can_stop}
          onClick={() => void command.current?.("stop", seconds)}
        >
          Stop capture
        </button>
      </div>
      {!valid && (
        <p role="alert">Enter a whole duration from 30 to 1,800 seconds.</p>
      )}
      <p className="flow-footnote">
        Runs the{" "}
        <a
          href="https://github.com/degenRobot/hft-deeplearning-poc/blob/codex/live-training-lab/scripts/capture_training_data.py"
          target="_blank"
          rel="noreferrer"
        >
          public capture script ↗
        </a>{" "}
        locally, with 100 MB / 500,000-event caps. A completed capture becomes
        the next training dataset only after the chronological split checks
        pass. Short or interrupted captures are retained; the previous dataset
        stays selected. An active training run keeps its original data.
      </p>
      {capture && !stale && capture.status !== "idle" && (
        <div aria-label="Capture progress">
          <strong>
            {capture.status === "validating"
              ? "Checking the training split"
              : `Capture ${capture.status}`}{" "}
            ·{" "}
            {capture.progress == null
              ? "—"
              : `${Math.floor(capture.progress * 100)}%`}
          </strong>
          {capture.progress != null && (
            <progress
              aria-label="Public data capture completion"
              max={1}
              value={capture.progress}
            />
          )}
          <p>
            {capture.elapsed_seconds == null
              ? "—"
              : Math.round(capture.elapsed_seconds)}{" "}
            / {capture.requested_seconds ?? "—"}s ·{" "}
            {capture.events == null ? "—" : capture.events.toLocaleString()}{" "}
            events ·{" "}
            {capture.bytes == null ? "—" : (capture.bytes / 1e6).toFixed(2)} MB
          </p>
          <p className="flow-footnote">
            Elapsed time includes connection shutdown and training-split
            validation.
          </p>
          {capture.error && <p role="status">{capture.error}</p>}
        </div>
      )}
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p>Reading the selected public dataset…</p>}
    </section>
  );
}
