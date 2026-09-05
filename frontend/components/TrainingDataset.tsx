"use client";
import { useEffect, useRef, useState } from "react";
import { API_DEFAULT } from "../lib/connection";
import {
  parsePublicTrainingData,
  historicalRequest,
  defaultHistoricalRange,
  HISTORY_SYMBOLS,
  type HistoricalRequest,
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
  const [mode, setMode] = useState("historical");
  const [observedNow, setObservedNow] = useState(0);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [range, setRange] = useState({ start: "", end: "" });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [stale, setStale] = useState(false);
  const command = useRef<
    | ((
        action: "start" | "stop" | "history",
        payload?: number | HistoricalRequest,
      ) => Promise<void>)
    | null
  >(null);
  useEffect(() => {
    let disposed = false,
      version = 0,
      busy = false;
    // Initialize the UTC picker after mounting, without a server/client date mismatch.
    void Promise.resolve().then(() => {
      if (!disposed) {
        const now = Date.now();
        setRange(defaultHistoricalRange(now));
        setObservedNow(now);
      }
    });
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    const request = async (
      action?: "start" | "stop" | "history",
      payload?: number | HistoricalRequest,
    ) => {
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
              action === "start"
                ? { seconds: payload }
                : action === "history"
                  ? payload
                  : {},
            ),
            signal: current.signal,
          });
          if (!r.ok) {
            let detail: unknown;
            if (r.status === 400) {
              try {
                detail = (await r.json())?.detail;
              } catch {
                /* Use status if no JSON detail. */
              }
            }
            throw new Error(
              typeof detail === "string" &&
              detail.length > 0 &&
              detail.length <= 500
                ? detail
                : `Public data ${action} returned ${r.status}`,
            );
          }
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
          setObservedNow(Date.now());
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
  const liveValid =
    Number.isInteger(seconds) && seconds >= 30 && seconds <= 1800;
  const history = historicalRequest(
    symbol,
    range.start,
    range.end,
    observedNow,
  );
  const valid = mode === "historical" ? !!history : liveValid;
  const historical =
    selected?.source_mode === "historical_candles_1s" ||
    selected?.source === "binance_historical_candles";
  const downloading = capture?.mode === "historical";
  return (
    <section className="training-dataset" aria-label="Public training dataset">
      <span className="flow-kicker">PUBLIC DATA / YOUR NEXT TRAINING RUN</span>
      <h2>Choose a pair and a past time range.</h2>
      <p>
        Fetch free Binance historical 1-second candles without an API key, or
        capture a live stream. Both feed 30-frame windows and a delayed 5-second
        learning target.
      </p>
      {selected && (
        <>
          <div className="dataset-facts">
            <span>
              <strong>{selected.event_count.toLocaleString()}</strong>
              {historical ? "candles" : "public events"}
            </span>
            {historical ? (
              <span>
                <strong>1 second</strong>OHLCV candle interval
              </span>
            ) : (
              <>
                <span>
                  <strong>{selected.book_count.toLocaleString()}</strong>book
                  updates
                </span>
                <span>
                  <strong>{selected.trade_count.toLocaleString()}</strong>
                  aggregate trades
                </span>
              </>
            )}
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
          <p>
            {historical
              ? "Historical OHLCV, taker-buy volume and trade counts support close-price, flow and reversion proxies. Spread, book imbalance, microprice and quote updates are unavailable and zero-filled. This dataset cannot validate order-book strategies."
              : `${selected.symbol} best bid / ask prices and sizes, plus aggregate trades. Live captures retain at most one book update per 100 ms and every trade within the capture limits.`}
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
      <div className="training-capture-controls">
        <label>
          Data acquisition
          <select
            value={mode}
            disabled={pending || active}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="historical">Historical data</option>
            <option value="live">Live capture</option>
          </select>
        </label>
        {mode === "historical" ? (
          <>
            <label>
              Trading pair
              <select
                value={symbol}
                disabled={pending || active}
                onChange={(e) => setSymbol(e.target.value)}
              >
                {HISTORY_SYMBOLS.map((pair) => (
                  <option key={pair} value={pair}>
                    {pair}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Start (UTC, inclusive)
              <input
                type="datetime-local"
                step={1}
                value={range.start}
                disabled={pending || active}
                onInput={(e) => {
                  const value = e.currentTarget.value;
                  setRange((r) => ({ ...r, start: value }));
                }}
                onChange={(e) => {
                  const value = e.target.value;
                  setRange((r) => ({ ...r, start: value }));
                }}
              />
            </label>
            <label>
              End (UTC, exclusive)
              <input
                type="datetime-local"
                step={1}
                value={range.end}
                disabled={pending || active}
                onInput={(e) => {
                  const value = e.currentTarget.value;
                  setRange((r) => ({ ...r, end: value }));
                }}
                onChange={(e) => {
                  const value = e.target.value;
                  setRange((r) => ({ ...r, end: value }));
                }}
              />
            </label>
          </>
        ) : (
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
        )}
        <button
          className="button primary"
          disabled={!data || !valid || pending || active}
          onClick={() => {
            if (mode === "historical") {
              if (history) void command.current?.("history", history);
            } else void command.current?.("start", seconds);
          }}
        >
          {pending
            ? "Updating data…"
            : mode === "historical"
              ? "Fetch historical data"
              : "Capture more public data"}
        </button>
        <button
          className="button ghost"
          disabled={pending || !capture?.can_stop}
          onClick={() => void command.current?.("stop")}
        >
          Stop capture
        </button>
      </div>
      {mode === "historical" ? (
        <p className="flow-footnote">
          Enter UTC times, regardless of your device timezone. Choose a closed
          past range of 10 minutes to 5 hours in whole seconds; the end second
          is excluded.
        </p>
      ) : (
        <p className="flow-footnote">
          Captures live books and trades starting now for 30–1,800 seconds.
        </p>
      )}
      {!valid && (
        <p role="alert">
          {mode === "historical"
            ? "Choose a valid past UTC range of 10 minutes to 5 hours, in whole seconds."
            : "Enter a whole duration from 30 to 1,800 seconds."}
        </p>
      )}
      <p className="flow-footnote">
        Runs the{" "}
        <a
          href={`https://github.com/degenRobot/hft-deeplearning-poc/blob/codex/live-training-lab/scripts/${mode === "historical" ? "fetch_training_history" : "capture_training_data"}.py`}
          target="_blank"
          rel="noreferrer"
        >
          {mode === "historical"
            ? "historical download script"
            : "public capture script"}{" "}
          ↗
        </a>{" "}
        locally. A dataset becomes selected only after chronological split
        checks pass. Short or interrupted acquisitions retain the previous
        selection. An active training run keeps its original data.
      </p>
      {capture && !stale && capture.status !== "idle" && (
        <div aria-label="Capture progress">
          <strong>
            {capture.status === "validating"
              ? "Checking the training split"
              : `${downloading ? "Historical download" : "Capture"} ${capture.status}`}{" "}
            ·{" "}
            {capture.progress == null
              ? "—"
              : `${Math.floor(capture.progress * 100)}%`}
          </strong>
          {capture.progress != null && (
            <progress
              aria-label={
                downloading
                  ? "Historical time range processed"
                  : "Public data capture completion"
              }
              max={1}
              value={capture.progress}
            />
          )}
          <p>
            {downloading ? (
              <>
                {capture.candle_count ?? capture.events ?? "—"} candles ·{" "}
                {capture.requested_seconds ?? "—"}s requested market span
              </>
            ) : (
              <>
                {capture.events == null ? "—" : capture.events.toLocaleString()}{" "}
                events · {capture.requested_seconds ?? "—"}s requested capture
              </>
            )}
            {" · "}
            {capture.bytes == null ? "—" : (capture.bytes / 1e6).toFixed(2)} MB
          </p>
          <p className="flow-footnote">
            Elapsed:{" "}
            {capture.elapsed_seconds == null
              ? "—"
              : Math.round(capture.elapsed_seconds)}
            s, including shutdown and validation.
            {downloading &&
              " Progress measures the time range processed, not elapsed time; gaps are not filled."}
          </p>
          {downloading && capture.coverage_fraction != null && (
            <p className="flow-footnote">
              Candles received: {(capture.coverage_fraction * 100).toFixed(1)}%
              of requested seconds
              {capture.missing_candle_count != null &&
                ` · ${capture.missing_candle_count} seconds without downloaded candles`}
            </p>
          )}
          {capture.error && <p role="status">{capture.error}</p>}
        </div>
      )}
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p>Reading the selected public dataset…</p>}
    </section>
  );
}
