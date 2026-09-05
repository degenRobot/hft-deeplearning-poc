"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { API_DEFAULT } from "../lib/connection";
import "./trainingExtras.css";
import { LearningActivity } from "./LearningActivity";
const API = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(/\/$/, "");
export type LearningStatus = {
  enabled: boolean;
  interval_seconds: number;
  stage: string;
  updates: number;
  source: string;
  symbol: string;
  run_id: string;
  model_version: string | null;
  weight_delta: number;
  reward: number | null;
  step_duration_ms: number | null;
  next_update_in_seconds: number | null;
  error: string | null;
};
export function parseLearning(value: unknown): LearningStatus | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  const finite = (x: unknown): x is number =>
    typeof x === "number" && Number.isFinite(x);
  if (
    typeof v.enabled !== "boolean" ||
    !finite(v.interval_seconds) ||
    !Number.isInteger(v.interval_seconds) ||
    v.interval_seconds < 5 ||
    v.interval_seconds > 60 ||
    !finite(v.updates) ||
    v.updates < 0 ||
    !Number.isInteger(v.updates) ||
    !finite(v.weight_delta) ||
    v.weight_delta < 0 ||
    !(v.reward === null || finite(v.reward)) ||
    ![v.step_duration_ms, v.next_update_in_seconds].every(
      (x) => x === null || (finite(x) && x >= 0),
    ) ||
    ![v.source, v.symbol, v.run_id, v.stage].every(
      (x) => typeof x === "string",
    ) ||
    !(v.error === null || typeof v.error === "string") ||
    !(v.model_version === null || typeof v.model_version === "string")
  )
    return null;
  return v as LearningStatus;
}
export function useLiveLearning() {
  const [data, setData] = useState<LearningStatus | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const command = useRef<
    ((patch: Record<string, boolean | number>) => void) | null
  >(null);
  useEffect(() => {
    let disposed = false,
      version = 0,
      busy = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout>;
    let activeTimeout: ReturnType<typeof setTimeout> | null = null;
    async function request(patch?: Record<string, boolean | number>) {
      const generation = ++version;
      if (activeTimeout !== null) clearTimeout(activeTimeout);
      controller?.abort();
      const current = new AbortController();
      controller = current;
      const timeout = setTimeout(() => {
        if (disposed || generation !== version) return;
        current.abort();
        version++;
        busy = false;
        setPending(false);
        setData(null);
        setError("Live RL connection timed out");
        timer = setTimeout(() => void request(), 1000);
      }, 5000);
      activeTimeout = timeout;
      try {
        const result = await fetch(`${API}/learning`, {
          method: patch ? "PATCH" : "GET",
          cache: "no-store",
          signal: current.signal,
          ...(patch
            ? {
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patch),
              }
            : {}),
        });
        if (!result.ok) {
          const body = await result.json().catch(() => null);
          throw new Error(
            typeof body?.detail === "string"
              ? body.detail.slice(0, 180)
              : `Live RL returned ${result.status}`,
          );
        }
        const parsed = parseLearning(await result.json());
        if (!parsed) throw new Error("Live RL telemetry unavailable");
        if (disposed || generation !== version || current.signal.aborted)
          return;
        setData(parsed);
        setError("");
      } catch (cause) {
        if (disposed || generation !== version) return;
        setData(null);
        setError(
          current.signal.aborted
            ? "Live RL connection timed out"
            : cause instanceof Error
              ? cause.message
              : "Live RL unavailable",
        );
      } finally {
        clearTimeout(timeout);
        if (!disposed && generation === version) {
          busy = false;
          setPending(false);
          timer = setTimeout(() => void request(), 1000);
        }
      }
    }
    command.current = (patch) => {
      if (disposed || busy) return;
      busy = true;
      clearTimeout(timer);
      setPending(true);
      void request(patch);
    };
    void request();
    return () => {
      disposed = true;
      version++;
      controller?.abort();
      clearTimeout(timer);
      if (activeTimeout !== null) clearTimeout(activeTimeout);
      command.current = null;
    };
  }, []);
  const change = useCallback(
    (patch: Record<string, boolean | number>) => command.current?.(patch),
    [],
  );
  return { data, error, pending, change };
}
export type LearningController = ReturnType<typeof useLiveLearning>;
export function LiveLearningPanel({
  learning,
  lab = false,
}: {
  learning: LearningController;
  lab?: boolean;
}) {
  const { data, error, pending, change } = learning;
  return (
    <section
      className="live-learning flow-card"
      aria-label="Live reinforcement learning"
    >
      <div className="learning-heading">
        <div>
          <span className="flow-kicker">CONTINUOUS LEARNING / LOCAL CPU</span>
          <h2>
            Live RL{" "}
            <span className={data?.enabled ? "learning-on" : ""}>
              {!data ? "UNKNOWN" : data.enabled ? "ON" : "OFF"}
            </span>
          </h2>
        </div>
        <div className="learning-controls">
          <label>
            <input
              type="checkbox"
              role="switch"
              aria-label="Enable live RL"
              checked={data?.enabled ?? false}
              disabled={!data || pending}
              onChange={(e) => change({ enabled: e.target.checked })}
            />
            Learn while running
          </label>
          <label>
            Every
            <select
              aria-label="Live RL interval"
              value={data?.interval_seconds ?? 10}
              disabled={!data || pending}
              onChange={(e) =>
                change({ interval_seconds: Number(e.target.value) })
              }
            >
              {[5, 10, 30, 60].map((x) => (
                <option key={x} value={x}>
                  {x}s
                </option>
              ))}
            </select>
          </label>
          <button
            className="flow-toggle"
            disabled={!data || pending}
            onClick={() => change({ reset: true, enabled: false })}
          >
            Reset weights
          </button>
        </div>
      </div>
      <LearningActivity data={data} error={error} pending={pending} />
      <p className="flow-footnote">
        Learns the live gate’s output head; hidden layers stay frozen. Pause
        retains weights; reset or feed changes restore the demo checkpoint.{" "}
        {lab ? "" : <Link href="/training">Watch weights change ↗</Link>}
      </p>
      <details className="feature-values">
        <summary>What is the reward?</summary>
        <p>
          A sampled expert’s signal × the observed mid-price change after 5
          seconds, clipped to ±5 bps. One REINFORCE step per fresh example; no
          fills, fees or P&amp;L. The sampled action teaches the gate; synthetic
          quotes still use its smoothed expert blend and deterministic risk
          checks. Updates stay in memory on this server.
        </p>
        <p>
          Learning begins after a fresh 30-frame window and pauses across stale
          feeds or gaps. This is a small contextual-bandit RL example, not
          evidence of trading performance.
        </p>
      </details>
    </section>
  );
}
export function TerminalLearning() {
  const learning = useLiveLearning();
  return <LiveLearningPanel learning={learning} />;
}
