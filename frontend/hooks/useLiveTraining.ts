"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { API_DEFAULT } from "../lib/connection";
import {
  parseLiveTraining,
  trainingIsStale,
  type LiveTraining,
} from "../lib/liveTraining";
const API = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(/\/$/, "");
export function useLiveTraining() {
  const [data, setData] = useState<LiveTraining | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const command = useRef<((action: "start" | "stop") => Promise<void>) | null>(
    null,
  );
  useEffect(() => {
    let disposed = false;
    let epoch = 0;
    let busy = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout>;
    const request = async (action?: "start" | "stop") => {
      const version = ++epoch;
      controller?.abort();
      const current = new AbortController();
      controller = current;
      const timeout = setTimeout(() => {
        current.abort();
        if (!disposed && version === epoch) {
          setData(null);
          setError("Training telemetry timed out. Display cleared.");
          setLoading(false);
        }
      }, 5000);
      try {
        if (action) {
          const result = await fetch(`${API}/training/live/${action}`, {
            method: "POST",
            signal: current.signal,
          });
          if (!result.ok)
            throw new Error(`Training ${action} returned ${result.status}`);
        }
        if (disposed || version !== epoch || current.signal.aborted) return;
        const result = await fetch(`${API}/training/live`, {
          cache: "no-store",
          signal: current.signal,
        });
        if (!result.ok)
          throw new Error(`Training telemetry returned ${result.status}`);
        const parsed = parseLiveTraining(await result.json());
        if (disposed || version !== epoch || current.signal.aborted) return;
        if (!parsed) throw new Error("Training telemetry has an invalid shape");
        if (trainingIsStale(parsed, Date.now()))
          throw new Error(
            "Training telemetry is stale. Waiting for a fresh update.",
          );
        setData(parsed);
        setError("");
      } catch (cause) {
        if (disposed || version !== epoch) return;
        setData(null);
        setError(
          current.signal.aborted
            ? "Training telemetry timed out. Display cleared."
            : cause instanceof Error
              ? cause.message
              : "Training telemetry unavailable",
        );
      } finally {
        clearTimeout(timeout);
        if (!disposed && version === epoch) {
          setLoading(false);
          busy = false;
          setPending(false);
          timer = setTimeout(() => void request(), 500);
        }
      }
    };
    command.current = async (action) => {
      if (disposed || busy) return;
      busy = true;
      clearTimeout(timer);
      setPending(true);
      setData(null);
      setError("");
      await request(action);
    };
    void request();
    return () => {
      disposed = true;
      epoch++;
      controller?.abort();
      clearTimeout(timer);
      command.current = null;
    };
  }, []);
  const start = useCallback(() => command.current?.("start"), []);
  const stop = useCallback(() => command.current?.("stop"), []);
  return { data, error, loading, pending, start, stop };
}
