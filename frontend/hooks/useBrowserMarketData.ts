"use client";
import { useEffect, useState } from "react";
import { API_DEFAULT, websocketUrl } from "../lib/connection";
import { parseStateMessage } from "../lib/normalize";
import {
  collectMarketCandles,
  type BrowserCapture,
} from "../lib/browserMarketData";

const API = process.env.NEXT_PUBLIC_API_URL || API_DEFAULT;

/** A read-only subscription. No feed changes, training calls or credential storage. */
export function useBrowserMarketData() {
  const [capture, setCapture] = useState<BrowserCapture | null>(null);
  const [status, setStatus] = useState("Connecting to the terminal feed…");
  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;
    let watchdog: ReturnType<typeof setTimeout>;
    const connect = () => {
      if (disposed) return;
      const current = new WebSocket(websocketUrl(API));
      socket = current;
      const active = () => !disposed && socket === current;
      const arm = () => {
        clearTimeout(watchdog);
        watchdog = setTimeout(() => {
          if (!active()) return;
          setCapture(null);
          setStatus("Feed paused · waiting for fresh candles");
          current.close();
        }, 5000);
      };
      arm();
      current.onmessage = (event) => {
        if (!active()) return;
        const snapshot =
          typeof event.data === "string" ? parseStateMessage(event.data) : null;
        if (!snapshot) {
          setCapture(null);
          setStatus("Feed unavailable · invalid snapshot");
          return;
        }
        arm();
        setCapture((previous) => collectMarketCandles(previous, snapshot));
        setStatus(
          snapshot.source !== "binance"
            ? "Select Binance in Live Terminal to collect live data"
            : !snapshot.health.ready
              ? "Waiting for fresh live market data"
              : "Collecting live candles",
        );
      };
      current.onerror = () => {
        if (!active()) return;
        setCapture(null);
        setStatus("Live feed unavailable · reconnecting");
        current.close();
      };
      current.onclose = () => {
        if (!active()) return;
        clearTimeout(watchdog);
        setCapture(null);
        setStatus("Feed disconnected · reconnecting");
        retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(retry);
      clearTimeout(watchdog);
      socket?.close();
    };
  }, []);
  return { capture, status };
}
