"use client";

import { useEffect, useState } from "react";
import { API_DEFAULT } from "../lib/connection";
import {
  normalizeTrainingReceipt,
  type TrainingReceipt,
} from "../lib/training";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(
  /\/$/,
  "",
);

export function useTrainingReceipt() {
  const [receipt, setReceipt] = useState<TrainingReceipt | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadReceipt() {
      try {
        const response = await fetch(`${API_URL}/training`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Training receipt returned ${response.status}`);
        }
        const parsed = normalizeTrainingReceipt(await response.json());
        if (!parsed) throw new Error("Training receipt has an invalid shape");
        setReceipt(parsed);
        setError("");
      } catch (cause: unknown) {
        if (cause instanceof DOMException && cause.name === "AbortError")
          return;
        setReceipt(null);
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not load training receipt",
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    void loadReceipt();
    return () => controller.abort();
  }, []);

  return { receipt, loading, error };
}
