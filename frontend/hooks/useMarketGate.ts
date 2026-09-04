"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_DEFAULT, websocketUrl } from "../lib/connection";
import {
  configsEqual,
  normalizeConfig,
  presetById,
  validateConfig,
  type PresetId,
} from "../lib/config";
import { parseResetResponse } from "../lib/reset";
import { parseStateMessage } from "../lib/normalize";
import {
  isCurrentConfigResponse,
  mergeConfigResponse,
  type ConfigRequestToken,
} from "../lib/configSync";
import {
  DEFAULT_CONFIG,
  EMPTY_STATE,
  type AppConfig,
  type ConnectionStatus,
  type MarketGateState,
  type ResetResponse,
} from "../lib/types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(
  /\/$/,
  "",
);

const errorText = (cause: unknown, fallback: string) =>
  cause instanceof Error ? cause.message : fallback;

const request = async (
  path: string,
  init: RequestInit,
  label: string,
  jsonFallback?: unknown,
) => {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) throw new Error(`${label} ${response.status}`);
  return jsonFallback === undefined
    ? response.json()
    : response.json().catch(() => jsonFallback);
};

export function useMarketGate() {
  const [state, setState] = useState(EMPTY_STATE);
  const [history, setHistory] = useState<MarketGateState[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [draftConfig, setDraftConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [appliedConfig, setAppliedConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [configLoading, setConfigLoading] = useState(true);
  const [settingsError, setSettingsError] = useState("");
  const [applying, setApplying] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState("");
  const [resetResult, setResetResult] = useState<ResetResponse | null>(null);
  const [lastError, setLastError] = useState("");
  const [reconnectToken, setReconnectToken] = useState(0);
  const runIdRef = useRef<string | null>(null);
  const revisionRef = useRef<number | null>(null);
  const configRequestGenerationRef = useRef(0);
  const configMutationGenerationRef = useRef(0);
  const configAbortRef = useRef<AbortController | null>(null);
  const draftConfigRef = useRef<AppConfig>(DEFAULT_CONFIG);
  const appliedConfigRef = useRef<AppConfig>(DEFAULT_CONFIG);

  const clearLiveState = useCallback(() => {
    setState(EMPTY_STATE);
    setHistory([]);
    runIdRef.current = null;
    revisionRef.current = null;
  }, []);

  const clearTransient = () => {
    setSettingsError("");
    setResetResult(null);
    setResetError("");
  };

  const setDraft = (next: AppConfig) => {
    draftConfigRef.current = next;
    setDraftConfig(next);
  };

  const invalidateConfigReads = () => {
    configAbortRef.current?.abort();
    configAbortRef.current = null;
    configRequestGenerationRef.current += 1;
  };

  const syncConfig = useCallback(async () => {
    configAbortRef.current?.abort();
    const controller = new AbortController();
    configAbortRef.current = controller;
    const token: ConfigRequestToken = {
      requestGeneration: ++configRequestGenerationRef.current,
      mutationGeneration: configMutationGenerationRef.current,
    };
    const isCurrent = () =>
      isCurrentConfigResponse(token, {
        requestGeneration: configRequestGenerationRef.current,
        mutationGeneration: configMutationGenerationRef.current,
      });
    try {
      const payload = await request(
        "/config",
        { signal: controller.signal },
        "Config request returned",
      );
      const canonical = normalizeConfig(payload);
      if (!isCurrent()) return;
      const merged = mergeConfigResponse(
        draftConfigRef.current,
        appliedConfigRef.current,
        canonical,
      );
      appliedConfigRef.current = merged.appliedConfig;
      setAppliedConfig(merged.appliedConfig);
      if (merged.draftConfig !== draftConfigRef.current) {
        draftConfigRef.current = merged.draftConfig;
        setDraftConfig(merged.draftConfig);
      }
      setSettingsError("");
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (!isCurrent()) return;
      setSettingsError(errorText(error, "Could not load config"));
    } finally {
      if (isCurrent()) {
        setConfigLoading(false);
        configAbortRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    // Config state is synchronized by the async request, not this effect body.
    void syncConfig();
    return () => configAbortRef.current?.abort();
  }, [syncConfig]);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let closedByEffect = false;
    // The socket is an external subscription; clear prior values before it is opened.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStatus("connecting");
    setLastError("");
    clearLiveState();
    const socket = new WebSocket(websocketUrl(API_URL));
    socket.onopen = () => {
      setStatus("connecting");
      setLastError("");
      void syncConfig();
    };
    socket.onmessage = (event) => {
      const parsed =
        typeof event.data === "string" ? parseStateMessage(event.data) : null;
      if (!parsed) {
        setLastError("Received an invalid state snapshot");
        return;
      }
      setLastError("");
      if (
        runIdRef.current !== null &&
        parsed.health.run_id !== runIdRef.current
      ) {
        setHistory([]);
        revisionRef.current = null;
        void syncConfig();
      }
      runIdRef.current = parsed.health.run_id;
      setState(parsed);
      if (!parsed.health.ready) {
        setStatus("connecting");
        return;
      }
      setStatus("connected");
      if (parsed.gate.revision !== revisionRef.current) {
        revisionRef.current = parsed.gate.revision;
        setHistory((items) => [...items, parsed].slice(-32));
      }
    };
    socket.onerror = () => {
      clearLiveState();
      setStatus("error");
      setLastError(`Could not reach ${API_URL}`);
    };
    socket.onclose = () => {
      if (closedByEffect) return;
      clearLiveState();
      setStatus("disconnected");
      reconnectTimer = setTimeout(
        () => setReconnectToken((token) => token + 1),
        3500,
      );
    };
    return () => {
      closedByEffect = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket.close();
    };
  }, [clearLiveState, reconnectToken, syncConfig]);

  const updateDraft = (key: keyof AppConfig, value: string | number) => {
    const next = {
      ...draftConfigRef.current,
      [key]: value,
    } as AppConfig;
    setDraft(next);
    clearTransient();
  };

  const selectPreset = (id: PresetId) => {
    const next = { ...draftConfigRef.current, ...presetById(id).patch };
    setDraft(next);
    clearTransient();
  };

  const revertDraft = () => {
    invalidateConfigReads();
    setDraft(appliedConfigRef.current);
    clearTransient();
    void syncConfig();
  };

  const applyConfig = async () => {
    const errors = validateConfig(draftConfig);
    if (errors.length) {
      setSettingsError(errors[0]);
      return;
    }
    clearTransient();
    setApplying(true);
    invalidateConfigReads();
    try {
      const payload = await request(
        "/config",
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(draftConfig),
        },
        "Config update returned",
        draftConfig,
      );
      const canonical = normalizeConfig(payload, draftConfig);
      // The backend starts a new engine for every applied config. Clear the
      // previous run before exposing its new labels or cadence in the UI.
      clearLiveState();
      setStatus("connecting");
      configMutationGenerationRef.current += 1;
      appliedConfigRef.current = canonical;
      setAppliedConfig(canonical);
      setDraft(canonical);
    } catch (error: unknown) {
      setSettingsError(errorText(error, "Could not update config"));
    } finally {
      setApplying(false);
    }
  };

  const resetRun = async () => {
    setResetting(true);
    clearTransient();
    clearLiveState();
    try {
      const payload = await request(
        "/reset",
        { method: "POST" },
        "Reset returned",
      );
      const result = parseResetResponse(payload);
      if (!result) throw new Error("Reset returned an invalid run receipt");
      // A snapshot from the prior run can arrive while the reset request is in
      // flight. Clear once more after the backend acknowledges the new run.
      clearLiveState();
      setStatus("connecting");
      setResetResult(result);
      await syncConfig();
    } catch (error: unknown) {
      setResetError(errorText(error, "Could not reset run"));
    } finally {
      setResetting(false);
    }
  };

  const reconnect = () => setReconnectToken((token) => token + 1);
  const configErrors = validateConfig(draftConfig);
  return {
    apiUrl: API_URL,
    state,
    history,
    hasSnapshot: state.health.ready,
    status,
    draftConfig,
    appliedConfig,
    configLoading,
    settingsError,
    applying,
    resetting,
    resetError,
    resetResult,
    lastError,
    nextRefresh: state.health.ready ? state.gate.next_refresh_ms : 0,
    dirty: !configsEqual(draftConfig, appliedConfig),
    configErrors,
    updateDraft,
    selectPreset,
    applyConfig,
    revertDraft,
    resetRun,
    reconnect,
  };
}
