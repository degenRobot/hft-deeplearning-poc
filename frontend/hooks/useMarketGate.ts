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
  shouldSeedDraft,
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

export function useMarketGate() {
  const [state, setState] = useState(EMPTY_STATE);
  const [hasSnapshot, setHasSnapshot] = useState(false);
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
  const [nextRefresh, setNextRefresh] = useState(0);
  const [reconnectToken, setReconnectToken] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const runIdRef = useRef<string | null>(null);
  const revisionRef = useRef<number | null>(null);
  const configRequestGenerationRef = useRef(0);
  const configMutationGenerationRef = useRef(0);
  const configAbortRef = useRef<AbortController | null>(null);
  const draftWasEditedRef = useRef(false);

  const clearLiveState = useCallback(() => {
    setState(EMPTY_STATE);
    setHasSnapshot(false);
    setHistory([]);
    setNextRefresh(0);
    runIdRef.current = null;
    revisionRef.current = null;
  }, []);

  const invalidateConfigReads = useCallback(() => {
    configAbortRef.current?.abort();
    configAbortRef.current = null;
    configRequestGenerationRef.current += 1;
  }, []);

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
      const response = await fetch(`${API_URL}/config`, {
        signal: controller.signal,
      });
      if (!response.ok)
        throw new Error(`Config request returned ${response.status}`);
      const canonical = normalizeConfig(await response.json());
      if (!isCurrent()) return;
      setAppliedConfig(canonical);
      if (shouldSeedDraft(draftWasEditedRef.current)) setDraftConfig(canonical);
      setSettingsError("");
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (!isCurrent()) return;
      setSettingsError(
        error instanceof Error ? error.message : "Could not load config",
      );
    } finally {
      if (isCurrent()) {
        setConfigLoading(false);
        configAbortRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    // Config state is synchronized by the async request, not this effect body.
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
    wsRef.current = socket;
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
      setHasSnapshot(parsed.health.ready);
      setNextRefresh(parsed.health.ready ? parsed.gate.next_refresh_ms : 0);
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
      wsRef.current = null;
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

  const updateDraft = useCallback(
    (key: keyof AppConfig, value: string | number) => {
      draftWasEditedRef.current = true;
      setDraftConfig((current) => ({ ...current, [key]: value }) as AppConfig);
      setSettingsError("");
      setResetResult(null);
      setResetError("");
    },
    [],
  );

  const selectPreset = useCallback((id: PresetId) => {
    draftWasEditedRef.current = true;
    setDraftConfig((current) => ({ ...current, ...presetById(id).patch }));
    setSettingsError("");
    setResetResult(null);
    setResetError("");
  }, []);

  const revertDraft = useCallback(() => {
    invalidateConfigReads();
    draftWasEditedRef.current = false;
    setDraftConfig(appliedConfig);
    setSettingsError("");
    setResetResult(null);
    setResetError("");
    void syncConfig();
  }, [appliedConfig, invalidateConfigReads, syncConfig]);

  const applyConfig = useCallback(async () => {
    const errors = validateConfig(draftConfig);
    if (errors.length) {
      setSettingsError(errors[0]);
      return;
    }
    setSettingsError("");
    setResetResult(null);
    setResetError("");
    setApplying(true);
    invalidateConfigReads();
    try {
      const response = await fetch(`${API_URL}/config`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draftConfig),
      });
      if (!response.ok)
        throw new Error(`Config update returned ${response.status}`);
      const canonical = normalizeConfig(
        await response.json().catch(() => draftConfig),
        draftConfig,
      );
      // The backend starts a new engine for every applied config. Clear the
      // previous run before exposing its new labels or cadence in the UI.
      clearLiveState();
      setStatus("connecting");
      configMutationGenerationRef.current += 1;
      setAppliedConfig(canonical);
      setDraftConfig(canonical);
      draftWasEditedRef.current = false;
    } catch (error: unknown) {
      setSettingsError(
        error instanceof Error ? error.message : "Could not update config",
      );
    } finally {
      setApplying(false);
    }
  }, [clearLiveState, draftConfig, invalidateConfigReads]);

  const resetRun = useCallback(async () => {
    setResetting(true);
    setResetError("");
    setResetResult(null);
    clearLiveState();
    try {
      const response = await fetch(`${API_URL}/reset`, { method: "POST" });
      if (!response.ok) throw new Error(`Reset returned ${response.status}`);
      const result = parseResetResponse(await response.json());
      if (!result) throw new Error("Reset returned an invalid run receipt");
      // A snapshot from the prior run can arrive while the reset request is in
      // flight. Clear once more after the backend acknowledges the new run.
      clearLiveState();
      setStatus("connecting");
      setResetResult(result);
      await syncConfig();
    } catch (error: unknown) {
      setResetError(
        error instanceof Error ? error.message : "Could not reset run",
      );
    } finally {
      setResetting(false);
    }
  }, [clearLiveState, syncConfig]);

  const reconnect = useCallback(
    () => setReconnectToken((token) => token + 1),
    [],
  );
  const configErrors = validateConfig(draftConfig);
  return {
    apiUrl: API_URL,
    state,
    history,
    hasSnapshot,
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
    nextRefresh,
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
