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
  type ResetResponse,
} from "../lib/types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(
  /\/$/,
  "",
);
export const SNAPSHOT_TIMEOUT_MS = 2500;

const errorText = (cause: unknown, fallback: string) =>
  cause instanceof Error ? cause.message : fallback;

const request = async (path: string, init: RequestInit, label: string) => {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) throw new Error(`${label} ${response.status}`);
  return response.json();
};

export function useMarketGate() {
  const [state, setState] = useState(EMPTY_STATE);
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
  const runRef = useRef<ResetResponse | null>(null);
  const socketEpochRef = useRef(0);
  const mutatingRef = useRef(false);
  const configRequestGenerationRef = useRef(0);
  const configMutationGenerationRef = useRef(0);
  const configAbortRef = useRef<AbortController | null>(null);
  const draftConfigRef = useRef<AppConfig>(DEFAULT_CONFIG);
  const appliedConfigRef = useRef<AppConfig>(DEFAULT_CONFIG);

  const clearLiveState = useCallback(() => {
    setState(EMPTY_STATE);
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

  const syncConfig = useCallback(async (mutationError = "") => {
    if (mutatingRef.current) return;
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
      setSettingsError(mutationError);
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
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    const epoch = ++socketEpochRef.current;
    const current = () => socketEpochRef.current === epoch && !closed;
    runRef.current = null;
    const armWatchdog = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => {
        if (!current()) return;
        if (mutatingRef.current) {
          armWatchdog();
          return;
        }
        clearLiveState();
        setStatus("error");
        setLastError("Backend snapshots stopped arriving");
      }, SNAPSHOT_TIMEOUT_MS);
    };
    // The socket is an external subscription; clear prior values before it is opened.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStatus("connecting");
    setLastError("");
    clearLiveState();
    const socket = new WebSocket(websocketUrl(API_URL));
    socket.onopen = () => {
      if (!current()) return;
      setStatus("connecting");
      setLastError("");
      void syncConfig();
      armWatchdog();
    };
    socket.onmessage = (event) => {
      if (!current() || mutatingRef.current) return;
      const parsed =
        typeof event.data === "string" ? parseStateMessage(event.data) : null;
      if (!parsed) {
        clearLiveState();
        setStatus("error");
        setLastError("Received an invalid state snapshot");
        return;
      }
      const previous = runRef.current;
      const received = parsed.health;
      if (
        previous &&
        (received.feed_generation < previous.feed_generation ||
          (received.feed_generation === previous.feed_generation &&
            received.run_id !== previous.run_id))
      )
        return;
      armWatchdog();
      setLastError("");
      if (previous && received.run_id !== previous.run_id) {
        void syncConfig();
      }
      runRef.current = {
        run_id: received.run_id,
        feed_generation: received.feed_generation,
      };
      setState(parsed);
      // Transport is connected once valid snapshots arrive. Market readiness
      // independently gates the dashboard, including stale or restarting feeds.
      setStatus("connected");
    };
    socket.onerror = () => {
      if (!current()) return;
      clearLiveState();
      setStatus("error");
      setLastError(`Could not reach ${API_URL}`);
    };
    socket.onclose = () => {
      if (!current()) return;
      closed = true;
      clearTimeout(watchdog);
      clearLiveState();
      setStatus("disconnected");
      reconnectTimer = setTimeout(() => {
        if (socketEpochRef.current === epoch)
          setReconnectToken((token) => token + 1);
      }, 3500);
    };
    return () => {
      if (current()) socketEpochRef.current += 1;
      clearTimeout(watchdog);
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
    if (mutatingRef.current) return;
    invalidateConfigReads();
    setDraft(appliedConfigRef.current);
    clearTransient();
    void syncConfig();
  };

  const applyConfig = async () => {
    if (mutatingRef.current) return;
    const submitted = draftConfigRef.current;
    const errors = validateConfig(submitted);
    if (errors.length) {
      setSettingsError(errors[0]);
      return;
    }
    clearTransient();
    mutatingRef.current = true;
    const epoch = socketEpochRef.current;
    configMutationGenerationRef.current += 1;
    setApplying(true);
    invalidateConfigReads();
    clearLiveState();
    setStatus("connecting");
    let mutationError = "";
    try {
      const payload = await request(
        "/config",
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(submitted),
        },
        "Config update returned",
      );
      const receipt = parseResetResponse(payload);
      if (!receipt)
        throw new Error("Config update returned an invalid run receipt");
      const canonical = normalizeConfig(payload, submitted);
      // The backend starts a new engine for every applied config. Clear the
      // previous run before exposing its new labels or cadence in the UI.
      clearLiveState();
      setStatus("connecting");
      if (socketEpochRef.current === epoch) runRef.current = receipt;
      else reconnect();
      appliedConfigRef.current = canonical;
      setAppliedConfig(canonical);
      if (configsEqual(draftConfigRef.current, submitted)) setDraft(canonical);
    } catch (error: unknown) {
      mutationError = errorText(error, "Could not update config");
      setSettingsError(mutationError);
      reconnect();
    } finally {
      mutatingRef.current = false;
      setApplying(false);
      await syncConfig(mutationError);
    }
  };

  const resetRun = async () => {
    if (mutatingRef.current) return;
    mutatingRef.current = true;
    const epoch = socketEpochRef.current;
    configMutationGenerationRef.current += 1;
    invalidateConfigReads();
    setResetting(true);
    clearTransient();
    clearLiveState();
    setStatus("connecting");
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
      if (socketEpochRef.current === epoch) runRef.current = result;
      else reconnect();
    } catch (error: unknown) {
      setResetError(errorText(error, "Could not reset run"));
      reconnect();
    } finally {
      mutatingRef.current = false;
      setResetting(false);
      await syncConfig();
    }
  };

  const reconnect = () => {
    socketEpochRef.current += 1;
    setReconnectToken((token) => token + 1);
  };
  const configErrors = validateConfig(draftConfig);
  return {
    apiUrl: API_URL,
    state,
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
