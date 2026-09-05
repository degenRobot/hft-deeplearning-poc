"use client";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { API_DEFAULT } from "../lib/connection";
import { TrainingCost } from "./TrainingProgress";
import {
  DEFAULT_TRAINING_OPTIONS,
  parseTrainingSettings,
  trainingCanStop,
  trainingParameterCount,
  validTrainingOptions,
  type LiveTraining,
  type TrainingOptions,
  type TrainingSettings,
} from "../lib/liveTraining";
const API = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(/\/$/, "");
const presets = [
  { label: "Small · 64 / 32", first: 64, second: 32 },
  { label: "Medium · 256 / 128", first: 256, second: 128 },
  { label: "Large · 1024 / 512", first: 1024, second: 512 },
];
function CredentialForm({
  disabled,
  onSaved,
}: {
  disabled: boolean;
  onSaved: (modal: TrainingSettings["modal"]) => void;
}) {
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      request.current?.abort();
      request.current = null;
    },
    [],
  );
  async function save(event: FormEvent) {
    event.preventDefault();
    if (
      request.current ||
      saving ||
      disabled ||
      !tokenId.trim() ||
      !tokenSecret.trim()
    )
      return;
    const body = JSON.stringify({
      token_id: tokenId.trim(),
      token_secret: tokenSecret.trim(),
    });
    setTokenId("");
    setTokenSecret("");
    setSaving(true);
    setError("");
    const controller = new AbortController();
    request.current = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const result = await fetch(`${API}/training/credentials`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: controller.signal,
      });
      if (!result.ok) throw new Error("save failed");
      const value: unknown = await result.json();
      const modal =
        value && typeof value === "object" && "modal" in value
          ? value.modal
          : null;
      if (
        !modal ||
        typeof modal !== "object" ||
        !("configured" in modal) ||
        typeof modal.configured !== "boolean" ||
        !("available" in modal) ||
        typeof modal.available !== "boolean"
      )
        throw new Error("invalid status");
      if (!controller.signal.aborted)
        onSaved({ configured: modal.configured, available: modal.available });
    } catch {
      if (request.current === controller)
        setError(
          "Could not confirm the keys were saved. Check the backend connection and retry. Enter the values again; the form has been cleared.",
        );
    } finally {
      clearTimeout(timeout);
      if (request.current === controller) {
        request.current = null;
        setSaving(false);
      }
    }
  }
  return (
    <form className="training-credentials" onSubmit={save} autoComplete="off">
      <p>
        Saved locally in <code>.env</code> with private permissions; never
        returned to this page.
      </p>
      <div className="training-fields">
        <label>
          Modal token ID
          <input
            name="modal-token-id"
            type="password"
            autoComplete="new-password"
            spellCheck={false}
            value={tokenId}
            disabled={disabled || saving}
            onChange={(e) => setTokenId(e.target.value)}
          />
        </label>
        <label>
          Modal token secret
          <input
            name="modal-token-secret"
            type="password"
            autoComplete="new-password"
            spellCheck={false}
            value={tokenSecret}
            disabled={disabled || saving}
            onChange={(e) => setTokenSecret(e.target.value)}
          />
        </label>
      </div>
      <button
        className="button ghost"
        type="submit"
        disabled={disabled || saving || !tokenId.trim() || !tokenSecret.trim()}
      >
        {saving ? "Saving keys…" : "Save keys to .env"}
      </button>
      {error && (
        <p role="alert" className="training-run-error">
          {error}
        </p>
      )}
    </form>
  );
}
export function TrainingControls({
  data,
  loading,
  pending,
  start,
  stop,
}: {
  data: LiveTraining | null;
  loading: boolean;
  pending: boolean;
  start: (options: TrainingOptions) => void | Promise<void> | undefined;
  stop: () => void | Promise<void> | undefined;
}) {
  const [settings, setSettings] = useState<TrainingSettings | null>(null);
  const [settingsError, setSettingsError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [options, setOptions] = useState<TrainingOptions>(
    DEFAULT_TRAINING_OPTIONS,
  );
  const [customPreset, setCustomPreset] = useState(false);
  const [replaceKeys, setReplaceKeys] = useState(false);
  const [saved, setSaved] = useState(false);
  const busy = pending || data?.status === "running";
  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    let disposed = false;
    async function load() {
      try {
        const result = await fetch(`${API}/training/settings`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!result.ok) throw new Error("unavailable");
        const parsed = parseTrainingSettings(await result.json());
        if (!parsed) throw new Error("invalid settings");
        if (!disposed) {
          setSettings(parsed);
          setOptions(parsed.defaults);
          setSettingsError("");
        }
      } catch {
        if (!disposed)
          setSettingsError(
            "Training settings could not be loaded. Check the backend connection and retry.",
          );
      } finally {
        clearTimeout(timeout);
      }
    }
    void load();
    return () => {
      disposed = true;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [attempt]);
  const limits = settings?.limits ?? {
    hidden_min: 8,
    hidden_max: 1024,
    epochs_max: 50,
  };
  const valid = validTrainingOptions(options, limits);
  const preset = presets.findIndex(
    (p) => p.first === options.hidden_1 && p.second === options.hidden_2,
  );
  const modalReady = Boolean(
    settings?.modal.configured && settings.modal.available,
  );
  const blocked =
    loading ||
    busy ||
    !data ||
    !settings ||
    !valid ||
    (options.backend === "modal" && !modalReady);
  const changeNumber = (
    key: "hidden_1" | "hidden_2" | "epochs" | "learning_rate",
    value: string,
  ) =>
    setOptions((previous) => ({
      ...previous,
      [key]: value === "" ? NaN : Number(value),
    }));
  const numberValue = (value: number) => (Number.isFinite(value) ? value : "");
  return (
    <section className="training-setup" aria-label="Configure training">
      <div className="training-setup-heading">
        <div>
          <span className="flow-kicker">02 / TRAIN THE MODEL</span>
          <h2>Set up your run.</h2>
        </div>
        <span className="training-setup-caption">
          Selected data · your model
        </span>
      </div>
      <fieldset
        disabled={busy || !settings}
        className="training-fields training-options"
      >
        <legend className="sr-only">Training configuration</legend>
        <label>
          Train on
          <select
            value={options.backend}
            onChange={(e) =>
              setOptions((previous) => ({
                ...previous,
                backend: e.target.value as TrainingOptions["backend"],
              }))
            }
          >
            <option value="local">Local machine</option>
            <option value="modal">Modal cloud</option>
          </select>
        </label>
        <label>
          Model preset
          <select
            value={customPreset || preset < 0 ? "custom" : String(preset)}
            onChange={(e) => {
              setCustomPreset(e.target.value === "custom");
              const next = presets[Number(e.target.value)];
              if (next)
                setOptions((previous) => ({
                  ...previous,
                  hidden_1: next.first,
                  hidden_2: next.second,
                }));
            }}
          >
            <option value="custom">Custom widths</option>
            {presets.map((p, index) => (
              <option key={p.label} value={index}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <details style={{ gridColumn: "1 / -1" }}>
          <summary>Advanced settings</summary>
          <div className="training-fields">
            <label>
              Hidden layer 1
              <input
                type="number"
                min={limits.hidden_min}
                max={limits.hidden_max}
                step="1"
                value={numberValue(options.hidden_1)}
                onChange={(e) => changeNumber("hidden_1", e.target.value)}
              />
            </label>
            <label>
              Hidden layer 2
              <input
                type="number"
                min={limits.hidden_min}
                max={limits.hidden_max}
                step="1"
                value={numberValue(options.hidden_2)}
                onChange={(e) => changeNumber("hidden_2", e.target.value)}
              />
            </label>
            <label>
              Warmup epochs
              <input
                type="number"
                min="1"
                max={limits.epochs_max}
                step="1"
                value={numberValue(options.epochs)}
                onChange={(e) => changeNumber("epochs", e.target.value)}
              />
            </label>
            <label>
              Learning rate
              <input
                type="number"
                min="0.00001"
                max="0.01"
                step="0.00001"
                value={numberValue(options.learning_rate)}
                onChange={(e) => changeNumber("learning_rate", e.target.value)}
              />
            </label>
          </div>
        </details>
      </fieldset>
      <div className="training-model-summary">
        <span>
          300 inputs <b>→</b> {numberValue(options.hidden_1) || "?"} ReLU{" "}
          <b>→</b> {numberValue(options.hidden_2) || "?"} ReLU <b>→</b> 3
          experts
        </span>
        <strong>
          {valid
            ? trainingParameterCount(
                options.hidden_1,
                options.hidden_2,
              ).toLocaleString("en-US")
            : "—"}{" "}
          parameters
        </strong>
      </div>
      {!valid && (
        <p className="training-run-error" role="alert">
          Use whole layer widths from {limits.hidden_min} to {limits.hidden_max}
          , 1–{limits.epochs_max} epochs, and a learning rate from 0.00001 to
          0.01.
        </p>
      )}
      {options.backend === "modal" && (
        <div className="training-modal-setup">
          <div className="training-modal-title">
            <strong>Modal cloud</strong>
            <span className={modalReady ? "teal" : ""}>
              {saved
                ? "Keys saved"
                : settings?.modal.configured
                  ? "Saved keys configured"
                  : "Keys needed"}
            </span>
          </div>
          <p>
            Uses your Modal account. Charges apply beyond available credits.
          </p>
          <details className="training-modal-help">
            <summary>Compute, cost estimate and credits</summary>
            <p>
              <a href="https://modal.com/" target="_blank" rel="noreferrer">
                Modal ↗
              </a>{" "}
              runs the training in a cloud container. Its{" "}
              <a
                href="https://modal.com/pricing"
                target="_blank"
                rel="noreferrer"
              >
                Starter plan includes $30/month in free compute credits ↗
              </a>{" "}
              to experiment. Usage beyond your available credits is billed by
              Modal.
            </p>
            <p className="training-resource-note">
              This run requests {settings?.resources.cpu ?? 2} CPUs and{" "}
              {settings?.resources.memory_gib ?? 2} GiB memory, with a{" "}
              {settings?.resources.timeout_seconds ?? 600}-second execution
              limit. Image build and startup time are additional; this is not a
              billing cap.
            </p>
            <TrainingCost pricing={settings?.pricing} />
          </details>
          {!settings?.modal.available && (
            <p role="status">
              The backend needs the Modal Python package before cloud training
              is available.
            </p>
          )}
          <details className="training-modal-help">
            <summary>How to get Modal API keys</summary>
            <ol>
              <li>
                Create an account at{" "}
                <a href="https://modal.com/" target="_blank" rel="noreferrer">
                  modal.com
                </a>
                .
              </li>
              <li>
                Open Settings →{" "}
                <a
                  href="https://modal.com/settings/tokens"
                  target="_blank"
                  rel="noreferrer"
                >
                  API Tokens
                </a>{" "}
                and create a token.
              </li>
              <li>
                Paste its token ID and token secret below, or set{" "}
                <code>MODAL_TOKEN_ID</code> and <code>MODAL_TOKEN_SECRET</code>{" "}
                in the project root <code>.env</code>.
              </li>
            </ol>
          </details>
          {settings?.modal.configured && (
            <button
              type="button"
              className="training-disclosure"
              disabled={busy}
              aria-expanded={replaceKeys}
              onClick={() => setReplaceKeys((v) => !v)}
            >
              {replaceKeys ? "Cancel replacing keys" : "Replace saved keys"}
            </button>
          )}
          {(!settings?.modal.configured || replaceKeys) && (
            <CredentialForm
              disabled={busy}
              onSaved={(modal) => {
                setSettings((previous) =>
                  previous ? { ...previous, modal } : previous,
                );
                setReplaceKeys(false);
                setSaved(true);
              }}
            />
          )}
        </div>
      )}
      {settingsError && (
        <p className="training-run-error" role="alert">
          {settingsError}{" "}
          <button
            type="button"
            className="training-disclosure"
            onClick={() => {
              setSettingsError("");
              setAttempt((v) => v + 1);
            }}
          >
            Retry settings
          </button>
        </p>
      )}
      {!settings && !settingsError && (
        <p className="flow-footnote">Loading training settings…</p>
      )}
      <div className="training-launch">
        <div className="training-controls">
          <button
            className="button primary"
            disabled={blocked}
            onClick={() => void start({ ...options })}
          >
            {pending
              ? "Updating run…"
              : options.backend === "modal"
                ? "Train on Modal"
                : "Train locally"}
          </button>
          <button
            className="button ghost"
            disabled={pending || !trainingCanStop(data)}
            onClick={() => void stop()}
          >
            Stop run
          </button>
        </div>
        <p>
          {busy
            ? "Settings unlock when this run ends."
            : options.backend === "modal"
              ? "Runs in your Modal account."
              : "Runs on this machine."}
        </p>
      </div>
    </section>
  );
}
