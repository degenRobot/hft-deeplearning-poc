import type { ChangeEvent } from "react";
import { PRESETS, presetMatches, type PresetId } from "../lib/config";
import type { AppConfig, ResetResponse } from "../lib/types";

interface SettingsPanelProps {
  config: AppConfig;
  dirty: boolean;
  invalid: boolean;
  errors: string[];
  applying: boolean;
  resetting: boolean;
  error: string;
  resetError: string;
  resetResult: ResetResponse | null;
  onChange: (key: keyof AppConfig, value: string | number) => void;
  onPreset: (id: PresetId) => void;
  onApply: () => void;
  onRevert: () => void;
  onReset: () => void;
}
type FieldKind = "text" | "number" | "select";
type Field = {
  key: keyof AppConfig;
  label: string;
  kind: FieldKind;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  help?: string;
  options?: readonly [string, string][];
};
const text = (key: keyof AppConfig, label: string, help?: string): Field => ({
  key,
  label,
  kind: "text",
  help,
});
const numeric = (
  key: keyof AppConfig,
  label: string,
  unit: string,
  min: number,
  max: number,
  step: number,
  help?: string,
): Field => ({ key, label, kind: "number", unit, min, max, step, help });
const select = (
  key: keyof AppConfig,
  label: string,
  options: readonly [string, string][],
  help?: string,
): Field => ({ key, label, kind: "select", options, help });
const fields = [
  select(
    "source",
    "Feed source",
    [
      ["replay", "Replay fixture"],
      ["binance", "Binance public"],
    ],
    "Replay is deterministic and available offline.",
  ),
  text("symbol", "Symbol", "Venue symbol, for example BTCUSDT."),
  select("gate_mode", "Gate mode", [
    ["neural", "Neural model"],
    ["uniform", "Uniform baseline"],
    ["static", "Static baseline"],
  ]),
  numeric(
    "gate_interval_ms",
    "Gate interval",
    "ms",
    100,
    60000,
    100,
    "100–60,000 ms.",
  ),
  numeric(
    "higher_level_influence",
    "Higher-level influence",
    "0–1",
    0,
    1,
    0.05,
  ),
  numeric("expert_strength", "Expert strength", "×", 0, 5, 0.05),
  numeric(
    "flow_window_trades",
    "Flow window",
    "trades",
    4,
    512,
    1,
    "4–512 recent trades.",
  ),
  numeric("base_spread_bps", "Base spread", "bps", 0.01, 1000, 0.5),
  numeric("max_inventory", "Max inventory", "BTC", 0.001, 1000000, 0.001),
];

function RuntimeField({
  field,
  config,
  busy,
  onChange,
}: {
  field: Field;
  config: AppConfig;
  busy: boolean;
  onChange: SettingsPanelProps["onChange"];
}) {
  const id = `runtime-${field.key}`;
  const helpId = `${id}-help`;
  const handleChange = (
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) =>
    onChange(
      field.key,
      field.kind === "number"
        ? Number(event.target.value)
        : field.key === "symbol"
          ? event.target.value.toUpperCase()
          : event.target.value,
    );
  const describedBy = field.help ? helpId : undefined;
  const input =
    field.kind === "select" ? (
      <select
        id={id}
        value={String(config[field.key])}
        onChange={handleChange}
        aria-describedby={describedBy}
      >
        {field.options?.map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
    ) : (
      <span className={field.unit ? "input-suffix" : undefined}>
        <input
          id={id}
          type={field.kind === "number" ? "number" : "text"}
          value={config[field.key]}
          onChange={handleChange}
          disabled={busy}
          min={field.min}
          max={field.max}
          step={field.step}
          inputMode={field.kind === "number" ? "decimal" : "text"}
          aria-describedby={describedBy}
        />
        {field.unit && <em>{field.unit}</em>}
      </span>
    );
  return (
    <label htmlFor={id}>
      {field.label}
      {input}
      {field.help && <small id={helpId}>{field.help}</small>}
    </label>
  );
}

export function SettingsPanel({
  config,
  dirty,
  invalid,
  errors,
  applying,
  resetting,
  error,
  resetError,
  resetResult,
  onChange,
  onPreset,
  onApply,
  onRevert,
  onReset,
}: SettingsPanelProps) {
  const busy = applying || resetting;
  return (
    <details className="settings" aria-busy={busy}>
      <summary>
        Runtime settings <span>draft, apply, and reset the simulation</span>
      </summary>
      <div className="settings-body">
        <fieldset className="preset-fieldset" disabled={busy}>
          <legend>Draft presets</legend>
          <p className="form-help">
            Choose a starting point. Presets edit strategy fields only; source,
            symbol, spread, and inventory stay unchanged until Apply.
          </p>
          <div className="preset-grid">
            {PRESETS.map((preset) => {
              const selected = presetMatches(config, preset);
              return (
                <button
                  type="button"
                  key={preset.id}
                  className={`preset ${selected ? "selected" : ""}`}
                  aria-pressed={selected}
                  onClick={() => onPreset(preset.id)}
                >
                  <strong>{preset.label}</strong>
                  <span>{preset.description}</span>
                </button>
              );
            })}
          </div>
        </fieldset>
        <fieldset className="settings-grid" disabled={busy}>
          <legend>Parameters</legend>
          {fields.map((field) => (
            <RuntimeField
              key={field.key}
              field={field}
              config={config}
              busy={busy}
              onChange={onChange}
            />
          ))}
        </fieldset>
        <div className="settings-actions">
          <div
            className={`draft-state ${dirty ? "unsaved" : "saved"}`}
            role="status"
          >
            <span className="state-dot" />
            {dirty ? "Unsaved draft changes" : "Draft matches applied config"}
          </div>
          {(invalid || error) && (
            <span className="error-text" role="alert">
              {invalid ? errors[0] : error}
            </span>
          )}
          <button
            type="button"
            className="button ghost"
            onClick={onRevert}
            disabled={!dirty || busy}
          >
            Revert draft
          </button>
          <button
            type="button"
            className="button primary"
            onClick={onApply}
            disabled={!dirty || invalid || busy}
          >
            {applying ? "Applying…" : "Apply settings"}
          </button>
        </div>
        <div className="reset-row">
          <div>
            <strong>Reset simulation run</strong>
            <small>
              Restarts the applied config and clears this client’s live view.
              Unsaved draft changes remain local; this is not an exchange
              action.
            </small>
            {resetResult && (
              <span className="reset-success" role="status">
                New run {resetResult.run_id} · feed generation{" "}
                {resetResult.feed_generation}
              </span>
            )}
            {resetError && (
              <span className="error-text" role="alert">
                {resetError}
              </span>
            )}
          </div>
          <button
            type="button"
            className="button reset-button"
            onClick={onReset}
            disabled={busy}
          >
            {resetting ? "Resetting…" : "Reset simulation"}
          </button>
        </div>
      </div>
    </details>
  );
}
