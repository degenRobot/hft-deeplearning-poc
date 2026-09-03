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
  return (
    <details className="settings">
      <summary>
        Runtime settings <span>draft, apply, and reset the simulation</span>
      </summary>
      <div className="settings-body">
        <fieldset className="preset-fieldset">
          <legend>Draft presets</legend>
          <p className="form-help">
            Choose a starting point. Presets edit strategy fields only; source,
            symbol, spread, and inventory stay unchanged until Apply.
          </p>
          <div className="preset-grid">
            {PRESETS.map((preset) => (
              <button
                type="button"
                key={preset.id}
                className={`preset ${presetMatches(config, preset) ? "selected" : ""}`}
                onClick={() => onPreset(preset.id)}
              >
                <strong>{preset.label}</strong>
                <span>{preset.description}</span>
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset className="settings-grid">
          <legend>Parameters</legend>
          <label>
            Feed source
            <select
              value={config.source}
              onChange={(event) => onChange("source", event.target.value)}
            >
              <option value="replay">Replay fixture</option>
              <option value="binance">Binance public</option>
            </select>
            <small>Replay is deterministic and available offline.</small>
          </label>
          <label>
            Symbol
            <input
              value={config.symbol}
              onChange={(event) =>
                onChange("symbol", event.target.value.toUpperCase())
              }
              inputMode="text"
              aria-describedby="symbol-help"
            />
            <small id="symbol-help">Venue symbol, for example BTCUSDT.</small>
          </label>
          <label>
            Gate mode
            <select
              value={config.gate_mode}
              onChange={(event) => onChange("gate_mode", event.target.value)}
            >
              <option value="neural">Neural model</option>
              <option value="uniform">Uniform baseline</option>
              <option value="static">Static baseline</option>
            </select>
          </label>
          <label>
            Gate interval{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="100"
                max="60000"
                step="100"
                value={config.gate_interval_ms}
                onChange={(event) =>
                  onChange("gate_interval_ms", Number(event.target.value))
                }
              />
              <em>ms</em>
            </span>
            <small>100–60,000 ms.</small>
          </label>
          <label>
            Higher-level influence{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={config.higher_level_influence}
                onChange={(event) =>
                  onChange("higher_level_influence", Number(event.target.value))
                }
              />
              <em>0–1</em>
            </span>
          </label>
          <label>
            Expert strength{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="0"
                max="5"
                step="0.05"
                value={config.expert_strength}
                onChange={(event) =>
                  onChange("expert_strength", Number(event.target.value))
                }
              />
              <em>x</em>
            </span>
          </label>
          <label>
            Flow window{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="4"
                max="512"
                step="1"
                value={config.flow_window_trades}
                onChange={(event) =>
                  onChange("flow_window_trades", Number(event.target.value))
                }
              />
              <em>trades</em>
            </span>
            <small>4–512 recent trades.</small>
          </label>
          <label>
            Base spread{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="0.01"
                max="1000"
                step="0.5"
                value={config.base_spread_bps}
                onChange={(event) =>
                  onChange("base_spread_bps", Number(event.target.value))
                }
              />
              <em>bps</em>
            </span>
          </label>
          <label>
            Max inventory{" "}
            <span className="input-suffix">
              <input
                type="number"
                min="0.001"
                max="1000000"
                step="0.001"
                value={config.max_inventory}
                onChange={(event) =>
                  onChange("max_inventory", Number(event.target.value))
                }
              />
              <em>BTC</em>
            </span>
          </label>
        </fieldset>
        <div className="settings-actions">
          <div
            className={`draft-state ${dirty ? "unsaved" : "saved"}`}
            role="status"
          >
            <span className="state-dot" />
            {dirty ? "Unsaved draft changes" : "Draft matches applied config"}
          </div>
          {invalid && (
            <span className="error-text" role="alert">
              {errors[0]}
            </span>
          )}
          {error && (
            <span className="error-text" role="alert">
              {error}
            </span>
          )}
          <button
            type="button"
            className="button ghost"
            onClick={onRevert}
            disabled={!dirty || applying}
          >
            Revert draft
          </button>
          <button
            type="button"
            className="button primary"
            onClick={onApply}
            disabled={!dirty || invalid || applying}
          >
            {applying ? "Applying…" : "Apply settings"}
          </button>
        </div>
        <div className="reset-row">
          <div>
            <strong>Reset simulation run</strong>
            <small>
              Starts a new backend run and clears this client’s live view. This
              is not an exchange action.
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
            disabled={resetting}
          >
            {resetting ? "Resetting…" : "Reset simulation"}
          </button>
        </div>
      </div>
    </details>
  );
}
