"use client";

import { Dashboard } from "../components/Dashboard";
import { Header } from "../components/Header";
import { SettingsPanel } from "../components/SettingsPanel";
import { useMarketGate } from "../hooks/useMarketGate";

export default function Home() {
  const market = useMarketGate();
  const statusMessage = market.configLoading
    ? "Loading runtime config…"
    : market.status === "connecting"
      ? "Opening the backend socket; waiting for a ready snapshot…"
      : market.lastError ||
        (market.status === "connected"
          ? market.state.source === "binance"
            ? "Streaming Binance public market data"
            : "Streaming the deterministic replay fixture"
          : "Backend stream is not connected");

  return (
    <main className="shell">
      <Header
        status={market.status}
        source={
          market.hasSnapshot ? market.state.source : market.appliedConfig.source
        }
        symbol={
          market.hasSnapshot ? market.state.symbol : market.appliedConfig.symbol
        }
        cadenceMs={
          market.hasSnapshot
            ? market.state.gate.cadence_ms
            : market.appliedConfig.gate_interval_ms
        }
        ready={market.hasSnapshot}
        apiUrl={market.apiUrl}
        message={statusMessage}
        onReconnect={market.reconnect}
      />
      <Dashboard
        state={market.state}
        history={market.history}
        ready={market.hasSnapshot}
        nextRefresh={market.nextRefresh}
        mode={
          market.hasSnapshot
            ? market.state.gate.mode
            : market.appliedConfig.gate_mode
        }
      />
      <SettingsPanel
        config={market.draftConfig}
        dirty={market.dirty}
        invalid={market.configErrors.length > 0}
        errors={market.configErrors}
        applying={market.applying}
        resetting={market.resetting}
        error={market.settingsError}
        resetError={market.resetError}
        resetResult={market.resetResult}
        onChange={market.updateDraft}
        onPreset={market.selectPreset}
        onApply={market.applyConfig}
        onRevert={market.revertDraft}
        onReset={market.resetRun}
      />
      <footer className="footer">
        <div>
          <strong>Market Gate Lab</strong>
          <span> · educational simulation</span>
        </div>
        <p>
          Not real HFT. Not profitability evidence. No live orders are sent from
          this interface.
        </p>
      </footer>
    </main>
  );
}
