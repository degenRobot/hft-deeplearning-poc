import type { ConnectionStatus, FeedSource } from "../lib/types";

export function ConnectionPill({ status }: { status: ConnectionStatus }) {
  const labels: Record<ConnectionStatus, string> = {
    connecting: "Connecting",
    connected: "Connected",
    disconnected: "Disconnected",
    error: "Connection error",
  };
  return (
    <span className={`connection-pill ${status}`}>
      <span className="status-dot" />
      {labels[status]}
    </span>
  );
}

interface HeaderProps {
  status: ConnectionStatus;
  source: FeedSource;
  symbol: string;
  cadenceMs: number;
  apiUrl: string;
  message: string;
  onReconnect: () => void;
}

export function Header({
  status,
  source,
  symbol,
  cadenceMs,
  apiUrl,
  message,
  onReconnect,
}: HeaderProps) {
  const cadenceLabel =
    cadenceMs < 1000
      ? `${cadenceMs} ms gate`
      : `${(cadenceMs / 1000).toFixed(1)} s gate`;
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">MG</span>
          <span>Market Gate Lab</span>
        </div>
        <div className="topbar-actions">
          <ConnectionPill status={status} />
          <span className="read-only">READ ONLY</span>
          <button className="button ghost" onClick={onReconnect}>
            {status === "connected" ? "Reconnect" : "Connect"}
          </button>
        </div>
      </header>
      <section className="intro">
        <div>
          <span className="eyebrow">
            Educational simulation · event speed + {cadenceLabel}
          </span>
          <h1>Every tick. Two speeds.</h1>
          <p>
            Watch market events become signals, then watch a slower neural gate
            change the mix.
          </p>
        </div>
      </section>
      <div
        className={`stream-banner ${status}`}
        role="status"
        aria-live="polite"
      >
        <span className="stream-icon">
          {status === "connected" ? "↗" : "·"}
        </span>
        <span>{message}</span>
        <span className="stream-context">
          {symbol} ·{" "}
          {source === "binance" ? "Binance public" : "Replay fixture"}
        </span>
        <span className="stream-endpoint">{apiUrl}</span>
      </div>
    </>
  );
}
