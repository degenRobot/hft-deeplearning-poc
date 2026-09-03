"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { age, formatTimestamp, number, percent, price, signed } from "../lib/format";
import { parseStateMessage } from "../lib/normalize";
import { DEFAULT_CONFIG, EMPTY_STATE, type AppConfig, type ConnectionStatus, type MarketGateState } from "../lib/types";
import { API_DEFAULT, websocketUrl } from "../lib/connection";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || API_DEFAULT).replace(/\/$/, "");

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function ConnectionPill({ status }: { status: ConnectionStatus }) {
  const labels: Record<ConnectionStatus, string> = { connecting: "Connecting", connected: "Live feed", disconnected: "Disconnected", error: "Connection error" };
  return <span className={`connection-pill ${status}`}><span className="status-dot" />{labels[status]}</span>;
}

function SectionTitle({ index, eyebrow, title, children }: { index: string; eyebrow: string; title: string; children?: React.ReactNode }) {
  return <div className="section-title"><div><span className="eyebrow">{index} / {eyebrow}</span><h2>{title}</h2></div>{children && <p>{children}</p>}</div>;
}

function Metric({ label, value, detail, tone = "default" }: { label: string; value: string; detail?: string; tone?: string }) {
  return <div className={`metric ${tone}`}><span className="metric-label">{label}</span><strong>{value}</strong>{detail && <span className="metric-detail">{detail}</span>}</div>;
}

function WeightRow({ label, value, color }: { label: string; value: number; color: string }) {
  return <div className="weight-row"><div className="weight-label"><span>{label}</span><b>{percent(value)}</b></div><div className="weight-track"><span style={{ width: `${clamp(value * 100, 0, 100)}%`, background: color }} /></div></div>;
}

function HistoryChart({ history }: { history: MarketGateState[] }) {
  const width = 760;
  const height = 214;
  const padding = { left: 34, right: 16, top: 16, bottom: 28 };
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  const points = history.length > 1 ? history : [EMPTY_STATE, EMPTY_STATE];
  const toPoint = (value: number, index: number) => `${padding.left + (index / Math.max(points.length - 1, 1)) * usableWidth},${padding.top + (1 - clamp(value, 0, 1)) * usableHeight}`;
  const series = [
    { key: "microprice", color: "#0b8f7c", label: "Microprice" },
    { key: "flow", color: "#7568b3", label: "Flow" },
    { key: "reversion", color: "#b56a3c", label: "Reversion" },
  ] as const;
  return <div className="chart-wrap">
    <div className="chart-legend">{series.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label}</span>)}</div>
    <svg className="history-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Expert weight history chart">
      {[0, 0.5, 1].map((tick) => <g key={tick}><line x1={padding.left} x2={width - padding.right} y1={padding.top + (1 - tick) * usableHeight} y2={padding.top + (1 - tick) * usableHeight} className="chart-grid" /><text x="4" y={padding.top + (1 - tick) * usableHeight + 4} className="chart-axis">{Math.round(tick * 100)}%</text></g>)}
      {series.map((item) => <polyline key={item.key} points={points.map((snapshot, index) => toPoint(snapshot.gate.weights[item.key], index)).join(" ")} fill="none" stroke={item.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />)}
      <text x={padding.left} y={height - 6} className="chart-axis">{history.length ? "earliest" : "waiting for snapshots"}</text>
      <text x={width - padding.right} y={height - 6} textAnchor="end" className="chart-axis">now</text>
    </svg>
  </div>;
}

function Settings({ config, onChange, onApply, applying, error }: { config: AppConfig; onChange: (key: keyof AppConfig, value: string | number) => void; onApply: () => void; applying: boolean; error: string }) {
  return <details className="settings"><summary>Runtime settings <span>configure the simulation</span></summary><div className="settings-grid">
    <label>Feed source<select value={config.source} onChange={(event) => onChange("source", event.target.value)}><option value="replay">Replay fixture</option><option value="binance">Binance public</option></select></label>
    <label>Symbol<input value={config.symbol} onChange={(event) => onChange("symbol", event.target.value.toUpperCase())} inputMode="text" /></label>
    <label>Gate mode<select value={config.gate_mode} onChange={(event) => onChange("gate_mode", event.target.value)}><option value="neural">Neural gate</option><option value="uniform">Uniform baseline</option><option value="static">Static baseline</option></select></label>
    <label>Gate interval <span className="input-suffix"><input type="number" min="100" step="100" value={config.gate_interval_ms} onChange={(event) => onChange("gate_interval_ms", Number(event.target.value))} /><em>ms</em></span></label>
    <label>Higher-level influence <span className="input-suffix"><input type="number" min="0" max="1" step="0.05" value={config.higher_level_influence} onChange={(event) => onChange("higher_level_influence", Number(event.target.value))} /><em>0–1</em></span></label>
    <label>Expert strength <span className="input-suffix"><input type="number" min="0" max="2" step="0.05" value={config.expert_strength} onChange={(event) => onChange("expert_strength", Number(event.target.value))} /><em>x</em></span></label>
    <label>Base spread <span className="input-suffix"><input type="number" min="0" step="0.5" value={config.base_spread_bps} onChange={(event) => onChange("base_spread_bps", Number(event.target.value))} /><em>bps</em></span></label>
    <label>Max inventory <span className="input-suffix"><input type="number" min="0" step="0.001" value={config.max_inventory} onChange={(event) => onChange("max_inventory", Number(event.target.value))} /><em>BTC</em></span></label>
  </div><div className="settings-actions"><span className="settings-note">Changes affect the backend simulation after Apply.</span>{error && <span className="error-text" role="alert">{error}</span>}<button className="button primary" onClick={onApply} disabled={applying}>{applying ? "Applying…" : "Apply settings"}</button></div></details>;
}

export default function Home() {
  const [state, setState] = useState(EMPTY_STATE);
  const [hasSnapshot, setHasSnapshot] = useState(false);
  const [history, setHistory] = useState<MarketGateState[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [settingsError, setSettingsError] = useState("");
  const [configLoading, setConfigLoading] = useState(true);
  const [applying, setApplying] = useState(false);
  const [reconnectToken, setReconnectToken] = useState(0);
  const [lastError, setLastError] = useState("");
  const [nextRefresh, setNextRefresh] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/config`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Config request returned ${response.status}`);
        return await response.json() as Partial<AppConfig>;
      })
      .then((body) => {
        if (!active) return;
        setConfig((current) => ({ ...current, ...body }));
        setSettingsError("");
      })
      .catch((error: unknown) => {
        if (active) setSettingsError(error instanceof Error ? error.message : "Could not load config");
      })
      .finally(() => { if (active) setConfigLoading(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let closedByEffect = false;
    // A socket subscription is an external system; its initial state is set as it opens.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStatus("connecting");
    setLastError("");
    const socket = new WebSocket(websocketUrl(API_URL));
    wsRef.current = socket;
    socket.onopen = () => { setStatus("connected"); setLastError(""); };
    socket.onmessage = (event) => {
      const parsed = typeof event.data === "string" ? parseStateMessage(event.data) : null;
      if (!parsed) { setLastError("Received an invalid state snapshot"); return; }
      setState(parsed);
      setHasSnapshot(true);
      setNextRefresh(parsed.gate.cadence_ms);
      setHistory((items) => [...items, parsed].slice(-32));
    };
    socket.onerror = () => { setStatus("error"); setLastError(`Could not reach ${API_URL}`); };
    socket.onclose = () => {
      wsRef.current = null;
      if (closedByEffect) return;
      setStatus("disconnected");
      reconnectTimer = setTimeout(() => setReconnectToken((token) => token + 1), 3500);
    };
    return () => { closedByEffect = true; if (reconnectTimer) clearTimeout(reconnectTimer); socket.close(); };
  }, [reconnectToken]);

  useEffect(() => {
    if (!hasSnapshot || nextRefresh <= 0) return;
    const timer = setInterval(() => setNextRefresh((remaining) => Math.max(0, remaining - 100)), 100);
    return () => clearInterval(timer);
  }, [hasSnapshot, nextRefresh]);

  const updateConfig = (key: keyof AppConfig, value: string | number) => setConfig((current) => ({ ...current, [key]: value } as AppConfig));
  const applyConfig = async () => {
    setSettingsError("");
    setApplying(true);
    try {
      const response = await fetch(`${API_URL}/config`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
      if (!response.ok) throw new Error(`Config update returned ${response.status}`);
      const body = await response.json().catch(() => null) as Partial<AppConfig> | null;
      if (body) setConfig((current) => ({ ...current, ...body }));
    } catch (error) { setSettingsError(error instanceof Error ? error.message : "Could not update config"); }
    finally { setApplying(false); }
  };

  const statusMessage = useMemo(() => {
    if (configLoading) return "Loading runtime config…";
    if (status === "connecting") return "Connecting to backend…";
    if (status === "connected" && !hasSnapshot) return "Connected; waiting for first snapshot…";
    if (lastError) return lastError;
    return "Backend stream is not connected";
  }, [configLoading, hasSnapshot, lastError, status]);

  const empty = !hasSnapshot;
  return <main className="shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">MG</span><span>Market Gate Lab</span></div><div className="topbar-actions"><ConnectionPill status={status} /><span className="read-only">READ ONLY</span><button className="button ghost" onClick={() => setReconnectToken((token) => token + 1)}>{status === "connected" ? "Reconnect" : "Connect"}</button></div></header>

    <section className="intro"><div><span className="eyebrow">Educational simulation · two clocks</span><h1>Make the slow gate visible.</h1><p>A read-only market microstructure lab where a learned model changes the mix of fast, deterministic experts. The gate chooses weights; the fast plane owns quotes, inventory and safety.</p></div><div className="intro-note"><span className="note-kicker">Boundary</span><strong>1 Hz gate</strong><span>→ weights only →</span><strong>event-speed quote</strong></div></section>

    <div className={`stream-banner ${status}`} role="status"><span className="stream-icon">{status === "connected" ? "↗" : "·"}</span><span>{statusMessage}</span><span className="stream-endpoint">{API_URL}</span></div>

    <section className="dashboard-section"><SectionTitle index="01" eyebrow="market state" title="What the feed says"><span>Normalized snapshots from the backend. Values remain intentionally blank until a real snapshot arrives.</span></SectionTitle><div className="metric-grid market-metrics"><Metric label="Mid price" value={empty ? "—" : price(state.market.mid)} detail={state.symbol} tone="accent" /><Metric label="Spread" value={empty ? "—" : `${number(state.market.spread_bps, 2)} bps`} detail="top of book" /><Metric label="Imbalance" value={empty ? "—" : signed(state.market.imbalance)} detail="bid pressure" tone={state.market.imbalance >= 0 ? "positive" : "negative"} /><Metric label="Trade flow" value={empty ? "—" : signed(state.market.trade_flow)} detail="signed impulse" tone={state.market.trade_flow >= 0 ? "positive" : "negative"} /></div></section>

    <section className="dashboard-section"><SectionTitle index="02" eyebrow="slow plane" title="The neural gate sets the mix"><span>Refreshes at its configured cadence. It never emits an order.</span></SectionTitle><div className="split-grid"><article className="panel gate-panel"><div className="panel-heading"><span className="panel-kicker violet">GATE REGIME</span><span className="model-version">{empty ? "—" : state.gate.model_version}</span></div><div className="regime">{empty ? "Awaiting feed" : state.gate.regime}</div><div className="confidence"><span>Confidence</span><b>{empty ? "—" : percent(state.gate.confidence)}</b><div className="confidence-track"><span style={{ width: `${empty ? 0 : state.gate.confidence * 100}%` }} /></div></div><div className="gate-footer"><span>{empty ? "Next refresh —" : `Next refresh ${age(nextRefresh)}`}</span><span>{empty ? "—" : `${state.gate.mode} · ${state.gate.cadence_ms} ms`}</span></div></article><article className="panel weights-panel"><div className="panel-heading"><span className="panel-kicker teal">CURRENT WEIGHTS</span><span className="weight-total">Σ 1.00</span></div><WeightRow label="Microprice pressure" value={empty ? 0 : state.gate.weights.microprice} color="#0b8f7c" /><WeightRow label="Trade-flow impulse" value={empty ? 0 : state.gate.weights.flow} color="#7568b3" /><WeightRow label="Short reversion" value={empty ? 0 : state.gate.weights.reversion} color="#b56a3c" /><p className="panel-caption">{empty ? "Waiting for the gate’s first decision." : "Bounded and smoothed before reaching the fast mixer."}</p></article></div></section>

    <section className="dashboard-section"><SectionTitle index="03" eyebrow="fast plane" title="Three experts, one accountable quote"><span>Every contribution stays legible. Scores are event-speed signals, not trade recommendations.</span></SectionTitle><div className="expert-grid">{state.experts.map((expert, index) => <article className="expert-card" key={expert.id}><div className="expert-index">0{index + 1}</div><h3>{expert.label}</h3><div className="expert-values"><div><span>score</span><b className={expert.score >= 0 ? "up" : "down"}>{empty ? "—" : signed(expert.score)}</b></div><div><span>weight</span><b>{empty ? "—" : percent(expert.weight)}</b></div><div><span>contribution</span><b className={expert.contribution >= 0 ? "up" : "down"}>{empty ? "—" : signed(expert.contribution)}</b></div></div></article>)}</div><div className="panel chart-panel"><div className="panel-heading"><div><span className="panel-kicker teal">WEIGHT / CONTRIBUTION HISTORY</span><h3>How the mix has moved</h3></div><span className="chart-window">last {history.length || 0} snapshots</span></div><HistoryChart history={history} /></div></section>

    <section className="dashboard-section"><SectionTitle index="04" eyebrow="paper state" title="Synthetic quote, no execution"><span>A toy quote and paper ledger make the effect inspectable without connecting to an order endpoint.</span></SectionTitle><div className="lower-grid"><article className="quote-panel"><span className="panel-kicker lime">SYNTHETIC QUOTE</span>{state.quote && !empty ? <><div className="quote-values"><div><span>bid</span><strong>{price(state.quote.bid)}</strong></div><i>/</i><div><span>ask</span><strong>{price(state.quote.ask)}</strong></div></div><div className="quote-sub">mid {price(state.market.mid)} · spread {number(state.market.spread_bps)} bps</div></> : <div className="quote-empty">No quote until a healthy snapshot arrives.</div>}</article><article className="panel paper-panel"><span className="panel-kicker teal">PAPER INVENTORY / P&amp;L</span><div className="paper-values"><Metric label="Inventory" value={empty ? "—" : signed(state.paper.inventory, 4)} detail="BTC" /><Metric label="P&amp;L" value={empty ? "—" : signed(state.paper.pnl)} detail="paper units" tone={state.paper.pnl >= 0 ? "positive" : "negative"} /></div></article><article className="panel health-panel"><span className="panel-kicker teal">FEED / RISK HEALTH</span><div className="health-status"><span className={`health-dot ${empty ? "waiting" : state.health.status === "ok" ? "ok" : "warn"}`} /><strong>{empty ? "Waiting" : state.health.status}</strong></div><dl><div><dt>Message age</dt><dd>{empty ? "—" : age(state.health.message_age_ms)}</dd></div><div><dt>Reconnects</dt><dd>{empty ? "—" : state.health.reconnects}</dd></div><div><dt>Last snapshot</dt><dd>{formatTimestamp(state.timestamp)}</dd></div></dl></article></div></section>

    <Settings config={config} onChange={updateConfig} onApply={applyConfig} applying={applying} error={settingsError} />
    <footer className="footer"><div><strong>Market Gate Lab</strong><span> · educational simulation</span></div><p>Not real HFT. Not profitability evidence. No live orders are sent from this interface.</p></footer>
  </main>;
}
