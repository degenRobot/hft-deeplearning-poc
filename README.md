# HFT Deep Learning POC

A small, read-only demo of one way a learned model can sit above a fast market-making loop.
Three deterministic experts react to book and trade events. Once per second, a tiny neural
network changes their weights. Deterministic code still owns the synthetic quote and every
safety check.

This is an educational simulation. It sends no orders, needs no exchange credentials, and does
not claim to reproduce production HFT or prove profitability.

## What is here

- A FastAPI backend with deterministic replay and a public Binance market-data adapter.
- Three legible expert signals: microprice pressure, trade-flow impulse, and short reversion.
- A 21,443-parameter gate with a causal `30 x 10` input window: PyTorch for the
  training example, NumPy-only inference in the live demo.
- Uniform and static baselines, weight smoothing, stale-feed suppression, and an attribution
  ledger.
- A short recorded Binance sample, an offline training script, and a synthetic teaching notebook.
- A Next.js dashboard with draft presets for changing the feed, gate mode, influence, cadence,
  trade-flow window, expert strength, spread, and paper inventory limit.

```text
book + trades ──> fast experts ──> weighted mixer ──> risk kernel ──> synthetic quote
                       ▲                                   │
                       │ weights only                      └── no order route
              1 Hz neural gate
              30 x 10 features
```

## Run the demo

Install the two small applications:

```sh
uv sync --extra dev
cd frontend && pnpm install --frozen-lockfile && cd ..
```

Start the backend in one terminal:

```sh
uv run uvicorn market_gate.api:app --app-dir src --reload
```

Start the dashboard in another:

```sh
cd frontend
pnpm dev
```

Open <http://localhost:3000>. Replay mode is the default, so the whole screen works without an
internet connection. Open **Runtime settings** to switch to Binance public data or compare the
neural, uniform, and static weighting modes. Presets change only the local draft until you choose
**Apply settings**, so it is easy to compare a proposed setup with the running one.

The backend also exposes `GET /health`, `GET/PATCH /config`, `POST /reset`, `GET /ledger`,
`GET /training`, and `WS /ws/market` on <http://localhost:8000>.

## Record and train the example gate

```sh
uv sync --extra training
uv run --extra training python scripts/record_binance.py
uv run --extra training python scripts/train_gate.py
```

The recorder listens for up to two minutes, or 20,000 saved events, on Binance's public
`bookTicker` and `aggTrade` streams. It keeps all aggregate trades, samples the top of book once
per second, and writes normalized JSONL without credentials or order access. The trainer builds
causal one-second frames, forms `30 x 10` windows, keeps validation later than training, and
writes two inspectable outputs:

- `models/gate-binance-demo.npz`, a separate model that does not replace the live demo model.
- `artifacts/training-demo.json`, the receipt shown in the dashboard.

The committed sample and receipt let the UI work without making a fresh network call. See
`data/README.md` for provenance. The companion `notebooks/gate_training.ipynb` remains a smaller
synthetic walkthrough of the same model shape and Torch-free export.

The labels are deliberately simple one-second proxies for the three live experts; the live flow
and reversion experts keep longer rolling state. The short run proves only that the data-to-model
plumbing works. It is not a backtest or evidence of generalization, trading performance, or
profitability.

### Optional Modal parity plan

Modal is optional; local CPU remains the default. Inspect the safe local plan with:

```sh
make modal-plan
```

That dry-run prints the source and model hashes plus the training config, and contacts no remote
compute. A real smoke run requires the explicit `--run` flag:

```sh
uv run --with modal==1.5.2 python scripts/train_on_modal.py --run
```

Check current Modal credits and expected spend before starting it. The committed receipt is from
local CPU, and no remote Modal smoke run has been completed here.

## Change the inputs

Defaults live in `configs/demo.toml`. The dashboard sends the same safe fields to the backend at
runtime. Useful first comparisons are:

- `gate_mode = "uniform"` to remove the learned gate.
- `gate_mode = "static"` for a fixed `50/35/15` mix.
- `higher_level_influence = 0` to make the higher-level gate inert.
- `flow_window_trades = 16` to make the flow expert react to a shorter recent window.
- `source = "binance"` to consume public `bookTicker` and `aggTrade` streams.

Runtime changes are intentionally in-memory. **Reset simulation** starts a fresh run with the
applied runtime config; restarting the backend restores the TOML defaults. Feed-generation and
event counters in the health panel make both transitions visible.

## Verify it

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
cd frontend && pnpm format:check && pnpm test && pnpm lint && pnpm build
```

## Proof boundary

The demo does not model queue position, partial fills, fees, exchange latency, self-impact, or
real P&L. Python and internet WebSockets are useful here because the mechanism is easy to inspect,
not because they meet colocated trading latency. The main design idea is the authority boundary:
the learned model proposes bounded weights while deterministic code enforces market-data and risk
rules.

Binance is the only live venue adapter in this slice. RISEx remains the next adapter because its
snapshot-before-ack and checksum rules deserve their own tests instead of a decorative selector.

The structure was informed by the slow-model / deterministic-controller separation explored in
the Apache-2.0-licensed [VRM NN Lab](https://github.com/degenRobot/vrm-nn-lab). This repository's
implementation was written as a smaller, trading-specific proof of concept.
