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
- A 21,443-parameter PyTorch gate with a causal `30 x 10` input window.
- Uniform and static baselines, weight smoothing, stale-feed suppression, and an attribution
  ledger.
- A training script and notebook using synthetic, replay-shaped data.
- A Next.js dashboard for changing the feed, gate mode, influence, cadence, expert strength,
  spread, and paper inventory limit.

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
neural, uniform, and static weighting modes.

The backend also exposes `GET /health`, `GET/PATCH /config`, `GET /ledger`, and
`WS /ws/market` on <http://localhost:8000>.

## Train the example gate

```sh
uv sync --extra training
uv run python scripts/train_gate.py
```

This writes `artifacts/gate-demo.pt`, which is ignored by Git. The companion notebook is
`notebooks/gate_training.ipynb`. Both use deterministic synthetic data so the example stays
small and reproducible. A real experiment would need chronological train/validation/test
windows, train-only normalization, cost assumptions, and an untouched promotion holdout.

## Change the inputs

Defaults live in `configs/demo.toml`. The dashboard sends the same safe fields to the backend at
runtime. Useful first comparisons are:

- `gate_mode = "uniform"` to remove the learned gate.
- `gate_mode = "static"` for a fixed `50/35/15` mix.
- `higher_level_influence = 0` to make the higher-level gate inert.
- `source = "binance"` to consume public `bookTicker` and `aggTrade` streams.

Runtime changes are intentionally in-memory. Restarting the backend restores the TOML defaults.

## Verify it

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
cd frontend && pnpm test && pnpm lint && pnpm build
```

## Proof boundary

The demo does not model queue position, partial fills, fees, exchange latency, self-impact, or
real P&L. Python and internet WebSockets are useful here because the mechanism is easy to inspect,
not because they meet colocated trading latency. The main design idea is the authority boundary:
the learned model proposes bounded weights while deterministic code enforces market-data and risk
rules.

The structure was informed by the slow-model / deterministic-controller separation explored in
the Apache-2.0-licensed [VRM NN Lab](https://github.com/degenRobot/vrm-nn-lab). This repository's
implementation was written as a smaller, trading-specific proof of concept.
