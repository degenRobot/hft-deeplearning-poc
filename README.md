# HFT Deep Learning POC

A small, read-only demo of a learned model above a fast market-making loop. Three deterministic
experts react to book and trade events; once per second, a tiny neural network changes their
weights. Deterministic code owns the synthetic quote and safety checks. The demo sends no orders,
needs no exchange credentials, and makes no production-HFT or profitability claim.

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

```sh
uv sync --extra dev
cd frontend && pnpm install --frozen-lockfile && cd ..
```

Start the backend and dashboard in separate terminals:

```sh
uv run uvicorn market_gate.api:app --app-dir src --reload
```

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

The committed sample and receipt work offline; `data/README.md` records provenance, while
`notebooks/gate_training.ipynb` shows the same model shape and Torch-free export synthetically.

The labels are one-second proxies; the live flow and reversion experts keep longer rolling state.
This proves the plumbing, not generalization, trading performance, or profitability. For optional
cloud parity, `make modal-plan` prints a no-contact plan. After checking credits and spend, run
`uv run --with modal==1.5.2 python scripts/train_on_modal.py --run`. No Modal run is recorded here.

## Change the inputs

Defaults live in `configs/demo.toml`; the dashboard edits the same fields. Try uniform or static
gate mode, zero higher-level influence, a shorter flow window, or Binance public data. Changes
stay in memory: **Reset simulation** starts a fresh run, while a backend restart restores defaults.

## Verify it

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
cd frontend && pnpm format:check && pnpm test && pnpm lint && pnpm build
```

## Proof boundary

The demo omits queue position, partial fills, fees, exchange latency, self-impact, and real P&L.
Python and internet WebSockets favor inspection, not colocated latency. The model proposes bounded
weights; deterministic code enforces market-data and risk rules.

Binance is the only live venue adapter in this slice. RISEx remains the next adapter because its
snapshot-before-ack and checksum rules deserve their own tests instead of a decorative selector.

The design borrows the slow-model / deterministic-controller split from the Apache-2.0-licensed
[VRM NN Lab](https://github.com/degenRobot/vrm-nn-lab), pared down for this trading POC.
