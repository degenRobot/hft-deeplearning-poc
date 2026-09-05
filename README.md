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

## Read the visual flow

The dark terminal starts with model control: the last causal 30 × 10 feature window,
all 64 and 32 hidden ReLU activations, then proposed versus applied expert weights.
The activation values come from the same forward pass as those weights. Brightness
uses `log1p(value) / log1p(layer maximum)` within each layer and pass; zero stays dim.
The drawn lines show layer flow, not individual connection strengths. This is inference,
not live retraining. Baseline policies bypass the activation display.

The matching expert colors continue into the fast path: actual scores, weighted
contributions and the deterministic risk check before a synthetic quote. Input
explanations and exact neuron values are available in expandable inspectors.

The market view pairs the event tape with one-second **trade-price** OHLC candles and
traded volume. Candles aggregate every accepted trade before UI sampling, using the
local receive clock on Binance and the shifted fixture clock on replay. Empty seconds
remain gaps. The latest candle may still be forming; history starts with the current
run. The 30/60/90-second controls change the visible window, not candle duration.

Telemetry is bounded to 48 recent decision-producing events, 30 feature frames,
96 hidden activations and 90 observed candle buckets. The tape shows the latest 12
retained events and can filter to trades. WebSocket snapshots arrive every 100 ms;
the event pace reflects only the retained window, not total exchange throughput.
Pulses indicate observed updates, not measured inference or transport latency.

`Reduce motion` stops animation while values keep updating. Keyboard and touch users
can inspect candles and neurons through native selectors. Stale or disconnected
snapshots clear the charts and quote, and new runs reset observation buffers.

## Understand the training artifacts

The active dashboard gate loads `models/gate-demo.npz`. The dashboard's **original v1 offline
walkthrough** reads the archived `artifacts/training-demo.json`; its separate
`models/gate-binance-demo.npz` is not the active gate. Running a trainer does not update the
dashboard or either committed model.

The corrected v2 pipeline retains event-time frame timestamps, resets feature history at gaps,
excludes windows crossing missing seconds, and leaves an embargo between training and validation.
It compares the neural gate with uniform, static 50/35/15 and all three individual experts on the
same held-out proxy labels. These one-second labels differ from the live receive-time clock and
rolling expert state; results do not establish runtime equivalence or trading performance.

The September 5 bounded experiment attempted three public two-minute captures. Their usable
training/validation counts were **0/16, 3/17 and 24/0**, below the predefined **30/15** minimum.
All three were retained and rejected, with **zero experimental models and zero replacement
captures**. Reproduce the counts and verify the evidence hashes with `make experiment-check`.
See [the frozen contract](docs/experiment-contract.md) and
[the complete experiment receipt](artifacts/phase2-summary.json).

## Run a local reproducibility example

For the new public-data supervised + RL training view, open **`/training`**.
It displays actual input windows, activations, gradients and weight changes while a
bounded example runs. See [the training lab guide](docs/training-lab.md) for data,
temporal splits, local replay updates and the one-shot Modal runner.

```sh
uv sync --extra dev --extra training
make train RUN_DIR=artifacts/my-first-run
```

This uses `fixtures/parity-replay-v2.jsonl`, five deterministic cycles of the existing synthetic
replay fixture, to check v2 training and portable NumPy export. It is separate from the public-data
experiment and makes no model-quality claim. It writes a new model and receipt under the selected
run directory and refuses to overwrite them. Choose a new directory for each run.

To record public Binance data for a separately planned experiment, use an explicit new path:

```sh
make record RUN_DIR=artifacts/my-public-sample
make train INPUT=artifacts/my-public-sample/recording.jsonl RUN_DIR=artifacts/my-public-fit
```

The recorder stops after two minutes or 20,000 saved events, keeps aggregate trades and samples
books at 1,000 ms. Two minutes need not yield enough contiguous windows. Set an acceptance policy
before collecting; a nonempty trainer split alone does not satisfy the September experiment's
30/15 floor. The original sample's provenance is in [data/README.md](data/README.md).

`make modal-plan` prints a no-contact plan against the **new synthetic v2 reference**
`models/gate-parity-v2.npz`. Remote execution is optional and remains **not run**. It uploads
tracked private source and the selected fixture; account balance, payload and spend need approval
before invoking `--run`. CPU/memory limits and execution timeout do not cap whole-job charges.
See [the parity reference receipt](artifacts/parity-reference-v2.json).

## Change the inputs

Defaults live in `configs/demo.toml`; the dashboard edits the same fields. Try uniform or static
gate mode, zero higher-level influence, a shorter flow window, or Binance public data. Changes
stay in memory: **Reset simulation** starts a fresh run, while a backend restart restores defaults.

## Verify it

```sh
uv run --extra dev --extra training ruff format --check .
uv run --extra dev --extra training ruff check .
uv run --extra dev --extra training pytest
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
