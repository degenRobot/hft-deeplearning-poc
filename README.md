# Market Gate Lab

A small, hands-on illustration of a slower neural network guiding several fast,
deterministic HFT-style algorithms. The fun is seeing the whole loop: market events,
expert signals, neural activations, changing weights and synthetic quotes.

The question is whether a learned controller could help decide **which simple rule to
trust, and by how much**, while deterministic code keeps control of the quote. This
is an educational experiment, with no exchange orders, production-HFT claim or proven alpha.

## Run it locally

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js and pnpm.

```sh
uv sync --extra dev --extra training
cd frontend
pnpm install --frozen-lockfile
cd ..
```

Start the backend and frontend in separate terminals:

```sh
uv run uvicorn market_gate.api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

```sh
cd frontend
pnpm dev
```

Open [localhost:3000](http://localhost:3000). The backend listens on loopback port
8000. The terminal defaults to deterministic replay; public Binance books and trades
are available from Runtime settings without exchange credentials. Apply settings to
start a fresh run. A backend restart restores the defaults in `configs/demo.toml`.

## Three ways to explore

| Page | What to look at |
| --- | --- |
| [Background](http://localhost:3000/background) | The idea, the architecture and what would be worth testing next. Static; works without the backend. |
| [Live terminal](http://localhost:3000) | Follow replay or public market data through the model, three experts and synthetic quotes. Compare neural, uniform and static weights. |
| [Training lab](http://localhost:3000/training) | Run supervised training and replay adaptation, inspect actual gradients and weight changes, or watch the optional Live RL updates. |

## How the loop works

```text
recent 30 × 10 feature window → slower neural gate → bounded, smoothed weights
                                                           ↓
public books + trades → three fixed experts → weighted mix → risk checks → synthetic quotes
```

The experts measure microprice pressure, trade-flow impulse and short reversion.
They run on accepted events; the gate refreshes their allocation once per second by
default. The live gate has 21,443 parameters and uses NumPy inference. Influence is
blended with uniform weights and changes are smoothed before reaching the mixer.
Deterministic rules suppress quotes for stale or invalid books and the paper inventory
cap. The network cannot bypass those checks.

## What actually learns

The ordinary terminal runs inference from `models/gate-demo.npz`. Optional **Live RL**
updates only its 99 output weights and biases; both hidden layers and all three expert
rules stay fixed. It uses a delayed directional reward from observed prices. Updates
stay in memory: pause retains them, while reset or runtime setting changes restore
the checkpoint. This is a small policy-learning demonstration, not a profit objective.

The Training Lab performs real optimizer updates on recorded data, then evaluates a
later holdout that never trains the model. Historical one-second candles are a separate
offline approximation: unavailable book features are zero, and candle closes proxy
midprices. Those models are teaching artifacts that the live book-based gate rejects.
Training does not automatically replace the live model.

Quotes and paper results omit realistic queue position, partial fills, fees, latency
and market impact. Proxy rewards and visible learning do not establish trading returns.
Python and internet WebSockets make the mechanism easy to inspect; they do not measure
colocated HFT performance.

## Training and evidence

- [Training lab guide](docs/training-lab.md): datasets, temporal splits, local runs, Live RL and optional Modal execution.
- [Data provenance](data/README.md): bundled recordings and their limitations.
- [Experiment contract](docs/experiment-contract.md) and [experiment receipt](artifacts/phase2-summary.json): acceptance rules and retained results, including rejected captures.

For a reproducible synthetic export example, run `make train RUN_DIR=artifacts/my-first-run`.
Choose a new output directory for each run. `make experiment-check` verifies the bounded
capture experiment; `make test` runs backend and frontend checks.

Modal is optional. Install its extra with `uv sync --extra dev --extra training --extra cloud`.
Configure `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` locally through the Training Lab or
the ignored `.env` at the main checkout root. A Modal run uploads the selected dataset
and training code and uses your account's compute. The guide describes what runs and
where its receipts are written.

The next useful experiments are longer book/trade captures with costs in evaluation,
regime features, shadow checks before model promotion, a full-depth adapter, and a
compiled fast path with the slow controller kept separate. These are hypotheses to
test, not implemented capabilities or evidence of an edge.

The slow-model / deterministic-controller split draws on
[VRM NN Lab](https://github.com/degenRobot/vrm-nn-lab).
