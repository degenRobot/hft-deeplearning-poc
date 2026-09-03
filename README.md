# Market Gate Lab

A replay-first, read-only teaching backend. Three deterministic experts observe normalized
book/trade events. A one-second gate selects bounded weights over a causal 30 x 10 feature
window; the deterministic risk kernel alone produces synthetic paper quotes.

It has no credentials, private streams, database, live trading, or execution endpoints. It is
not a profitability claim or a production HFT system.

## Run

```sh
uv sync --extra dev
uv run pytest
uv run uvicorn market_gate.api:app --app-dir src
```

The default `configs/demo.toml` replays `fixtures/replay.jsonl` at startup, so no network or
saved model is required. Visit `GET /health`, edit safe lab settings with `GET/PATCH /config`,
inspect attribution with `GET /ledger`, or connect to `WS /ws/market`.

For the optional PyTorch artifact demo:

```sh
uv sync --extra training
uv run python scripts/train_gate.py
```

The notebook at `notebooks/gate_training.ipynb` uses a deterministic synthetic training set,
compares uniform and static baselines, and demonstrates artifact save/load. Training artifacts
are ignored by Git.

## Design limits

- `BinancePublicFeed` uses only public `bookTicker` and `aggTrade` market data on the
  market-data-only host, with reconnect/backoff accounting.
- The gate is advisory. It cannot bypass stale-data, invalid-book, or inventory-cap suppression.
- Replay is deterministic; decisions include scores, weights, contributions, and quote/risk
  outcome in a bounded ledger.
- The example does not model queue position, fills, fees, latency, impact, or real P&L.
