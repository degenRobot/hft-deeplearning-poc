# Binance sample

`binance-btcusdt-sample.jsonl` is a short normalized recording of public BTCUSDT market data.
It contains best bid and ask updates sampled at most once per second plus public aggregate trades.

Source: Binance's market-data-only WebSocket at `data-stream.binance.vision`, using the
`btcusdt@bookTicker` and `btcusdt@aggTrade` streams. Binance documents that host as public and
credential-free:

- <https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md>
- <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md>

Each line is one compact JSON object with a `kind` of `book` or `trade`. The repository keeps the
normalized fields used by the demo rather than the original WebSocket envelope. Run `make record`
to replace the sample with a capture of up to two minutes or 20,000 saved events.

This tiny sample is for education and pipeline testing. It is not a representative market history
and should not be used to evaluate a trading strategy.
