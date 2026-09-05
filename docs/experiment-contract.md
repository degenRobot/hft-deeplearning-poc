# Experiment contract

The lab demonstrates a neural gate above deterministic quote logic. Quotes and fills are synthetic; the offline training objective measures agreement with future mid-price movement, not trading returns.

## Runtime state

A quote is eligible to display or fill only while its underlying book is fresh. Fresh trades do not refresh the book. A trade arriving after quote expiry cannot fill that quote; a fresh book allows quoting to resume. The toy ledger marks inventory at the current book mid. Replay is explicitly BTCUSDT and event symbols must agree with the configured input.

The runtime owns run_id and a monotonically increasing feed_generation within one backend process. Reset and successful configuration mutations acknowledge that identity. The browser rejects older queued snapshots after acknowledgement, clears state on malformed input or receive timeout, and waits for a valid ready snapshot before displaying market values again. A new socket connection may establish a new backend process generation; an old socket cannot restore prior state.

## Offline measurement

The teaching pipeline retains event-time ordering. Binance runtime ordering uses receive time; book downsampling and proxy expert labels also differ from live rolling state. Sharing feature names does not make the offline model runtime-equivalent.

Frame close timestamps are retained. Rolling feature history resets after missing seconds, and windows spanning a missing second are excluded. The chronological split excludes every input/target window crossing its boundary. Prices and sizes must be finite and valid; each recording contains one venue and symbol. Portable gate exports use 30 frames by 10 features.

Every evaluated model is compared with uniform weighting, static 50/35/15 weighting and each individual expert on identical held-out utilities. The report includes input/model/config/source provenance, frame and exclusion counts, window timestamps, environment, and all baseline results. Original teaching artifacts remain unchanged; corrected training produces a new versioned reference.

## Bounded sample experiment

The local preregistration fixes three capture windows before collection. Each permits at most 120 seconds or 20,000 saved events, with book updates sampled at 1,000 ms. Failed or rejected captures are retained and are not replaced automatically.

For an accepted sample, use seeds 7, 17 and 27, 20 epochs, learning rate 0.001, lookback 30, horizon 5 and the fixed chronological split. Require at least 30 training and 15 validation windows after exclusions. These are operational floors for this experiment, not evidence of statistical adequacy. Nine fits over three samples are three market samples with initialization repeats.

## Cloud parity

Freeze the corrected source and a new local reference before remote comparison. The default command must make no Modal contact or writes. Remote invocation requires clean tracked source, an enumerated upload, a unique writable output path, explicit resource limits and account/payload/spend authorization.

CPU/memory requests and execution timeout are distinct from billing limits and whole-job elapsed time. Preserve the strict model-file hash result. A mismatch is a result to investigate; it does not authorize retries or a retrospective tolerance change. Record actual environment and execution identifiers, and mark unavailable facts explicitly. Cloud may remain not run without blocking local evaluation or browser verification.

## Ownership and release

Runtime, measurement, parity and frontend changes have separate file owners. Freeze shared contracts before parallel edits; the coordinator reviews diffs, reruns acceptance checks and owns Git/PR actions. Fresh review and browser evidence close a phase; passing a narrow helper test alone does not prove a full lifecycle.

Further venue work, hosted deployment, runtime-equivalent research, private streams and orders require a separate objective. No experiment artifact automatically replaces the live demonstration model.
