# Watch a model learn

The [recorded run summary](../artifacts/training-example/summary.json) contains the
completed September 5 local and Modal examples: 9,013 public events across 15 minutes,
415 supervised examples, 42 RL decisions and 116 holdout examples. Both phases changed
weights. Supervised loss fell from 1.1051 to 1.0826. All holdout rewards were zero, so
this capture demonstrates training mechanics and provides no useful model ranking.

The local and Modal models gave the same highest-weight expert on all 865 usable
windows; their probabilities differed by less than 2.81e-8. Their exported parameter
bytes are not identical across ARM macOS and x86 Linux. The models remain experimental.

Open `/training` and press **Start local training**. The page shows real optimizer
steps from a separate experimental gate: input features, hidden-neuron activations,
loss, gradients, parameter changes and expert probabilities before and after each update.
The live terminal continues using its own model.

## The training task

The network has the same dimensions as the terminal gate: 30 observed one-second
frames × 10 features → 64 ReLU neurons → 32 ReLU neurons → 3 expert weights.
There are 21,443 learned parameters. The raw inputs retain the existing feature order.
Normalization statistics come only from the supervised prefix. Export folds those
statistics into float64 first-layer weights so the existing NumPy inference reader can
consume raw inputs without a separate normalizer.

The recording is divided by frame position before fitting: 50% supervised, 30% replay
adaptation, 20% final holdout. Windows crossing boundaries are excluded, as are the
first 30 frames after each boundary because features depend on earlier price history.
Missing seconds exclude windows; no synthetic books or interpolated prices are inserted.
The run requires at least 30 supervised examples, 15 causal RL decisions and 15 holdout
examples after these checks. It fails if the recording falls short.

During 12 supervised epochs, soft targets favor experts whose proxy signals align with
the mid-price move five seconds later. Cross entropy updates the full supervised batch;
the pictured feature window and activations are one example in that batch. Gradients
and loss therefore describe the batch, not just the displayed window.

During replay adaptation, the policy samples one expert using only the current input.
The recorded five-second outcome then supplies a reward. REINFORCE applies
`-log(probability of sampled expert) × advantage`, with a running reward baseline.
The next decision occurs after the preceding reward matures. Rewards are scaled using
supervised-only statistics and bounded with `tanh`. This is a contextual-bandit teaching
example, following the [PyTorch REINFORCE construction](https://docs.pytorch.org/docs/stable/distributions.html).
The policy updates online within recorded-market replay; it does not learn from a
connected exchange or execute orders.

The final holdout never updates weights or normalization. It compares supervised and
adapted models against uniform weights and each individual expert on the same later
windows. Numbers are mean expert-proxy utility in basis points, not simulated fills,
fees-adjusted returns or trading P&L. Overlapping windows and a short sample do not
establish generalization. The adapted model can underperform a simple baseline.

## Data and reproduction

The public source combines Binance `bookTicker` and `aggTrade`. This capture samples
books at 100 ms and retains aggregate trades. `quote_updates` consequently counts
retained books, while the live terminal counts all received books. Training orders by
event time; the live engine sequences by receive time. Those differences remain explicit.

[Binance public archives](https://github.com/binance/binance-public-data) offer spot
trades, aggregate trades and candles, which do not contain the best bid/ask prices and
sizes needed by this feature set. See the
[public market-data endpoint documentation](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md).

```sh
uv sync --extra dev --extra training
# Use a new output path for a new capture. The bundled recording is already complete.
uv run python scripts/record_binance.py --symbol BTCUSDT --seconds 900 \
  --book-interval-ms 100 --max-events 500000 --max-bytes 100000000 \
  --output data/my-new-capture.jsonl

uv run python scripts/run_training_lab.py --input data/training-public.jsonl \
  --output artifacts/my-local-training --pace 0.25
```

The browser button uses the fixed `data/training-public.jsonl` dataset and creates a
new directory under `artifacts/training-runs/`. It cannot select arbitrary files or
start cloud compute. One OS lock prevents concurrent local and Modal training writers.
Stop terminates the local child process; server shutdown also stops a child it owns.
The UI labels completed runs and clears unavailable telemetry. Reloading a page does
not start training.

## One bounded Modal run

```sh
# Prints the exact source/input hashes and resource plan without contacting Modal.
uv run --with modal==1.5.2 python scripts/train_lab_on_modal.py \
  --output artifacts/my-modal-training

# Execute one run using the two local credentials, after reviewing the plan.
uv run --with modal==1.5.2 python scripts/train_lab_on_modal.py --run \
  --env-file /path/to/root/.env --output artifacts/my-modal-training
```

The runner reads only `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from the local dotenv
file. It uploads an explicit list of committed training modules and the selected public
recording bytes, with hashes. It never mounts the repository root or dotenv file.
The same generator trains locally and remotely. The remote function streams actual
steps through [Modal's generator invocation](https://modal.com/docs/guide/function-invocation-methods)
and returns two experimental NPZ models and a receipt. Completion is published after
the returned artifacts are saved locally.

The single ephemeral run has two CPU cores, a 2 GiB memory limit, no retries and a
600-second function timeout. There is no deployment, GPU or persistent volume.
[Modal resource limits](https://modal.com/docs/guide/resources) bound function resources;
the timeout excludes image build/startup and is not a whole-job billing cap.

Artifacts include the input hash, source hashes, package versions, split boundaries,
hyperparameters, model hashes and heldout comparisons. `modal-execution.json` additionally
records the actual app/function-call IDs. Runtime model promotion is a separate decision.
