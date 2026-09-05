# Watch a model learn

The [recorded run summary](../artifacts/training-example/summary.json) contains the
completed September 5 local and Modal examples: 9,013 public events across 15 minutes,
415 supervised examples, 42 RL decisions and 116 holdout examples. Both phases changed
weights. Supervised loss fell from 1.1051 to 1.0826. All holdout rewards were zero, so
this capture demonstrates training mechanics and provides no useful model ranking.

The local and Modal models gave the same highest-weight expert on all 865 usable
windows; their probabilities differed by less than 2.81e-8. Their exported parameter
bytes are not identical across ARM macOS and x86 Linux. The models remain experimental.

Open `/training`, choose **Local** or **Modal**, configure the model and start training.
Both backends show real optimizer steps: input features, hidden-neuron activations,
loss, gradients, parameter changes and expert probabilities before and after each update.
The live terminal continues using its own model.

The [configurable-run receipt](../artifacts/training-example/options-validation.json)
also records browser-launched medium local and large Modal runs. Each completed two
supervised updates and 42 replay decisions, with verified checkpoint shapes and hashes.
The large model's supervised loss increased over these two updates; these runs verify
configuration and transport, not improved model quality.

## The training task

Every model takes 30 observed one-second frames × 10 features, passes them through
two ReLU hidden layers and produces 3 expert weights. Choose a preset or enter custom
widths for the two hidden layers:

| Preset | Hidden layer 1 | Hidden layer 2 | Learned parameters |
| --- | ---: | ---: | ---: |
| Small | 64 | 32 | 21,443 |
| Medium | 256 | 128 | 110,339 |
| Large | 1,024 | 512 | 834,563 |

Each hidden layer accepts 8–1,024 neurons. Training accepts 1–50 supervised epochs
and a learning rate from 0.00001 to 0.01. Defaults remain the small model, 12 epochs
and a 0.001 learning rate; replay adaptation uses one fifth of that learning rate.
The UI reports the actual architecture and parameter count. To keep the visualization
readable, it displays up to the first 64 activations in layer 1 and 32 in layer 2,
with sampled counts labeled. All neurons participate in training.

The raw inputs retain the existing feature order. Normalization statistics come only
from the supervised prefix. Export folds those statistics into float64 first-layer
weights. The 64/32 model retains the `gate-npz-v1` format supported by the existing
NumPy inference reader. Other widths produce separate `training-mlp-v1` educational
artifacts with explicit layer sizes; the live gate cannot load them.

The recording is divided by frame position before fitting: 50% supervised, 30% replay
adaptation, 20% final holdout. Windows crossing boundaries are excluded, as are the
first 30 frames after each boundary because features depend on earlier price history.
Missing seconds exclude windows; no synthetic books or interpolated prices are inserted.
The run requires at least 30 supervised examples, 15 causal RL decisions and 15 holdout
examples after these checks. It fails if the recording falls short.

During supervised epochs, soft targets favor experts whose proxy signals align with
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
uv sync --extra dev --extra training --extra cloud
# Use a new output path for a new capture. The bundled recording is already complete.
uv run python scripts/record_binance.py --symbol BTCUSDT --seconds 900 \
  --book-interval-ms 100 --max-events 500000 --max-bytes 100000000 \
  --output data/my-new-capture.jsonl

uv run python scripts/run_training_lab.py --input data/training-public.jsonl \
  --output artifacts/my-local-training --pace 0.25
```

The browser starts with `data/training-public.jsonl`. **Data for your next run**
shows the selected recording's public source, event counts, time span and SHA-256.
Choose a duration from 30 to 1,800 seconds and capture a new range starting now.
This uses `scripts/capture_training_data.py` and the public Binance recorder; it
cannot reconstruct historical best bid/ask data from candles or trades.

Captures use immutable files under `data/training-captures/`. Only a completed
capture with enough contiguous windows for the actual training split becomes the
next selected dataset. Short, interrupted or failed captures leave the previous
selection intact. Stop waits for connection shutdown or validation to finish;
elapsed time includes this cleanup. A 900-second capture is the working example.
A run already in progress keeps its original recording.

Each training run creates a new directory under `artifacts/training-runs/`.
Selecting Modal sends the selected dataset and chosen options to the cloud.
The browser cannot select arbitrary files.
One OS lock prevents concurrent local and Modal training writers. Stop signals the
owned runner; a Modal runner also cancels its remote function call when one has started.
Server shutdown also stops a child it owns. The UI labels completed
runs and clears unavailable telemetry. Reloading a page does not start training.

## Set up Modal in the UI

Create an account at [Modal](https://modal.com/) and obtain a token ID and token secret
from your Modal settings. The [Starter plan](https://modal.com/pricing) includes $30
in free compute credits per month, as checked on September 5, 2026.

Select Modal in the training page and open its credential settings. Enter
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the masked fields and save. The server saves
these two entries to `.env` at the main checkout root, with owner-only `0600`
permissions, and preserves other entries. Saved credentials are never echoed back to
the page; it reports whether they are configured. You can also enter the two variables
directly in that root `.env` file.

Choose a model preset or custom widths, epochs and learning rate, then start the Modal
run. Larger models still use the fixed CPU resources below; selecting Large does not
request a GPU or more memory.

## One bounded Modal run from the CLI

```sh
# Prints the exact source/input hashes and resource plan without contacting Modal.
uv run python scripts/train_lab_on_modal.py \
  --output artifacts/my-modal-training

# Execute one run using the two local credentials, after reviewing the plan.
uv run python scripts/train_lab_on_modal.py --run \
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
records the actual app/function-call IDs. Custom-width outputs are named
`training-mlp-supervised.npz` and `training-mlp-adapted.npz`; the default 64/32 model
keeps `gate-supervised.npz` and `gate-adapted.npz`. Runtime model promotion is a separate
decision.

## Progress and compute estimates

The progress bar counts actual supervised and replay optimizer updates. It starts
indeterminate while the dataset and cloud image are prepared, remains below 100%
during holdout evaluation and artifact transfer, and reaches 100% only after the
checkpoints are saved. It is a fraction of updates, not a prediction of time remaining.

At the [Modal rates](https://modal.com/pricing) checked September 5, 2026, two physical
CPU cores plus 2 GiB cost approximately $0.00184 of compute credits per minute.
The planner lets you supply an assumed duration. During a remote run, the page
also estimates compute from observed elapsed time after execution starts. Image
build, startup and other charges are excluded; these numbers are neither billed
usage nor your account balance. Larger models can take longer on the same resources.

The [progress and data validation receipt](../artifacts/training-example/progress-data-experts-validation.json)
records a real Modal run ending at 44/44 updates, plus a deliberately short public
capture that correctly retained the existing trainable dataset.

## Which experts are trained?

The original three experts are rules: microprice imbalance, signed trade flow and
mean reversion. Their parameters are not learned by the Training Lab. The larger
network learns how to weight these three rule signals.

The terminal also shows a separately trained **tiny neural expert** alongside them:
three current rule scores → eight tanh neurons → one bounded directional score.
Its 41 learned parameters come from chronological public-data training against a
five-second future mid-price target. It is a shadow demonstration; its output does
not enter the three-expert mixture or change quotes. Reproduce it with:

```sh
uv run python scripts/train_tiny_expert.py --help
```

The [model receipt](../models/tiny-expert-demo.json) records the source and dataset
hashes, split embargo, optimizer, parameter changes and measured inference timing.
Training preserves recording arrival order when receive timestamps tie, matching
the live engine. Local warm NumPy inference measured 6.50 µs median and 6.75 µs p95
across 5,000 calls; this excludes engine and networking work. The card displays its
own observed inference time. The holdout target has zero variance and a zero
predictor beats the trained network, so this is a training/latency example with no
evidence of predictive value.

The live heatmap adds columns from actual distinct decision events, retaining the
latest 64. Green means buy bias, red means sell bias, and intensity shows the bounded
signal magnitude. These are not calibrated confidence probabilities. The browser
receives snapshots at 10 Hz and can add multiple event columns per snapshot.
