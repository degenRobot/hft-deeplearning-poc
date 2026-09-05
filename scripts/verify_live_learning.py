"""Independently reconstruct the first <=40 live head updates from public telemetry.

Run against a fresh enabled run before its bounded history rolls over.
This read-only verifier also saves a small validation receipt; it never toggles RL.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

base = "http://127.0.0.1:8001"


def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.load(r)


snapshot = get("/learning/training")
status = snapshot["learning"]
health = get("/health")
assert snapshot["history"][0]["step"] == 1
with np.load("models/gate-demo.npz") as model:
    weights = {key: model[key].copy(order="K") for key in ["w1", "b1", "w2", "b2", "w3", "b3"]}
original = {key: value.copy() for key, value in weights.items()}


def forward(features):
    a = np.maximum(features.reshape(-1) @ weights["w1"] + weights["b1"], 0)
    b = np.maximum(a @ weights["w2"] + weights["b2"], 0)
    logits = b @ weights["w3"] + weights["b3"]
    e = np.exp(np.clip(logits - logits.max(), -50, 50))
    return e / e.sum(), a, b


baseline = 0.0
for step in snapshot["history"]:
    x = np.array(step["features"], dtype=np.float32)
    p, a, b = forward(x)
    np.testing.assert_array_equal(p, step["outputs_before"])
    np.testing.assert_array_equal(a, step["activations"]["hidden_1"])
    np.testing.assert_array_equal(b, step["activations"]["hidden_2"])
    onehot = np.zeros(3, dtype=np.float32)
    onehot[step["action"]] = 1
    d = (p - onehot) * ((step["reward"] - baseline) / 5)
    dw = b[:, None] * d[None, :]
    # Match the runtime scalar type so clipping retains float32 head arithmetic.
    norm = float(np.sqrt(np.sum(dw.astype(float) ** 2) + np.sum(d.astype(float) ** 2)))
    scale = min(1, 1 / max(norm, 1e-12))
    weights["w3"] = weights["w3"] - 0.001 * scale * dw
    weights["b3"] = weights["b3"] - 0.001 * scale * d
    np.testing.assert_array_equal(forward(x)[0], step["outputs_after"])
    baseline = 0.9 * baseline + 0.1 * step["reward"]
    assert step["target_ts_ms"] >= step["selected_ts_ms"] + 5000
    assert all(layer["weight_delta_norm"] == 0 for layer in step["layers"][:2])
computed = hashlib.sha256(weights["w3"].tobytes() + weights["b3"].tobytes()).hexdigest()
assert computed == status["head_sha256"]
assert status["updates"] == len(snapshot["history"])
assert health["source"] == "binance" and health["health"]["ready"]
assert health["gate"]["model_version"] == status["model_version"]
receipt = {
    "schema_version": 1,
    "source": "binance_public_websocket",
    "symbol": health["symbol"],
    "run_id": status["run_id"],
    "verified_steps": status["updates"],
    "base_head_sha256": hashlib.sha256(
        original["w3"].tobytes() + original["b3"].tobytes()
    ).hexdigest(),
    "adapted_head_sha256": computed,
    "independent_head_reconstruction": "byte_identical",
    "activations_and_before_after_probabilities": "exact_for_every_step",
    "hidden_layers": "unchanged",
    "delayed_outcomes": "all_at_least_5_seconds_after_selection",
    "last_step_duration_ms": status["step_duration_ms"],
    "events_processed": health["health"]["events_processed"],
    "book_age_ms": health["health"]["book_age_ms"],
    "paused_at_verification": not status["enabled"],
    "reward": "clipped_directional_mid_price_proxy_not_pnl",
    "execution": "synthetic_quotes_only",
    "verification_command": "uv run python scripts/verify_live_learning.py",
}
Path("artifacts/training-example/live-rl-validation.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt, indent=2))
