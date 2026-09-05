import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from market_gate.contracts import BookEvent, TradeEvent  # noqa: E402
from market_gate.gate import NumpyMLPGate  # noqa: E402
from market_gate.training import (  # noqa: E402
    build_examples,
    build_frame_dataset,
    load_recording,
    write_recording,
)
from market_gate.training_lab import (  # noqa: E402
    _evaluate,
    _export,
    _make_model,
    split_lab_examples,
    train_lab,
)


def recording(path: Path, *, future_from: int | None = None):
    events = []
    for second in range(650):
        timestamp = second * 1000
        mid = 100 + 0.05 * np.sin(second / 9) + 0.001 * second
        if future_from is not None and second >= future_from:
            mid += 0.1 * np.sin(second / 3)
        events.extend(
            [
                BookEvent(
                    "binance",
                    "BTCUSDT",
                    timestamp,
                    timestamp + 1,
                    second,
                    mid - 0.005,
                    1 + second % 3,
                    mid + 0.005,
                    1 + second % 5,
                ),
                TradeEvent(
                    "binance",
                    "BTCUSDT",
                    timestamp + 200,
                    timestamp + 201,
                    second,
                    mid,
                    0.1 + second % 4,
                    "buy" if second % 3 else "sell",
                ),
            ]
        )
    write_recording(path, events, 1)
    return path


def examples(path):
    frames = build_frame_dataset(load_recording(path))
    return build_examples(frames.values, frames.mids, 30, 5, close_ts_ms=frames.close_ts_ms), frames


def test_split_excludes_feature_history_and_waits_for_rewards(tmp_path):
    rows, frames = examples(recording(tmp_path / "data.jsonl"))
    supervised, rl, holdout = split_lab_examples(rows, len(frames.values), 120)
    assert supervised[-1].target_frame < rl[0].start_frame - 30
    assert rl[-1].target_frame < holdout[0].start_frame - 30
    assert all(a.target_ts_ms <= b.input_end_ts_ms for a, b in zip(rl, rl[1:]))
    assert all(e.input_end_ts_ms < e.target_ts_ms for e in rl)
    with pytest.raises(ValueError, match="insufficient contiguous"):
        split_lab_examples(rows[:100], 150, 120)


def test_exports_preserve_raw_numpy_inference_and_evaluation_does_not_update(tmp_path):
    torch.manual_seed(8)
    model = _make_model()
    mean = torch.linspace(-0.2, 0.3, 300)
    scale = torch.linspace(0.01, 2, 300)
    raw = torch.linspace(-0.3, 0.5, 300)
    expected = torch.softmax(model((raw - mean) / scale), dim=-1).detach().numpy()
    path = tmp_path / "gate.npz"
    _export(model, mean, scale, path)
    actual = np.array(list(NumpyMLPGate(path).predict(raw.reshape(30, 10)).values()))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert np.isfinite(_evaluate(model, ((raw - mean) / scale)[None], torch.ones(1, 3)))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_real_training_updates_weights_and_keeps_holdout_frozen(tmp_path):
    source = recording(tmp_path / "data.jsonl")
    events = list(train_lab(source, tmp_path / "first", epochs=2, max_rl_steps=15))
    steps = [e["step"] for e in events if e["kind"] == "step"]
    assert [e["kind"] for e in events][0] == "dataset"
    assert len(steps) == 17
    assert {s["phase"] for s in steps} == {"supervised", "rl"}
    for step in steps:
        assert np.asarray(step["features"]).shape == (30, 10)
        assert len(step["activations"]["hidden_1"]) == 64
        assert len(step["activations"]["hidden_2"]) == 32
        assert step["target_ts_ms"] > step["input_end_ts_ms"]
        assert np.isclose(sum(step["outputs_before"]), 1)
        assert np.isclose(sum(step["outputs_after"]), 1)
    for phase in ("supervised", "rl"):
        assert any(
            layer["gradient_norm"] > 0 and layer["weight_delta_norm"] > 0
            for step in steps
            if step["phase"] == phase
            for layer in step["layers"]
        )
    completed = events[-1]
    assert completed["kind"] == "completed"
    assert set(completed["evaluation"]) == {
        "supervised",
        "adapted",
        "uniform",
        "microprice",
        "trade_flow",
        "reversion",
    }
    assert completed["parameter_count"] == 21443
    assert (
        completed["artifacts"]["supervised"]["sha256"]
        != completed["artifacts"]["adapted"]["sha256"]
    )
    json.dumps(events, allow_nan=False)
    # Changing only the held-out future must not alter either exported policy.
    changed = recording(tmp_path / "changed.jsonl", future_from=560)
    second = list(train_lab(changed, tmp_path / "second", epochs=2, max_rl_steps=15))
    for phase in ("supervised", "adapted"):
        first_gate = np.load(completed["artifacts"][phase]["path"])
        second_gate = np.load(second[-1]["artifacts"][phase]["path"])
        assert all(np.array_equal(first_gate[name], second_gate[name]) for name in first_gate.files)
    assert completed["evaluation"] != second[-1]["evaluation"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        list(train_lab(source, tmp_path / "first", epochs=2, max_rl_steps=15))


def test_future_reward_cannot_change_sampled_action_or_input(tmp_path):
    source = recording(tmp_path / "data.jsonl")
    rows, frames = examples(source)
    _, rl, _ = split_lab_examples(rows, len(frames.values), 15)
    future_second = rl[0].target_ts_ms // 1000
    changed = recording(tmp_path / "changed.jsonl", future_from=future_second)
    first = list(train_lab(source, tmp_path / "first", epochs=1, max_rl_steps=15))
    second = list(train_lab(changed, tmp_path / "second", epochs=1, max_rl_steps=15))
    a = next(e["step"] for e in first if e["kind"] == "step" and e["step"]["phase"] == "rl")
    b = next(e["step"] for e in second if e["kind"] == "step" and e["step"]["phase"] == "rl")
    assert a["features"] == b["features"]
    assert a["outputs_before"] == b["outputs_before"]
    assert a["action"] == b["action"]
    assert a["reward"] != b["reward"]


def test_trained_export_matches_with_constant_and_low_variance_columns(tmp_path, monkeypatch):
    import market_gate.training_lab as lab

    source = recording(tmp_path / "data.jsonl")
    rows, _ = examples(source)
    raw = torch.from_numpy(np.stack([row.features for row in rows]))
    original = lab._export
    errors = []

    def checked_export(model, mean, scale, destination):
        assert int((scale == 1e-6).sum()) > 0
        original(model, mean, scale, destination)
        expected = torch.softmax(model((raw - mean) / scale), dim=-1).detach().numpy()
        gate = NumpyMLPGate(destination)
        actual = np.asarray([list(gate.predict(row.reshape(30, 10)).values()) for row in raw])
        errors.append(float(np.max(np.abs(expected - actual))))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        assert np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))

    monkeypatch.setattr(lab, "_export", checked_export)
    list(train_lab(source, tmp_path / "run", epochs=12, max_rl_steps=15))
    assert len(errors) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": 51},
        {"epochs": True},
        {"max_rl_steps": 14},
        {"max_rl_steps": 301},
        {"seed": -1},
        {"hidden_1": 7},
        {"hidden_2": 1025},
        {"hidden_1": True},
        {"hidden_2": 32.0},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": True},
        {"learning_rate": 0.011},
    ],
)
def test_hard_limits_reject_before_reading_or_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        list(train_lab(tmp_path / "absent", tmp_path / "out", **kwargs))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("widths", [(8, 16), (128, 64)])
def test_custom_architecture_trains_and_exports_separate_artifacts(tmp_path, widths):
    from market_gate.training_options import TrainingOptions

    options = TrainingOptions(hidden_1=widths[0], hidden_2=widths[1], learning_rate=0.002)
    model = _make_model(*widths)
    assert sum(p.numel() for p in model.parameters()) == options.parameter_count
    source = recording(tmp_path / "data.jsonl")
    events = list(
        train_lab(
            source,
            tmp_path / "run",
            hidden_1=widths[0],
            hidden_2=widths[1],
            learning_rate=options.learning_rate,
            epochs=2,
            max_rl_steps=15,
        )
    )
    dataset, completed = events[0]["dataset"], events[-1]
    assert dataset["hidden_sizes"] == list(widths)
    assert dataset["activation_sample_sizes"] == [min(widths[0], 64), min(widths[1], 32)]
    assert dataset["parameter_count"] == completed["parameter_count"] == options.parameter_count
    assert dataset["epochs"] == 2
    assert dataset["learning_rate"] == options.learning_rate
    assert completed["config"]["rl_learning_rate"] == options.learning_rate * 0.2
    assert completed["architecture"]["layer_sizes"] == [300, *widths, 3]
    for event in events:
        if event["kind"] == "step":
            assert [len(v) for v in event["step"]["activations"].values()] == dataset[
                "activation_sample_sizes"
            ]
    assert any(
        layer["gradient_norm"] > 0 and layer["weight_delta_norm"] > 0
        for event in events
        if event["kind"] == "step"
        for layer in event["step"]["layers"]
    )
    for artifact in completed["artifacts"].values():
        assert artifact["schema"] == "training-mlp-v1"
        assert artifact["runtime_compatible"] is False
        with np.load(artifact["path"]) as exported:
            assert exported["schema_version"].item() == "training-mlp-v1"
            assert exported["layer_sizes"].tolist() == [300, *widths, 3]
            assert exported["w1"].shape == (300, widths[0])
            assert exported["w2"].shape == widths
            assert exported["w3"].shape == (widths[1], 3)
            for name in ("w1", "w2", "w3", "b1", "b2", "b3"):
                assert exported[name].dtype == np.float64
                assert np.isfinite(exported[name]).all()
        with pytest.raises(ValueError, match="unsupported gate artifact"):
            NumpyMLPGate(artifact["path"])


def test_custom_export_preserves_raw_input_inference(tmp_path):
    torch.manual_seed(8)
    model = _make_model(128, 64)
    mean = torch.linspace(-0.2, 0.3, 300)
    scale = torch.linspace(0.01, 2, 300)
    raw = torch.linspace(-0.3, 0.5, 300)
    expected = torch.softmax(model((raw - mean) / scale), dim=-1).detach().numpy()
    path = tmp_path / "teaching.npz"
    _export(model, mean, scale, path)
    with np.load(path) as exported:
        hidden_1 = np.maximum(raw.numpy() @ exported["w1"] + exported["b1"], 0)
        hidden_2 = np.maximum(hidden_1 @ exported["w2"] + exported["b2"], 0)
        logits = hidden_2 @ exported["w3"] + exported["b3"]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
    np.testing.assert_allclose(probabilities, expected, rtol=2e-5, atol=2e-6)
