"""Recompute the frozen Phase 2 data gate without recording or training anything."""

import json
from pathlib import Path

from market_gate.training import build_examples, build_frame_dataset, file_sha256, load_recording


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "artifacts/phase2-summary.json").read_text())
    registration = root / report["preregistration_path"]
    policy = json.loads(registration.read_text())
    assert file_sha256(registration) == report["preregistration_sha256"]
    assert len(report["samples"]) == len(policy["windows"]) == 3
    for sample, window in zip(report["samples"], policy["windows"], strict=True):
        path = root / sample["input_path"]
        assert sample["id"] == window["id"]
        assert sample["scheduled_start"] == window["start_at"]
        assert file_sha256(path) == sample["input_sha256"]
        events = load_recording(path)
        dataset = build_frame_dataset(events)
        examples = build_examples(
            dataset.values,
            dataset.mids,
            policy["lookback_frames"],
            policy["horizon_frames"],
            close_ts_ms=dataset.close_ts_ms,
        )
        candidates = len(dataset.values) - policy["lookback_frames"] - policy["horizon_frames"] + 1
        boundary = int(candidates * 0.8)
        train = sum(e.target_frame < boundary for e in examples)
        validation = sum(e.start_frame >= boundary for e in examples)
        computed = {
            "recorded_events": len(events),
            "frames": len(dataset.values),
            "candidate_examples": candidates,
            "gap_excluded_examples": candidates - len(examples),
            "boundary_excluded_examples": len(examples) - train - validation,
            "train_examples": train,
            "validation_examples": validation,
        }
        assert all(sample[key] == value for key, value in computed.items()), sample["id"]
        assert (
            train < policy["minimum_train_examples"]
            or validation < policy["minimum_validation_examples"]
        )
        assert sample["status"] == "rejected" and sample["models_trained"] == 0
        print(f"{sample['id']}: {train} train / {validation} validation; rejected; hash verified")
    assert report["accepted_samples"] == report["experimental_models_trained"] == 0
    assert report["replacement_captures"] == 0
    assert report["cloud"]["status"] == "not_run"
    print(
        "Frozen report matches the recordings and acceptance policy. No captures or fits performed."
    )


if __name__ == "__main__":
    main()
