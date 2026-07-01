from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest


@dataclass
class Snapshot:
    def model_dump_json(self, indent=None):
        return '{"status": "ok"}'


@dataclass
class ReviewItem:
    row_id: str = "m2"
    text: str = "urgent email"
    predicted_label: int = 1
    predicted_probability: float = 0.7
    topic: int = 0
    topic_probability: float = 0.8
    reason: str = "low confidence"
    drift_score: float = 0.2

    def model_dump(self):
        return self.__dict__


def monitor_config(**overrides):
    defaults = {
        "label_drift_threshold": 0.1,
        "topic_drift_threshold": 0.1,
        "confidence_drift_threshold": 0.1,
        "uncertainty_threshold": 0.4,
        "review_confidence_threshold": 0.65,
        "min_samples": 2,
        "max_review_items": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_model_monitor_detects_drift_and_builds_review_queue():
    from src.emailurgency.pipelines.monitoring import ModelMonitor

    monitor = ModelMonitor(config=monitor_config())
    monitor.set_baseline(predictions=[0, 0], probabilities=[0.9, 0.8], topics=[0, 0])
    snapshot = monitor.monitor(predictions=[1, 1], probabilities=[0.3, 0.2], topics=[1, 1])
    queue = monitor.build_review_queue(
        pd.DataFrame({"message_id": ["m1", "m2"], "body": ["a", "b"]}),
        predictions=[1, 1],
        probabilities=[0.3, 0.7],
        topics=[1, 2],
        topic_probabilities=[0.9, 0.1],
    )

    assert snapshot.label_drift_score > 0
    assert {alert.name for alert in snapshot.alerts} >= {"label_drift", "confidence_drift"}
    assert [item.row_id for item in queue] == ["m1", "m2"]


def test_run_monitor_rejects_missing_probability_columns(monkeypatch, tmp_path):
    import scripts.run_monitor as run_monitor

    input_path = tmp_path / "eval.parquet"
    input_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(run_monitor, "parse_args", lambda: SimpleNamespace(input=input_path, output=tmp_path / "snapshot.json", review_output=tmp_path / "review.csv", baseline=None))
    monkeypatch.setattr(run_monitor, "get_monitor_config", lambda: monitor_config(min_samples=1))
    monkeypatch.setattr(run_monitor.pd, "read_parquet", lambda path, engine=None: pd.DataFrame({"pred": [1]}))

    with pytest.raises(ValueError, match="proba.*pred"):
        run_monitor.main()


def test_run_monitor_writes_snapshot_and_review_queue(monkeypatch, tmp_path):
    import scripts.run_monitor as run_monitor

    input_path = tmp_path / "eval.parquet"
    output_path = tmp_path / "snapshot.json"
    review_path = tmp_path / "review.csv"
    input_path.write_text("placeholder", encoding="utf-8")
    calls = {}

    class FakeModelMonitor:
        def __init__(self, config):
            calls["config"] = config

        def set_baseline(self, **kwargs):
            calls["baseline"] = kwargs

        def monitor(self, **kwargs):
            calls["monitor"] = kwargs
            return Snapshot()

        def build_review_queue(self, **kwargs):
            calls["review"] = kwargs
            return [ReviewItem()]

    df = pd.DataFrame(
        {
            "message_id": ["m1", "m2"],
            "body": ["normal", "urgent email"],
            "pred": [0, 1],
            "proba": [0.2, 0.7],
            "topic": [0, 1],
            "topic_probability": [0.8, 0.9],
        }
    )

    monkeypatch.setattr(run_monitor, "parse_args", lambda: SimpleNamespace(input=input_path, output=output_path, review_output=review_path, baseline=None))
    monkeypatch.setattr(run_monitor, "get_monitor_config", lambda: monitor_config(min_samples=1))
    monkeypatch.setattr(run_monitor, "ModelMonitor", FakeModelMonitor)
    monkeypatch.setattr(run_monitor.pd, "read_parquet", lambda path, engine=None: df.copy())

    run_monitor.main()

    assert output_path.read_text(encoding="utf-8") == '{"status": "ok"}'
    assert "row_id" in review_path.read_text(encoding="utf-8")
    assert calls["baseline"]["predictions"].tolist() == [0, 1]
    assert calls["monitor"]["probabilities"].tolist() == [0.2, 0.7]