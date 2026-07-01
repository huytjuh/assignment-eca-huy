from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from configs.global_config import MonitorConfig, get_monitor_config
from src.emailurgency.schemas.monitor import HumanReviewItem, MonitoringAlert, MonitoringSnapshot

class ModelMonitor:
    """Monitor model drift and build a human review queue for low-confidence or drifting cases."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or get_monitor_config()
        self.baseline_label_distribution: dict[str, float] | None = None
        self.baseline_topic_distribution: dict[str, float] | None = None
        self.baseline_mean_confidence: float | None = None

    def set_baseline(
        self,
        predictions: np.ndarray | list[int],
        probabilities: np.ndarray | list[float],
        topics: np.ndarray | list[int] | None = None,
        topic_probabilities: np.ndarray | list[float] | None = None,
    ) -> None:
        """Store a reference distribution for later drift checks."""
        preds = np.asarray(predictions, dtype=int)
        probs = np.asarray(probabilities, dtype=float)
        self.baseline_label_distribution = self._distribution(preds)

        if topics is not None:
            self.baseline_topic_distribution = self._distribution(np.asarray(topics, dtype=int))
        elif topic_probabilities is not None:
            self.baseline_topic_distribution = self._distribution_from_probabilities(np.asarray(topic_probabilities, dtype=float))
        else:
            self.baseline_topic_distribution = None

        self.baseline_mean_confidence = float(np.mean(probs)) if probs.size else 0.0

    def monitor(
        self,
        predictions: np.ndarray | list[int],
        probabilities: np.ndarray | list[float],
        topics: np.ndarray | list[int] | None = None,
        topic_probabilities: np.ndarray | list[float] | None = None,
        timestamp: str | None = None,
    ) -> MonitoringSnapshot:
        """Compute drift signals for a new batch of predictions."""
        if self.baseline_label_distribution is None:
            raise ValueError("Call set_baseline before monitoring")

        preds = np.asarray(predictions, dtype=int)
        probs = np.asarray(probabilities, dtype=float)
        sample_size = int(len(preds))

        if sample_size < self.config.min_samples:
            raise ValueError(f"Need at least {self.config.min_samples} samples for monitoring")

        current_label_distribution = self._distribution(preds)
        label_drift = self._distribution_distance(current_label_distribution, self.baseline_label_distribution)

        topic_distribution = self._topic_distribution(topics, topic_probabilities, preds)
        topic_drift = 0.0
        if self.baseline_topic_distribution is not None and topic_distribution:
            topic_drift = self._distribution_distance(topic_distribution, self.baseline_topic_distribution)

        mean_confidence = float(np.mean(probs)) if probs.size else 0.0
        confidence_drift = abs(mean_confidence - (self.baseline_mean_confidence or 0.0))

        uncertainty_rate = float(np.mean(probs < self.config.uncertainty_threshold)) if probs.size else 0.0

        alerts = self._build_alerts(label_drift, topic_drift, confidence_drift, uncertainty_rate)
        return MonitoringSnapshot(
            timestamp=timestamp or pd.Timestamp.utcnow().isoformat(),
            sample_size=sample_size,
            label_distribution={k: float(v) for k, v in current_label_distribution.items()},
            topic_distribution={k: float(v) for k, v in topic_distribution.items()},
            mean_confidence=mean_confidence,
            confidence_drift_score=confidence_drift,
            label_drift_score=label_drift,
            topic_drift_score=topic_drift,
            uncertainty_rate=uncertainty_rate,
            alerts=alerts,
            review_queue_size=0,
        )

    def build_review_queue(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray | list[int],
        probabilities: np.ndarray | list[float],
        topics: np.ndarray | list[int] | None = None,
        topic_probabilities: np.ndarray | list[float] | None = None,
        text_column: str = "body",
        row_id_column: str = "message_id",
        top_n: int | None = None,
    ) -> list[HumanReviewItem]:
        """Return the most suspicious samples for human review."""
        if df.empty:
            return []

        preds = np.asarray(predictions, dtype=int)
        probs = np.asarray(probabilities, dtype=float)
        if len(df) != len(preds) or len(df) != len(probs):
            raise ValueError("DataFrame and prediction arrays must have the same length")

        review_mask = probs < self.config.review_confidence_threshold
        if topics is not None:
            topic_array = np.asarray(topics, dtype=int)
            review_mask |= np.isin(topic_array, self._rare_topics(topic_array))

        if topic_probabilities is not None:
            topic_probs = np.asarray(topic_probabilities, dtype=float)
            review_mask |= topic_probs < 0.2

        selected = df.loc[review_mask].copy()
        if selected.empty:
            return []

        selected = selected.head(top_n or self.config.max_review_items)
        review_items: list[HumanReviewItem] = []
        for idx, row in selected.iterrows():
            pred_idx = int(preds[idx]) if idx in range(len(preds)) else None
            prob = float(probs[idx]) if idx in range(len(probs)) else None
            topic_value = None
            topic_prob_value = None
            if topics is not None and idx in range(len(topics)):
                topic_value = int(topics[idx])
            if topic_probabilities is not None and idx in range(len(topic_probabilities)):
                topic_prob_value = float(topic_probabilities[idx])

            reason_parts = ["low confidence"] if prob is not None and prob < self.config.review_confidence_threshold else []
            if topic_prob_value is not None and topic_prob_value < 0.2:
                reason_parts.append("uncertain topic")
            if topic_value is not None and topic_value in self._rare_topics(np.asarray(topics, dtype=int)):
                reason_parts.append("rare topic")

            review_items.append(
                HumanReviewItem(
                    row_id=row[row_id_column] if row_id_column in row.index else idx,
                    text=str(row[text_column]) if text_column in row.index else "",
                    predicted_label=pred_idx,
                    predicted_probability=prob,
                    topic=topic_value,
                    topic_probability=topic_prob_value,
                    reason="; ".join(reason_parts) or "manual review recommended",
                    drift_score=float(prob if prob is not None else 0.0),
                )
            )

        return review_items

    def _build_alerts(
        self,
        label_drift: float,
        topic_drift: float,
        confidence_drift: float,
        uncertainty_rate: float,
    ) -> list[MonitoringAlert]:
        alerts: list[MonitoringAlert] = []
        if label_drift > self.config.label_drift_threshold:
            alerts.append(
                MonitoringAlert(
                    name="label_drift",
                    severity="high" if label_drift > self.config.label_drift_threshold * 1.5 else "medium",
                    score=label_drift,
                    threshold=self.config.label_drift_threshold,
                    message="Predicted label distribution moved materially from the baseline.",
                )
            )
        if topic_drift > self.config.topic_drift_threshold:
            alerts.append(
                MonitoringAlert(
                    name="topic_drift",
                    severity="high" if topic_drift > self.config.topic_drift_threshold * 1.5 else "medium",
                    score=topic_drift,
                    threshold=self.config.topic_drift_threshold,
                    message="Topic distribution drifted away from the baseline BERTTopic profile.",
                )
            )
        if confidence_drift > self.config.confidence_drift_threshold:
            alerts.append(
                MonitoringAlert(
                    name="confidence_drift",
                    severity="medium",
                    score=confidence_drift,
                    threshold=self.config.confidence_drift_threshold,
                    message="Mean model confidence shifted noticeably compared with the baseline.",
                )
            )
        if uncertainty_rate > self.config.uncertainty_threshold:
            alerts.append(
                MonitoringAlert(
                    name="uncertainty_spike",
                    severity="high",
                    score=uncertainty_rate,
                    threshold=self.config.uncertainty_threshold,
                    message="A large share of samples fell below the confidence threshold.",
                )
            )
        return alerts

    def _distribution(self, values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {}
        labels, counts = np.unique(values, return_counts=True)
        total = counts.sum()
        return {str(int(label)): float(count / total) for label, count in zip(labels, counts)}

    def _distribution_from_probabilities(self, values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {}
        buckets = np.round(values, 2)
        labels, counts = np.unique(buckets, return_counts=True)
        total = counts.sum()
        return {str(float(label)): float(count / total) for label, count in zip(labels, counts)}

    def _distribution_distance(self, current: dict[str, float], baseline: dict[str, float]) -> float:
        all_keys = sorted(set(current) | set(baseline))
        if not all_keys:
            return 0.0
        current_vector = np.array([current.get(k, 0.0) for k in all_keys], dtype=float)
        baseline_vector = np.array([baseline.get(k, 0.0) for k in all_keys], dtype=float)
        return float(np.linalg.norm(current_vector - baseline_vector))

    def _topic_distribution(
        self,
        topics: np.ndarray | list[int] | None,
        topic_probabilities: np.ndarray | list[float] | None,
        predictions: np.ndarray | list[int],
    ) -> dict[str, float]:
        if topics is not None:
            return self._distribution(np.asarray(topics, dtype=int))
        if topic_probabilities is not None:
            return self._distribution_from_probabilities(np.asarray(topic_probabilities, dtype=float))
        return {str(int(label)): float(count / len(predictions)) for label, count in zip(np.unique(np.asarray(predictions, dtype=int)), np.bincount(np.asarray(predictions, dtype=int)))}

    def _rare_topics(self, topics: np.ndarray) -> set[int]:
        if topics.size == 0:
            return set()
        counts = pd.Series(topics).value_counts()
        threshold = max(2, int(len(topics) * 0.05))
        return set(counts[counts < threshold].index.astype(int).tolist())
