from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MonitoringAlert(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    severity: str
    score: float
    threshold: float
    message: str


class MonitoringSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    timestamp: str
    sample_size: int
    label_distribution: dict[str, float] = Field(default_factory=dict)
    topic_distribution: dict[str, float] = Field(default_factory=dict)
    mean_confidence: float = 0.0
    confidence_drift_score: float = 0.0
    label_drift_score: float = 0.0
    topic_drift_score: float = 0.0
    uncertainty_rate: float = 0.0
    alerts: list[MonitoringAlert] = Field(default_factory=list)
    review_queue_size: int = 0


class HumanReviewItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    row_id: Any
    text: str
    predicted_label: int | None = None
    predicted_probability: float | None = None
    topic: int | None = None
    topic_probability: float | None = None
    reason: str
    drift_score: float = 0.0