from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

class EvaluationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float