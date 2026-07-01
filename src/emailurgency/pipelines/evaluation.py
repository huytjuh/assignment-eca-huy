from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_curve

from emailurgency.schemas.eval import EvaluationResult

from src.emailurgency.schemas.eval import EvaluationResult

class Evaluation:

    @staticmethod
    def compute(y_true, y_proba, y_pred, positive_label: int = 1) -> EvaluationResult:
        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        y_pred = np.asarray(y_pred, dtype=int)

        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=positive_label)
        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            auc_roc=float(auc(fpr, tpr)),
            f1=float(2 * (precision_score(y_true, y_pred, zero_division=0) * recall_score(y_true, y_pred, zero_division=0)) / (precision_score(y_true, y_pred, zero_division=0) + recall_score(y_true, y_pred, zero_division=0)))
        )

    @staticmethod
    def roc_curve_df(y_true, y_proba) -> pd.DataFrame:
        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": thresholds}
        )