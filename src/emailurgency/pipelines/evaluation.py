from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_curve

from src.emailurgency.schemas.eval import EvaluationResult


class Evaluation:
    @staticmethod
    def compute(y_true, y_proba, y_pred, positive_label: int = 1) -> EvaluationResult:
        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        y_pred = np.asarray(y_pred, dtype=int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        labels = np.unique(y_true)
        auc_roc = 0.0
        if len(labels) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=positive_label)
            auc_roc = auc(fpr, tpr)

        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auc_roc=float(auc_roc),
        )

    @staticmethod
    def roc_curve_df(y_true, y_proba) -> pd.DataFrame:
        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
