from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from configs.train_config import LogisticRegressionConfig, get_logistic_regression_config


class LogisticClassifier:
    """Logistic regression classifier with optional class balancing."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        """Initialize LogisticClassifier."""
        self.config = config or get_logistic_regression_config()
        self.model = LogisticRegression(**self.config.model_params)
        self.report: dict | None = None
        self.threshold: float = 0.5

    def fit(self, X_train, y_train, X_test=None, y_test=None) -> LogisticClassifier:
        """Fit logistic regression model."""
        self.model.fit(X_train, np.asarray(y_train, dtype=int))

        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            self.report = classification_report(
                np.asarray(y_test, dtype=int),
                y_pred,
                output_dict=True,
                zero_division=0,
            )
        return self

    def save(self, threshold: float | None = None) -> None:
        if threshold is not None:
            self.threshold = float(threshold)
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.config.model_path)

    @classmethod
    def load(cls, config=None):
        loaded = joblib.load((config or get_logistic_regression_config()).model_path)
        if config is not None:
            loaded.config = config
        return loaded

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
