from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from configs.global_config import FeatureConfig, get_feature_config
from src.emailurgency.models.vectordb import VectorDB
from src.emailurgency.schemas.features import MetaFeatures


class BaseFeatureExtraction(ABC):
    """Extract features from emails"""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or get_feature_config()

    @abstractmethod
    def extract(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from emails"""
        pass


class MetaFeatureExtraction(BaseFeatureExtraction):
    """Extract meta features from emails"""

    REQUIRED_COLUMNS = [
        "message_id", "sender", "date", "subject", "body",
        "to", "cc", "bcc", "sender_domain", "recipient_domain"
    ]

    def extract(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract meta features from emails"""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

        X = X.fillna('').sort_values('date')

        meta_features = pd.DataFrame({
            'message_id': X['message_id'],
            'response_time': self._response_time(X),
            'time_since_last_email': self._time_since_last_email(X),
            'sent_hour': self._sent_hour(X),
            'sent_weekday': self._sent_weekday(X),
            'is_after_hours': self._is_after_hours(X),
            'is_weekend': self._is_weekend(X),
            'n_recipient': self._n_recipient(X),
            'n_recipient_domains': self._n_recipient_domains(X),
            'n_words': self._n_words(X),
            'n_sentences': self._n_sentences(X),
            'is_internal': self._is_internal(X),
            'is_external': self._is_external(X)
        })

        validated = [MetaFeatures.model_validate(x) for x in meta_features.to_dict('records')]
        return pd.DataFrame([row.model_dump() for row in validated])

    def _response_time(self, X: pd.DataFrame) -> np.ndarray:
        return X['date'].diff().dt.total_seconds().astype(object).fillna(None).to_numpy()

    def _time_since_last_email(self, X: pd.DataFrame) -> np.ndarray:
        return X.groupby('sender')['date'].diff().dt.total_seconds().astype(object).fillna(None).to_numpy()

    def _sent_hour(self, X: pd.DataFrame) -> np.ndarray:
        return X['date'].dt.hour.to_numpy()

    def _sent_weekday(self, X: pd.DataFrame) -> np.ndarray:
        return X['date'].dt.weekday.to_numpy()

    def _is_after_hours(self, X: pd.DataFrame) -> np.ndarray:
        return (X['date'].dt.hour > self.config.after_hours[0]).to_numpy() | (X['date'].dt.hour < self.config.after_hours[1]).to_numpy()

    def _is_weekend(self, X: pd.DataFrame) -> np.ndarray:
        return (X['date'].dt.weekday > 4).to_numpy()

    def _n_recipient(self, X: pd.DataFrame) -> np.ndarray:
        return X.loc[:, ['to', 'cc', 'bcc']].stack().explode().dropna().groupby(level=0).nunique().reindex(X.index, fill_value=0).to_numpy()

    def _n_recipient_domains(self, X: pd.DataFrame) -> np.ndarray:
        return X['recipient_domain'].map(len).to_numpy()

    def _n_words(self, X: pd.DataFrame) -> np.ndarray:
        return X['body'].fillna('').str.count(r"\S+").to_numpy()

    def _n_sentences(self, X: pd.DataFrame) -> np.ndarray:
        return X['body'].fillna('').str.count(r"[.!?]+").to_numpy()

    def _is_internal(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(lambda x: x['sender_domain'] in x['recipient_domain'], axis=1).to_numpy()

    def _is_external(self, X: pd.DataFrame) -> np.ndarray:
        return self._n_recipient_domains(X) > 1 | ~self._is_internal(X)


class TopicFeatureExtraction(BaseFeatureExtraction):
    """Extract topic features from emails."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        super().__init__(config)
        try:
            from src.emailurgency.models.berttopic import BERTTopicModel

            self.model = BERTTopicModel()
        except Exception:
            self.model = None

    def extract(self, corpus: pd.Series, embeddings: np.ndarray) -> pd.DataFrame:
        """Fit topic model and return in-sample topic features."""
        if self.model is None:
            return self._empty_features(len(corpus))
        try:
            topics, probs = self.model.fit_predict(corpus.fillna('').astype(str), embeddings)
            return self._features(topics, probs)
        except Exception:
            return self._empty_features(len(corpus))

    def extract_new(self, corpus: pd.Series, embeddings: np.ndarray) -> pd.DataFrame:
        """Return topic features for new rows."""
        if self.model is None:
            return self._empty_features(len(corpus))
        try:
            topics, probs = self.model.predict(corpus.fillna('').astype(str), embeddings)
            return self._features(topics, probs)
        except Exception:
            return self._empty_features(len(corpus))

    def _features(self, topics: np.ndarray, probs: np.ndarray | None) -> pd.DataFrame:
        probs = np.asarray(probs) if probs is not None else np.zeros(len(topics))
        if probs.ndim > 1:
            probs = probs.max(axis=1)
        return pd.DataFrame({'topic': np.asarray(topics), 'topic_probability': probs})

    def _empty_features(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame({'topic': np.full(n_rows, -1), 'topic_probability': np.zeros(n_rows)})


class SimilarityFeatureExtraction(BaseFeatureExtraction):
    """Extract similarity features from embeddings using VectorDB."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        super().__init__(config)
        self.vectordb = VectorDB(self.config)
        self.y: np.ndarray | None = None

    def extract(self, embeddings: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Fit VectorDB and return in-sample nearest-neighbor features."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.y = np.asarray(y, dtype=int)
        self.vectordb.store(embeddings)

        k = min(self.config.k + 1, len(embeddings))
        similarities, neighbors = self.vectordb.search(embeddings, k=k)
        return self._features(similarities[:, 1:], neighbors[:, 1:])

    def extract_new(self, embeddings: np.ndarray) -> pd.DataFrame:
        """Return similarity features against the VectorDB fitted in extract."""
        if self.y is None:
            raise ValueError('Call extract before extract_new.')

        embeddings = np.asarray(embeddings, dtype=np.float32)
        k = min(self.config.k, len(self.y))
        similarities, neighbors = self.vectordb.search(embeddings, k=k)
        return self._features(similarities, neighbors)

    def _features(self, similarities: np.ndarray, neighbors: np.ndarray) -> pd.DataFrame:
        labels = self.y[neighbors]
        pos = np.where(labels == 1, similarities, np.nan)
        neg = np.where(labels == 0, similarities, np.nan)
        pos_mean = np.nan_to_num(np.nanmean(pos, axis=1), nan=0.0)
        neg_mean = np.nan_to_num(np.nanmean(neg, axis=1), nan=0.0)

        return pd.DataFrame({
            'max_similarity': similarities.max(axis=1),
            'mean_similarity': similarities.mean(axis=1),
            'std_similarity': similarities.std(axis=1),
            'top_k_mean_similarity': similarities.mean(axis=1),
            'top_k_std_similarity': similarities.std(axis=1),
            'top_k_pos_similarity': pos_mean,
            'top_k_neg_similarity': neg_mean,
            'gap_similarity': pos_mean - neg_mean,
        })
