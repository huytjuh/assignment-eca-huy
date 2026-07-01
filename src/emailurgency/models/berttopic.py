from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from configs.train_config import BERTTopicConfig, get_berttopic_config


class BERTopicFitError(RuntimeError):
    """Raised when BERTopic cannot produce any non-noise clusters."""


class BERTTopicModel:
    """BERTTopicModel"""

    def __init__(self, config: BERTTopicConfig | None = None) -> None:
        """Initialize BERTTopicModel"""
        self.config = config or get_berttopic_config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.embeddings_model = SentenceTransformer(self.config.embeddings_model, device=self.device)
        self.umap_model = UMAP(**self.config.umap_params)
        self.hbdscan_model = HDBSCAN(**self.config.hbdscan_params)
        vectorizer_params = getattr(self.config, 'vectorizer_params', getattr(self.config, 'vectorizer_params', {}))
        self.vectorizer_model = CountVectorizer(**vectorizer_params)

        self.berttopic_model = self._build_model()

    def _build_model(self, hdbscan_params: dict | None = None) -> BERTopic:
        """Build a BERTopic instance with optional HDBSCAN overrides."""
        model = BERTopic(
            embedding_model=self.embeddings_model,
            umap_model=self.umap_model,
            hdbscan_model=HDBSCAN(**(hdbscan_params or self.config.hbdscan_params)),
            vectorizer_model=self.vectorizer_model,
        )
        return model

    def _fit_with_retries(self, corpus: list[str], embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit BERTopic and retry with relaxed HDBSCAN settings if everything becomes noise."""
        topics, probs = self.berttopic_model.fit_transform(corpus, embeddings)
        if np.asarray(topics).ndim == 0:
            raise BERTopicFitError("BERTopic returned an invalid topic assignment")

        if np.unique(topics).size > 1 and np.any(np.asarray(topics) != -1):
            return topics, probs

        fallback_params = dict(self.config.hbdscan_params)
        fallback_params['min_cluster_size'] = max(2, int(fallback_params.get('min_cluster_size', 5) // 2))
        fallback_params['min_samples'] = max(1, int(fallback_params.get('min_cluster_size', 5) // 2))
        fallback_params['cluster_selection_epsilon'] = 0.0

        self.berttopic_model = self._build_model(hdbscan_params=fallback_params)
        topics, probs = self.berttopic_model.fit_transform(corpus, embeddings)
        if np.unique(topics).size <= 1 and np.all(np.asarray(topics) == -1):
            raise BERTopicFitError("BERTopic produced only noise labels after retry")
        return topics, probs

    def fit(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> BERTTopicModel:
        """Fit BERTTopicModel."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        self._fit_with_retries(corpus.tolist(), embeddings)
        self.save(self.config.model_path)
        return self

    def fit_predict(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Fit BERTTopicModel and return topic assignments."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        topics, probs = self._fit_with_retries(corpus.tolist(), embeddings)
        self.save(self.config.model_path)
        return topics, probs
    
    def predict(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict BERTTopicModel."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        topics, probs = self.berttopic_model.transform(corpus.tolist(), embeddings)
        return topics, probs

    def save(self, path: Path) -> None:
        """Save BERTTopicModel"""
        self.berttopic_model.save(path, serialization="pickle")
        return self
    
    @classmethod
    def load(cls, path: Path, config: BERTTopicConfig | None = None) -> BERTTopicModel:
        """Load BERTTopicModel"""
        instance = cls(config or get_berttopic_config())
        instance.berttopic_model = BERTopic.load(path)
        return instance
    
    


