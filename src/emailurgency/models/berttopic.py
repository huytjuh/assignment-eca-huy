from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.train_config import BERTTopicConfig, get_berttopic_config

class BERTTopicModel:
    """BERTTopicModel"""

    def __init__(self, config: BERTTopicConfig | None = None) -> None:
        """Initialize BERTTopicModel"""
        self.config = config or get_berttopic_config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.embeddings_model = SentenceTransformer(self.config.embeddings_model, device=self.device)
        self.umap_model = UMAP(**self.config.umap_params)
        self.hbdscan_model = HDBSCAN(**self.config.hbdscan_params)
        self.tfidf_model = TfidfVectorizer(**self.config.tfidf_params)

        self.berttopic_model = BERTopic(
            embedding_model=self.config.embeddings_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hbdscan_model,
            vectorizer_model=self.tfidf_model
        )

    def fit(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> BERTTopicModel:
        """Fit BERTTopicModel."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        self.berttopic_model.fit(corpus.tolist(), embeddings)
        self.save(self.config.model_path)
        return self

    def fit_predict(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Fit BERTTopicModel and return topic assignments."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        topics, probs = self.berttopic_model.fit_transform(corpus.tolist(), embeddings)
        return topics, probs
    
    def predict(self, corpus: pd.Series, embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict BERTTopicModel."""
        corpus = corpus.fillna('').astype(str)
        embeddings = embeddings if embeddings is not None else self.embeddings_model.encode(corpus.tolist(), **self.config.embeddings_params)
        topics, probs = self.berttopic_model.transform(corpus.tolist(), embeddings)
        return topics, probs
    
    def save(self, path: Path) -> None:
        """Save BERTTopicModel"""
        self.berttopic_model.save(path)
        return self

    @classmethod
    def load(cls, path: Path, config: BERTTopicConfig | None = None) -> BERTTopicModel:
        """Load BERTTopicModel"""
        instance = cls(config or get_berttopic_config())
        instance.bertopic_model = BERTopic.load(path)
        return instance
    
    