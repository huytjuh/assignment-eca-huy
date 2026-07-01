from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any

import faiss

from src.emailurgency.schemas.features import SimilarityFeatures

class VectorDB:
    """Vector Database using FAISS library"""

    def __init__(self, config: Any) -> None:
        """Initialize VectorDB"""
        self.config = config

    def store(self, vectors: np.ndarray) -> None:
        """Store vectors in the vector database"""
        vectors = np.asarray(vectors, dtype=np.float32)

        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors in the vector database"""
        query = np.asarray(query, dtype=np.float32)
        k = k or self.config.k + 1
        
        faiss.normalize_L2(query)
        return self.index.search(query, k)
    
    def reset(self) -> None:
        """Reset the vector database"""
        self.index.reset()
    
    def max_similarity(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the index and similarity of the most similar neighbor."""
        self.store(vectors)

        similarities, idx = self.search(vectors, k=2)
        return idx[:, 1:], similarities[:, 1]
    
    def retrieve_similarity_features(self, vectors: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
        """Retrieve similarity features from the vector database."""
        if y is not None:
            self.store(vectors)
            
        similarity, idx = self.search(vectors)

        similarity_features = SimilarityFeatures(
            max_similarity=self.max_similarity(vectors),
            mean_similarity=self._mean_similarity(vectors),
            std_similarity=self._std_similarity(vectors),
            top_k_mean_similarity=self._top_k_mean_similarity(vectors, self.config.k),
            top_k_std_similarity=self._top_k_std_similarity(vectors, self.config.k),
            top_k_pos_similarity=self._top_k_pos_similarity(vectors, idx, y, self.config.k),
            top_k_neg_similarity=self._top_k_neg_similarity(vectors, idx, y, self.config.k),
            gap_similarity=self._gap_similarity(vectors, idx, y, self.config.k),
        )

        validated = [SimilarityFeatures.model_validate(x) for x in similarity_features.to_dict('records')] 
        return pd.DataFrame([row.model_dump() for row in validated])
    
    def _mean_similarity(self, similarities: np.ndarray, k: int | None = None) -> np.ndarray:
        """Mean similarity"""
        if k is not None:
            return np.mean(similarities[:, 1:k+1], axis=1)
        return np.mean(similarities[:, 1:], axis=1)
    
    def _std_similarity(self, similarities: np.ndarray, k: int | None = None) -> np.ndarray:
        """Standard deviation of similarity"""
        if k is not None:
            return np.std(similarities[:, 1:k+1], axis=1)
        return np.std(similarities[:, 1:], axis=1)
    
    def _pos_similarity(self, similarities: np.ndarray, neigbors: np.ndarray, y: np.ndarray, k: int | None = None) -> np.ndarray:
        """Positive similarity"""
        sims = similarities[:, 1:k+1] if k is not None else similarities[:, 1:]
        labels = y[neigbors[:, 1:k+1]] if k is not None else y[neigbors[:, 1:]]

        pos_sims = np.where(labels == 1, sims, 0)
        out = np.nanmean(pos_sims, axis=1)
        return np.nan_to_num(out, nan=0.0)
    
    def _neg_similarity(self, similarities: np.ndarray, neigbors: np.ndarray, y: np.ndarray, k: int | None = None) -> np.ndarray:
        """Negative similarity"""
        sims = similarities[:, 1:k+1] if k is not None else similarities[:, 1:]
        labels = y[neigbors[:, 1:k+1]] if k is not None else y[neigbors[:, 1:]]

        neg_sims = np.where(labels == 0, sims, 0)
        out = np.nanmean(neg_sims, axis=1)
        return np.nan_to_num(out, nan=0.0)
    
    def _gap_similarity(self, similarities: np.ndarray, neighbors: np.ndarray, y: np.ndarray, k: int | None = None) -> np.ndarray:
        """Gap similarity"""
        return self._pos_similarity(similarities, neighbors, y, k) - self._neg_similarity(similarities, neighbors, y, k)