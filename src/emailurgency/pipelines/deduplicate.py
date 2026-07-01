from __future__ import annotations

import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

from src.emailurgency.models.vectordb import VectorDB
from configs.global_config import DataConfig, get_data_config

import logging
logger = logging.getLogger(__name__)

class Deduplicate:
    """Deduplicate emails using deduplication API"""

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize Deduplicate"""
        self.config = config or get_data_config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.embeddings_model = SentenceTransformer(self.config.embeddings_model, device=self.device)
        self.vectordb = VectorDB(self.config)

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates"""
        data = data.copy()

        data['to_key'] = data['to'].apply(
            lambda x: tuple(sorted(x)) if isinstance(x, (list, tuple))
            else tuple(sorted(x.tolist())) if hasattr(x, 'tolist')
            else x
        )

        unique_keys = ['from', 'to_key', 'subject', 'date']

        mask_folder = data.groupby(unique_keys)['x_folder'].transform('nunique') > 1

        mask_similarity = pd.Series(False, index=data.index)
        mask_similarity.loc[mask_folder] = (
            self._retrieve_max_similarity(data.loc[mask_folder]) > self.config.similarity_threshold
        )

        mask = mask_folder & mask_similarity

        duplicated = data.loc[mask].drop_duplicates(subset=unique_keys, keep='first')
        output = pd.concat([data.loc[~mask], duplicated], ignore_index=True)

        output = output.drop(columns=['to_key'])

        print(f'Deduplicated {len(data) - len(output)} emails')
        return output

    def _retrieve_max_similarity(self, data: pd.DataFrame) -> pd.Series:
        """Retrieve max similarity per row"""
        if len(data) < 2:
            return pd.Series(False, index=data.index)

        bodies = data['body'].fillna('').astype(str)

        embeddings = self.embeddings_model.encode(bodies.tolist(), **self.config.embeddings_params)

        self.vectordb.store(embeddings)
        similarities, _ = self.vectordb.search(embeddings, k=2)

        return pd.Series(similarities[:, 1], index=data.index)