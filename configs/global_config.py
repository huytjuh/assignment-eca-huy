from __future__ import annotations

from typing import Any
from pathlib import Path

from functools import lru_cache
from pydantic_settings import BaseSettings

class DataConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    use_spacy: bool = False
    test_size: float = 0.2
    random_seed: int = 42
    gold_label_path: str = 'data/gold_labels.parquet'

    k: int = 20
    similarity_threshold: float = 0.95
    embeddings_model: str = 'artifacts/transformers/all-MiniLM-L6-v2'
    embeddings_batch_size: int = 256
    
    @property
    def embeddings_params(self) -> dict[str, Any]:
        return {'batch_size': self.embeddings_batch_size, 'normalize_embeddings': True}

class PreProcessConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    spacy_model: str = 'artifacts/spacy/en_core_web_sm-3.8.0'

    boilerplate_file: Path = Path(r'lexicon\preprocess\boilerplate.csv')
    signature_file: Path = Path(r'lexicon\preprocess\signature.csv')

class FeatureConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    timezone: str = 'UTC'
    after_hours: tuple[int, int] = (8, 17)

    k: int = 10
    embeddings_model: str = 'artifacts/transformers/all-MiniLM-L6-v2'
    embeddings_batch_size: int = 256

    @property
    def embeddings_params(self) -> dict[str, Any]:
        return {'batch_size': self.embeddings_batch_size, 'normalize_embeddings': True}


@lru_cache
def get_data_config() -> DataConfig:
    return DataConfig()

@lru_cache
def get_preprocess_config() -> PreProcessConfig:
    return PreProcessConfig()

@lru_cache
def get_feature_config() -> FeatureConfig:
    return FeatureConfig()