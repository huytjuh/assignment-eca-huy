from __future__ import annotations

from pathlib import Path
from typing import Any

from functools import lru_cache
from pydantic_settings import BaseSettings

class TrainConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    train_path: Path = Path('data/train.parquet')
    test_path: Path = Path('data/test.parquet')
    eval_output_path: Path = Path('data/eval.parquet')
    label_path: Path = Path('data/gold_labels.parquet')

    val_size: float = 0.2
    n_splits: int = 5
    random_seed: int = 42
    sample_size: int | None = 10000
    output_path: Path = Path('data/dataset.parquet')

    spacy_model: str = 'artifacts/spacy/en_core_web_sm-3.8.0'
    spacy_max_length: int = 2000
    spacy_batch_size: int = 512
    spacy_n_processes: int = 4

    embeddings_model: str = 'artifacts/transformers/all-MiniLM-L6-v2'
    embeddings_batch_size: int = 256

    n_splits: int = 4
    shuffle: bool = True

    @property
    def spacy_pipe_params(self) -> dict[str, Any]:
        return {'batch_size': self.spacy_batch_size, 'n_process': self.spacy_n_processes}

    @property
    def embeddings_params(self) -> dict[str, Any]:
        return {'batch_size': self.embeddings_batch_size, 'normalize_embeddings': True}
    
    @property
    def kfold_params(self) -> dict[str, Any]:
        return {'n_splits': self.n_splits, 'random_state': self.random_seed, 'shuffle': self.shuffle}


class LexiconConfig(TrainConfig):
    model_config = {'env_file': '.env'}

    lexicon_path: Path = Path('lexicon')
    threshold: float = 0.7
    max_k: int = 100

    keyphrase_ngram_range: tuple[int, int] = (1, 3)
    use_mmr: bool = True
    diversity: float = 0.3
    top_n: int = 5

    @property
    def keybert_params(self) -> dict[str, Any]:
        return {'keyphrase_ngram_range': self.keyphrase_ngram_range, 'use_mmr': self.use_mmr, 'diversity': self.diversity, 'top_n': self.top_n}

class BERTTopicConfig(TrainConfig):
    model_config = {'env_file': '.env'}

    model_path: Path = Path('artifacts/bertopic')
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 15
    hdbscan_prediction_data: bool = True
    tfidf_max_features: int = 5000
    tfidf_stop_words: str = 'english'

    @property
    def umap_params(self) -> dict[str, Any]:
        return {'n_neighbors': self.umap_n_neighbors, 'n_components': self.umap_n_components, 'min_dist': self.umap_min_dist, 'metric': 'cosine'}

    @property
    def hbdscan_params(self) -> dict[str, Any]:
        return {'min_cluster_size': self.hdbscan_min_cluster_size, 'prediction_data': self.hdbscan_prediction_data}

    @property
    def tfidf_params(self) -> dict[str, Any]:
        return {'max_features': self.tfidf_max_features, 'stop_words': self.tfidf_stop_words}

class LabelConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    lexicon_path: Path = Path('lexicon/urgency.csv')
    lexicon_dir: Path = Path('lexicon')
    model_path: Path = Path('artifacts/labeler/snorkel.pkl')
    semantic_similarity_threshold: float = 0.5
    semantic_text_max_length: int = 2000
    embeddings_model: str = 'artifacts/transformers/all-MiniLM-L6-v2'
    embeddings_batch_size: int = 256
    cardinality: int = 2
    prob_threshold: float = 0.5

    n_trials: int = 10

    n_epochs: int = 500
    lr: float = 1e-3
    seed: int = 42
    log_freq: int = 100

    @property
    def fit_params(self) -> dict[str, Any]:
        return {'n_epochs': self.n_epochs, 'lr': self.lr, 'seed': self.seed, 'log_freq': self.log_freq}

    @property
    def embeddings_params(self) -> dict[str, Any]:
        return {'batch_size': self.embeddings_batch_size, 'normalize_embeddings': True}

class LoraConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    base_model: str = 'artifacts/transformers/distilbert-base-uncased'
    model_path: Path = Path('artifacts/lora')
    lora_path: Path = Path('artifacts/lora/adapter')
    merged_path: Path = Path('artifacts/lora/merged')

    num_labels: int = 2
    max_length: int = 128
    r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    bias: str = 'none'
    task_type: str = 'SEQ_CLS'
    target_modules: tuple[str, ...] = ('q_lin', 'v_lin')

    output_dir: Path = Path('artifacts/lora/checkpoints')
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: float = 2.0
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_strategy: str = 'no'
    eval_strategy: str = 'epoch'

    @property
    def peft_params(self) -> dict[str, Any]:
        return {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'bias': self.bias,
            'task_type': self.task_type,
            'target_modules': list(self.target_modules),
        }

    @property
    def training_args(self) -> dict[str, Any]:
        return {
            'output_dir': str(self.output_dir),
            'learning_rate': self.learning_rate,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'per_device_eval_batch_size': self.per_device_eval_batch_size,
            'num_train_epochs': self.num_train_epochs,
            'weight_decay': self.weight_decay,
            'logging_steps': self.logging_steps,
            'save_strategy': self.save_strategy,
            'eval_strategy': self.eval_strategy,
        }


class LogisticRegressionConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    model_path: Path = Path('artifacts/classifier/logistic.joblib')

    penalty: str = 'l2'
    C: float = 1.0
    solver: str = 'lbfgs'
    max_iter: int = 1000
    tol: float = 1e-4
    random_state: int = 42
    class_weight: str | dict[int, float] | None = 'balanced'

    @property
    def model_params(self) -> dict[str, Any]:
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
        }

class VectorDBConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    max_k: int = 10
    similarity_threshold: float = 0.5


@lru_cache
def get_train_config() -> TrainConfig:
    return TrainConfig()

@lru_cache
def get_lexicon_config() -> LexiconConfig:
    return LexiconConfig()

@lru_cache
def get_label_config() -> LabelConfig:
    return LabelConfig()

@lru_cache
def get_berttopic_config() -> BERTTopicConfig:
    return BERTTopicConfig()

@lru_cache
def get_lora_config() -> LoraConfig:
    return LoraConfig()

@lru_cache
def get_logistic_regression_config() -> LogisticRegressionConfig:
    return LogisticRegressionConfig()
