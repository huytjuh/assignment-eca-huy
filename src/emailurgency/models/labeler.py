from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

from configs.train_config import LabelConfig, get_label_config


URGENT, NOT_URGENT, ABSTAIN = 1, 0, -1


class WeakLabeler:
    """Weak urgency labeler backed by lexicon and semantic signals."""

    LEXICON_LABELS = {
        'automated': NOT_URGENT,
        'bulk': NOT_URGENT,
        'notification': NOT_URGENT,
        'financial': URGENT,
        'legal': URGENT,
        'request': URGENT,
        'urgency': URGENT,
    }

    def __init__(self, config: LabelConfig | None = None) -> None:
        """Initialize WeakLabeler."""
        self.config = config or get_label_config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.lexicons = self._load_lexicons(self.config.lexicon_dir)
        self.lexicon_patterns = self._compile_lexicon_patterns(self.lexicons)

        self.embeddings_model = SentenceTransformer(self.config.embeddings_model, device=self.device)
        self.semantic_embeddings = self._embed_lexicons(self.lexicons)

        self.lf_thresholds = {'semantic': self.config.semantic_similarity_threshold}
        self.prob_threshold = self.config.prob_threshold

        self.lfs = self._snorkel_lfs()
        self.applier = PandasLFApplier(self.lfs)
        self.label_model = LabelModel(cardinality=self.config.cardinality, verbose=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None, embeddings: np.ndarray | None = None) -> WeakLabeler:
        """Fit Snorkel label model and tune threshold when gold labels exist."""
        X = self._add_semantic_features(X, embeddings)
        if y is None:
            self._fit_label_model(X)
            return self

        mask = y.isin([0, 1]).to_numpy()
        X_train = X.iloc[~mask] if (~mask).any() else X
        X_val = X.iloc[mask]
        y_val = y.iloc[mask]
        val_embeddings = embeddings[mask] if embeddings is not None and mask.any() else None

        def objective(trial: optuna.Trial) -> float:
            self.lf_thresholds['semantic'] = trial.suggest_float('semantic_t', 0.30, 0.90)
            prob_threshold = trial.suggest_float('prob_threshold', 0.50, 0.95)
            self._refresh_lfs()

            self._fit_label_model(X_train)
            if X_val.empty:
                return 0.0

            probs = self.predict_proba(X_val, val_embeddings)
            pred = (probs >= prob_threshold).astype(int)
            return f1_score(y_val, pred, zero_division=0)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)

        self.prob_threshold = study.best_params['prob_threshold']
        self.lf_thresholds['semantic'] = study.best_params['semantic_t']
        self._refresh_lfs()
        self._fit_label_model(X)
        self.save()
        return self

    def save(self, path: Path | None = None) -> None:
        """Save fitted label model and tuned thresholds."""
        path = path or self.config.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            pickle.dump(
                {
                    'label_model': self.label_model,
                    'lf_thresholds': self.lf_thresholds,
                    'prob_threshold': self.prob_threshold,
                },
                file,
            )

    @classmethod
    def load(cls, path: Path | None = None, config: LabelConfig | None = None) -> WeakLabeler:
        """Load a fitted labeler."""
        labeler = cls(config)
        path = path or labeler.config.model_path
        with path.open('rb') as file:
            state = pickle.load(file)

        labeler.label_model = state['label_model']
        labeler.lf_thresholds = state['lf_thresholds']
        labeler.prob_threshold = state['prob_threshold']
        labeler._refresh_lfs()
        return labeler

    @classmethod
    def load_and_label(
        cls,
        X: pd.DataFrame,
        embeddings: np.ndarray | None = None,
        path: Path | None = None,
        config: LabelConfig | None = None,
    ) -> np.ndarray:
        """Load a fitted labeler and return silver labels."""
        return cls.load(path, config).predict(X, embeddings)

    def predict_proba(self, X: pd.DataFrame, embeddings: np.ndarray | None = None) -> np.ndarray:
        """Predict label probabilities."""
        X = self._add_semantic_features(X, embeddings)
        LF = self.applier.apply(X, progress_bar=False)
        return self.label_model.predict_proba(LF)[:, 1]

    def predict(self, X: pd.DataFrame, embeddings: np.ndarray | None = None) -> np.ndarray:
        """Predict labels."""
        return (self.predict_proba(X, embeddings) >= self.prob_threshold).astype(int)

    def _fit_label_model(self, X: pd.DataFrame) -> None:
        """Fit label model on current labeling functions."""
        LF = self.applier.apply(X, progress_bar=False)
        self.label_model.fit(LF, **self.config.fit_params)

    def _refresh_lfs(self) -> None:
        """Refresh Snorkel applier after LF threshold updates."""
        self.lfs = self._snorkel_lfs()
        self.applier = PandasLFApplier(self.lfs)

    def _snorkel_lfs(self) -> list[labeling_function]:
        """Create seed, generated, and semantic LF per lexicon category."""
        lfs = []
        for name, patterns in self.lexicon_patterns.items():
            label = self.LEXICON_LABELS.get(name, URGENT)
            lfs.append(self._pattern_lf(name, 'seed', label, patterns.get('seed')))
            lfs.append(self._pattern_lf(name, 'generated', label, patterns.get('generated')))
            lfs.append(self._semantic_lf(name, label))
        return lfs

    def _pattern_lf(self, name: str, source: str, label: int, pattern: re.Pattern[str] | None) -> labeling_function:
        """Create regex LF for one lexicon source."""
        @labeling_function(name=f'lf_{name}_{source}')
        def lf(row: pd.Series) -> int:
            text = f"{row.get('subject', '')} {row.get('body', '')}"
            return label if pattern is not None and pattern.search(text) else ABSTAIN

        return lf

    def _semantic_lf(self, name: str, label: int) -> labeling_function:
        """Create semantic similarity LF for one lexicon."""
        @labeling_function(name=f'lf_{name}_semantic')
        def lf(row: pd.Series) -> int:
            score = float(row.get(f'{name}_semantic_similarity', 0.0) or 0.0)
            return label if score >= self.lf_thresholds['semantic'] else ABSTAIN

        return lf

    def _add_semantic_features(self, X: pd.DataFrame, embeddings: np.ndarray | None = None) -> pd.DataFrame:
        """Add max semantic similarity for each loaded lexicon."""
        X = X.copy()
        for name in self.lexicons:
            X[f'{name}_semantic_similarity'] = 0.0

        if not self.semantic_embeddings or X.empty:
            return X

        if embeddings is None:
            text = X.get('subject', '').fillna('').astype(str) + ' ' + X.get('body', '').fillna('').astype(str)
            text = text.str.slice(0, self.config.semantic_text_max_length)
            embeddings = self.embeddings_model.encode(text.tolist(), **self.config.embeddings_params)

        text_embeddings = np.asarray(embeddings, dtype=np.float32)
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(X):
            raise ValueError('embeddings must have shape (len(X), embedding_dim).')

        for name, lexicon_embeddings in self.semantic_embeddings.items():
            if lexicon_embeddings.ndim != 2 or lexicon_embeddings.shape[0] == 0:
                continue
            if lexicon_embeddings.shape[1] != text_embeddings.shape[1]:
                continue
            X[f'{name}_semantic_similarity'] = (text_embeddings @ lexicon_embeddings.T).max(axis=1)
        return X

    def _load_lexicons(self, path: Path) -> dict[str, pd.DataFrame]:
        """Load top-level lexicon CSV files."""
        files = [path] if path.is_file() else sorted(path.glob('*.csv'))
        return {file.stem: self._load_lexicon(file) for file in files if file.stat().st_size > 0}

    def _load_lexicon(self, path: Path) -> pd.DataFrame:
        """Load one lexicon."""
        lexicon = pd.read_csv(path)
        missing = {'prototype', 'source', 'similarity'}.difference(lexicon.columns)
        if missing:
            raise ValueError(f'Lexicon {path} is missing required columns: {sorted(missing)}')

        lexicon = lexicon.copy()
        lexicon['prototype'] = lexicon['prototype'].astype(str).str.lower().str.strip()
        lexicon['source'] = lexicon['source'].astype(str).str.lower().str.strip()
        lexicon['similarity'] = pd.to_numeric(lexicon['similarity'], errors='coerce').fillna(0.0)
        return lexicon[lexicon['prototype'].ne('')].drop_duplicates('prototype', keep='first')

    def _compile_lexicon_patterns(self, lexicons: dict[str, pd.DataFrame]) -> dict[str, dict[str, re.Pattern[str] | None]]:
        """Compile seed and generated regexes for each lexicon."""
        return {
            name: {
                source: self._compile_pattern(lexicon.loc[lexicon['source'] == source, 'prototype'].to_list())
                for source in ('seed', 'generated')
            }
            for name, lexicon in lexicons.items()
        }

    def _embed_lexicons(self, lexicons: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        """Embed lexicon prototypes for semantic similarity LFs."""
        embeddings = {}
        for name, lexicon in lexicons.items():
            terms = lexicon['prototype'].dropna().astype(str).str.strip()
            terms = terms[terms.ne('')].to_list()
            if not terms:
                continue

            vectors = np.asarray(self.embeddings_model.encode(terms, **self.config.embeddings_params), dtype=np.float32)
            if vectors.ndim == 2 and vectors.shape[0] > 0:
                embeddings[name] = vectors
        return embeddings

    @staticmethod
    def _compile_pattern(terms: list[str]) -> re.Pattern[str] | None:
        """Compile terms into a word-boundary regex."""
        terms = [re.escape(term.strip()) for term in terms if term.strip()]
        return re.compile(r'\b(?:%s)\b' % '|'.join(terms), re.I) if terms else None
