from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import spacy

import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from configs.train_config import LexiconConfig, get_lexicon_config
from src.emailurgency.models.vectordb import VectorDB

class Lexicon:
    """Lexicon"""

    def __init__(self, config: LexiconConfig | None = None) -> None:
        """Initialize Lexicon"""
        self.config = config or get_lexicon_config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.nlp = spacy.load(self.config.spacy_model, disable=["ner", "tagger", "lemmatizer", "attribute_ruler", "textcat", "parser"])
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')

        self.embeddings_model = SentenceTransformer(self.config.embeddings_model, device=self.device)
        self.vectordb = VectorDB(self.config)
        self.keybert_model = KeyBERT(self.embeddings_model)

    def build_lexicon(self, embeddings: np.ndarray, corpus: pd.Series) -> None:
        """Build lexicon"""
        dict_seeds = self._initialize_seeds()
        email_embeddings = embeddings

        for file, seeds in dict_seeds.items():
            seed_embeddings = self.embeddings_model.encode(seeds['prototype'].tolist(), **self.config.embeddings_params)
            email_idx, _ = self._retrieve_neighbors(seed_embeddings, email_embeddings, self.config.max_k)
            
            chunks = self._chunking(corpus.iloc[email_idx])
            chunk_embeddings = self.embeddings_model.encode(chunks['sentences'].tolist(), **self.config.embeddings_params)

            idx, similarity = self._retrieve_neighbors(seed_embeddings, chunk_embeddings, self.config.max_k, self.config.threshold)
            neighbors = chunks.iloc[idx]['sentences'].to_numpy()
            self._add_prototypes(file, neighbors, similarity)

    def _initialize_seeds(self) -> dict[Path, pd.DataFrame]:
        """Initialize seed lexicons from CSV files with the model lexicon schema."""
        seeds: dict[Path, pd.DataFrame] = {}
        required_columns = {'prototype', 'source', 'similarity'}

        for path in self.config.lexicon_path.glob('*.csv'):
            if path.stat().st_size == 0:
                continue

            lexicon = pd.read_csv(path)
            if not required_columns.issubset(lexicon.columns):
                continue

            lexicon = lexicon.query('source == "Seed"')
            if not lexicon.empty:
                seeds[path] = lexicon

        return seeds

    def _chunking(self, corpus: pd.Series) -> pd.DataFrame:
        """Chunk emails into sentences"""
        chunks = []
        for row, doc in enumerate(self.nlp.pipe(corpus, **self.config.spacy_pipe_params)):
            for sent in doc.sents:
                if 3 <= len(sent.text.strip()) <= 200:
                    chunks.append({'row': row, 'sentences': sent.text.strip()})

        chunks = pd.DataFrame(chunks).groupby(['sentences'], as_index=False)['row'].max().sort_values('row').reset_index(drop=True)
        chunks['chunk_id'] = chunks.index
        return chunks

    def _retrieve_neighbors(self, seed_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve neighbors"""
        self.vectordb.store(corpus_embeddings)

        similarity, idx = self.vectordb.search(seed_embeddings, k=k)
        df_similarity = pd.DataFrame({'idx': idx.ravel(), 'similarity': similarity.ravel()})
        df_similarity = df_similarity.groupby(['idx'], as_index=False)['similarity'].max()
        if threshold is not None:
            df_similarity = df_similarity[df_similarity['similarity'] > threshold]

        self.vectordb.index.reset()

        return df_similarity['idx'].to_numpy(), df_similarity['similarity'].to_numpy()

    def _add_prototypes(self, path: Path, neighbors: np.ndarray, similarity: np.ndarray) -> None:
        """Add prototypes"""
        neighbors = neighbors.tolist()
        keywords = self.keybert_model.extract_keywords(neighbors, **self.config.keybert_params)
        if keywords and isinstance(keywords[0], tuple):
            keywords = [keywords]

        rows = []
        for sentence, sentence_keywords, sim in zip(neighbors, keywords, similarity):
            for prototype, keybert_score in sentence_keywords:
                rows.append({'prototype': prototype, 'source': 'Generated', 'similarity': np.round(sim, 2)})

        new = pd.DataFrame(rows)
        existing = pd.read_csv(path)
        output = pd.concat([existing, new], ignore_index=True).drop_duplicates('prototype', keep='first')
        output.to_csv(path, index=False)
