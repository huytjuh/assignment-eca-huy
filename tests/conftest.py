from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "fastapi" not in sys.modules:
    fastapi = ModuleType("fastapi")
    fastapi.__spec__ = importlib.machinery.ModuleSpec("fastapi", loader=None)

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi
if "pydantic_settings" not in sys.modules:
    pydantic_settings = ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = pydantic_settings

if "sentence_transformers" not in sys.modules:
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", loader=None)

    class SentenceTransformer:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, **params):
            return np.ones((len(texts), 2), dtype=float)

    sentence_transformers.SentenceTransformer = SentenceTransformer
    sentence_transformers.util = ModuleType("sentence_transformers.util")
    sys.modules["sentence_transformers"] = sentence_transformers

if "faiss" not in sys.modules:
    faiss = ModuleType("faiss")
    faiss.__spec__ = importlib.machinery.ModuleSpec("faiss", loader=None)
    faiss.normalize_L2 = lambda values: None

    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
            self.values = None

        def add(self, values):
            self.values = values

        def search(self, queries, k):
            return np.zeros((len(queries), k)), np.zeros((len(queries), k), dtype=int)

        def reset(self):
            self.values = None

    faiss.IndexFlatIP = IndexFlatIP
    sys.modules["faiss"] = faiss

if "keybert" not in sys.modules:
    keybert = ModuleType("keybert")
    keybert.__spec__ = importlib.machinery.ModuleSpec("keybert", loader=None)

    class KeyBERT:
        def __init__(self, model=None):
            self.model = model

        def extract_keywords(self, texts, **params):
            return [[("urgent", 0.9)] for _ in texts]

    keybert.KeyBERT = KeyBERT
    sys.modules["keybert"] = keybert

if "umap" not in sys.modules:
    umap = ModuleType("umap")

    class UMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    umap.UMAP = UMAP
    sys.modules["umap"] = umap

if "hdbscan" not in sys.modules:
    hdbscan = ModuleType("hdbscan")

    class HDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    hdbscan.HDBSCAN = HDBSCAN
    sys.modules["hdbscan"] = hdbscan

if "bertopic" not in sys.modules:
    bertopic = ModuleType("bertopic")

    class BERTopic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.saved_path = None

        def fit(self, docs, embeddings):
            self.docs = docs
            self.embeddings = embeddings
            return self

        def fit_transform(self, docs, embeddings):
            return np.arange(len(docs)), np.linspace(0.5, 0.9, len(docs))

        def transform(self, docs, embeddings):
            return np.arange(len(docs)), np.linspace(0.4, 0.8, len(docs))

        def save(self, path, *args, **kwargs):
            self.saved_path = path

        @classmethod
        def load(cls, path):
            model = cls()
            model.loaded_path = path
            return model

    bertopic.BERTopic = BERTopic
    sys.modules["bertopic"] = bertopic

if "datasets" not in sys.modules:
    datasets = ModuleType("datasets")

    class Dataset:
        @classmethod
        def from_dict(cls, data):
            return data

    datasets.Dataset = Dataset
    sys.modules["datasets"] = datasets

if "peft" not in sys.modules:
    peft = ModuleType("peft")

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    peft.LoraConfig = LoraConfig
    peft.get_peft_model = lambda model, config: model
    sys.modules["peft"] = peft

if "transformers" not in sys.modules:
    transformers = ModuleType("transformers")

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2] for _ in texts], "attention_mask": [[1, 1] for _ in texts]}

        def save_pretrained(self, path):
            return None

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    class DataCollatorWithPadding:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            return None

    class TrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForSequenceClassification = AutoModelForSequenceClassification
    transformers.DataCollatorWithPadding = DataCollatorWithPadding
    transformers.Trainer = Trainer
    transformers.TrainingArguments = TrainingArguments
    sys.modules["transformers"] = transformers