from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class IdentityPreProcess:
    def preprocess_subject(self, values):
        return values.fillna("")

    def preprocess_body(self, values):
        return values.fillna("")


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, **params):
        return np.arange(len(texts) * 2, dtype=float).reshape(len(texts), 2)


class FakeWeakLabeler:
    @classmethod
    def load(cls, config=None):
        return cls()

    def predict(self, df, embeddings):
        return np.array([0, 1][: len(df)])


class FakeMetaFeatureExtraction:
    def extract(self, df):
        return pd.DataFrame({"message_id": df["message_id"], "n_words": [3] * len(df)})


class FakeSimilarityFeatureExtraction:
    def extract(self, embeddings, labels):
        self.labels = labels
        return pd.DataFrame({"max_similarity": [0.5] * len(embeddings)})

    def extract_new(self, embeddings):
        return pd.DataFrame({"max_similarity": [0.5] * len(embeddings)})


class FakeTopicFeatureExtraction:
    load_existing = None

    def __init__(self, load_existing=False):
        self.load_existing = load_existing
        FakeTopicFeatureExtraction.load_existing = load_existing

    def extract_new(self, corpus, embeddings):
        return pd.DataFrame({"topic": [0] * len(corpus), "topic_probability": [0.8] * len(corpus)})


class FakeLoRA:
    def embeddings(self, texts):
        return np.ones((len(texts), 2))


class FakeLogisticClassifier:
    threshold = 0.6

    def __init__(self):
        self.config = SimpleNamespace(model_path=Path("fake.joblib"))

    @classmethod
    def load(cls, config=None):
        return cls()

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.3, 0.7]][: len(X)])



def test_berttopic_fit_predict_and_predict_use_synthetic_embeddings():
    from src.emailurgency.models.berttopic import BERTTopicModel

    config = SimpleNamespace(
        embeddings_model="fake-model",
        embeddings_params={},
        umap_params={},
        hbdscan_params={},
        tfidf_params={},
        model_path=Path("unused"),
    )
    model = BERTTopicModel(config=config)

    topics, probs = model.fit_predict(pd.Series(["one", None]), embeddings=np.ones((2, 2)))
    new_topics, new_probs = model.predict(pd.Series(["three"]), embeddings=np.ones((1, 2)))

    assert topics.tolist() == [0, 1]
    assert model.berttopic_model.saved_path == config.model_path
    assert probs.shape == (2,)
    assert new_topics.tolist() == [0]
    assert new_probs.tolist() == [0.4]


def test_berttopic_retries_with_relaxed_hdbscan_when_all_topics_are_noise(monkeypatch):
    from src.emailurgency.models.berttopic import BERTTopicModel

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **params):
            return np.ones((len(texts), 2), dtype=float)

    class FakeBERTopic:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, corpus, embeddings):
            type(self).calls += 1
            if type(self).calls == 1:
                return np.array([-1, -1]), np.ones(2)
            return np.array([0, 1]), np.ones(2)

        def transform(self, corpus, embeddings):
            return np.array([0]), np.ones(1)

        def save(self, path, *args, **kwargs):
            self.saved_path = path

    monkeypatch.setattr("src.emailurgency.models.berttopic.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("src.emailurgency.models.berttopic.BERTopic", FakeBERTopic)

    config = SimpleNamespace(
        embeddings_model="fake-model",
        embeddings_params={},
        umap_params={},
        hbdscan_params={},
        tfidf_params={},
        model_path=Path("unused"),
    )
    model = BERTTopicModel(config=config)

    topics, probs = model.fit_predict(pd.Series(["one", "two"]), embeddings=np.ones((2, 2)))

    assert topics.tolist() == [0, 1]
    assert probs.shape == (2,)


def test_evaluation_pipeline_computes_metrics_and_roc_curve():
    from src.emailurgency.pipelines.evaluation import Evaluation

    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.8, 0.9])
    y_pred = np.array([0, 0, 1, 1])

    result = Evaluation.compute(y_true, y_proba, y_pred)
    roc = Evaluation.roc_curve_df(y_true, y_proba)

    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.auc_roc == 1.0
    assert {"fpr", "tpr", "threshold"}.issubset(roc.columns)


def test_run_eval_writes_predictions_and_metrics_from_synthetic_pipeline(monkeypatch, tmp_path):
    import scripts.run_eval as run_eval

    test_df = pd.DataFrame(
        {
            "message_id": ["m1", "m2"],
            "subject": ["hello", "urgent"],
            "body": ["body one", "body two"],
            "gold_labels": [np.nan, 1],
        }
    )
    written = {}
    eval_path = tmp_path / "eval.parquet"

    monkeypatch.setattr(run_eval, "parse_args", lambda: SimpleNamespace(sample=None, use_lora=False))
    monkeypatch.setattr(run_eval, "get_train_config", lambda: SimpleNamespace(test_path="test.parquet", eval_output_path=eval_path, embeddings_model="fake", embeddings_params={}, spacy_max_length=100))
    monkeypatch.setattr(run_eval, "get_label_config", lambda: SimpleNamespace())
    monkeypatch.setattr(run_eval.pd, "read_parquet", lambda path, engine=None: test_df.copy())
    monkeypatch.setattr(run_eval, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(run_eval, "WeakLabeler", FakeWeakLabeler)
    monkeypatch.setattr(run_eval, "MetaFeatureExtraction", FakeMetaFeatureExtraction)
    monkeypatch.setattr(run_eval, "SimilarityFeatureExtraction", FakeSimilarityFeatureExtraction)
    monkeypatch.setattr(run_eval, "TopicFeatureExtraction", FakeTopicFeatureExtraction)
    monkeypatch.setattr(run_eval, "PreProcess", IdentityPreProcess)
    monkeypatch.setattr(run_eval, "LoRA", FakeLoRA)
    monkeypatch.setattr(run_eval, "LogisticClassifier", FakeLogisticClassifier)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kwargs: written.setdefault(str(path), self.copy()))

    run_eval.main()

    output = written[str(eval_path)]
    assert output["proba"].tolist() == [0.2, 0.7]
    assert output["pred"].tolist() == [0, 1]
    assert output["threshold"].tolist() == [0.6, 0.6]
    assert FakeTopicFeatureExtraction.load_existing is True
