from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_email_data_parses_dates_and_recipient_strings():
    from src.emailurgency.schemas.data import EmailData

    email = EmailData.model_validate(
        {
            "message_id": "m1",
            "date": "Mon, 1 Jan 2001 09:30:00 -0500",
            "from": "Sender <sender@example.com>",
            "to": "A <a@example.com>; b@example.com; A <a@example.com>",
        }
    )

    assert email.date.tzinfo is not None
    assert email.date.hour == 14
    assert email.to == ["a@example.com", "b@example.com"]


def test_preprocess_body_removes_html_entities_and_contact_noise(tmp_path):
    from src.emailurgency.pipelines.preprocess import PreProcess

    config = SimpleNamespace(
        boilerplate_file=tmp_path / "boilerplate.csv",
        signature_file=tmp_path / "signature.csv",
    )
    config.boilerplate_file.write_text("term\nnevermatch\n", encoding="utf-8")
    config.signature_file.write_text("term\nnevermatch\n", encoding="utf-8")
    cleaner = PreProcess(config=config, use_spacy=False)

    text = pd.Series(["<p>Hello!!!</p> mail me at x@example.com or https://x.test call 123-456-7890"])

    assert cleaner.preprocess_body(text).iloc[0] == "hello! mail me at or call"


def test_similarity_feature_helper_uses_saved_training_embeddings(monkeypatch, tmp_path):
    import api.app as app_module

    artifact_path = tmp_path / "train_embeddings.npz"
    np.savez_compressed(artifact_path, embeddings=np.ones((2, 2), dtype=np.float32), labels=np.array([0, 1]))
    calls = {}

    class FakeSimilarityFeatureExtraction:
        def extract(self, embeddings, labels):
            calls["fit"] = (embeddings.copy(), labels.copy())
            return pd.DataFrame({"max_similarity": [0.2] * len(embeddings)})

        def extract_new(self, embeddings):
            calls["new"] = embeddings.copy()
            return pd.DataFrame({"max_similarity": [0.4] * len(embeddings)})

    monkeypatch.setattr(app_module, "_similarity_feature_extractor", None)
    monkeypatch.setattr(app_module, "get_train_config", lambda: SimpleNamespace(embeddings_artifact_path=artifact_path))
    monkeypatch.setattr(app_module, "SimilarityFeatureExtraction", FakeSimilarityFeatureExtraction)

    result = app_module._get_similarity_features(np.ones((1, 2), dtype=np.float32))

    assert result["max_similarity"].tolist() == [0.4]
    assert calls["fit"][1].tolist() == [0, 1]
    assert calls["new"].shape == (1, 2)


def test_run_data_builds_train_and_test_splits(monkeypatch):
    import scripts.run_data as run_data

    raw = pd.DataFrame({"raw": range(8)})
    built = pd.DataFrame(
        {
            "message_id": [f"m{i}" for i in range(8)],
            "thread_id": [f"t{i}" for i in range(8)],
        }
    )
    gold = pd.DataFrame({"message_id": ["m0", "m1"], "gold_labels": [0, 1]})
    written = {}

    class DataLoader:
        def run(self):
            return raw

    class Deduplicate:
        def remove_duplicates(self, data):
            return data

    class DatasetBuilder:
        def transform(self, data):
            return built

    monkeypatch.setattr(run_data, "get_data_config", lambda: SimpleNamespace(gold_label_path="gold.parquet", test_size=0.5, random_seed=42))
    monkeypatch.setattr(run_data, "DataLoader", DataLoader)
    monkeypatch.setattr(run_data, "Deduplicate", Deduplicate)
    monkeypatch.setattr(run_data, "DatasetBuilder", DatasetBuilder)
    monkeypatch.setattr(run_data.pd, "read_parquet", lambda path: gold)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kwargs: written.setdefault(str(path), self.copy()))

    run_data.main()

    assert set(written) == {"data/train.parquet", "data/test.parquet"}
    assert len(written["data/train.parquet"]) + len(written["data/test.parquet"]) == len(built)
    assert "gold_labels" in written["data/train.parquet"].columns

def test_build_feature_frame_uses_recipient_domains(monkeypatch):
    import api.app as app_module
    from src.emailurgency.schemas.data import EmailData

    monkeypatch.setattr(app_module, "get_train_config", lambda: SimpleNamespace(spacy_max_length=200))

    frame = app_module._build_feature_frame(
        EmailData.model_validate(
            {
                "from": "sender@internal.test",
                "to": ["a@example.com", "b@internal.test"],
                "cc": ["c@vendor.test"],
                "subject": "Hello",
                "body": "Body",
            }
        )
    )

    assert frame.loc[0, "sender_domain"] == "internal.test"
    assert frame.loc[0, "recipient_domain"] == ["example.com", "internal.test", "vendor.test"]


def test_predict_checks_classifier_artifact_not_dataset(monkeypatch, tmp_path):
    import api.app as app_module
    from src.emailurgency.schemas.data import EmailData

    dataset_path = tmp_path / "dataset.parquet"
    classifier_path = tmp_path / "missing.joblib"
    dataset_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(app_module, "get_train_config", lambda: SimpleNamespace(output_path=dataset_path))
    monkeypatch.setattr(app_module, "get_logistic_regression_config", lambda: SimpleNamespace(model_path=classifier_path))

    try:
        app_module.predict(EmailData(subject="x", body="y"))
    except app_module.HTTPException as exc:
        assert exc.status_code == 503
        assert "Classifier artifact" in exc.detail
    else:
        raise AssertionError("expected HTTPException")


def test_predict_falls_back_when_topic_model_unavailable(monkeypatch, tmp_path):
    import api.app as app_module
    from src.emailurgency.schemas.data import EmailData

    classifier_path = tmp_path / "logistic.joblib"
    classifier_path.write_text("placeholder", encoding="utf-8")
    captured = {}

    class FakeInnerModel:
        feature_names_in_ = np.array(["n_words", "topic", "topic_probability", "lora_0", "lora_1"])

    class FakeClassifier:
        model = FakeInnerModel()
        threshold = 0.6

        def predict_proba(self, X):
            captured["features"] = X.copy()
            return np.array([[0.25, 0.75]])

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, **params):
            return np.ones((len(texts), 2), dtype=float)

    class FakeMetaFeatureExtraction:
        def extract(self, df):
            return pd.DataFrame({"message_id": df["message_id"], "n_words": [3]})

    class BrokenTopicFeatureExtraction:
        def __init__(self, load_existing=False):
            raise FileNotFoundError("missing topic model")

    monkeypatch.setattr(app_module, "get_train_config", lambda: SimpleNamespace(embeddings_model="fake", embeddings_params={}, spacy_max_length=200))
    monkeypatch.setattr(app_module, "get_logistic_regression_config", lambda: SimpleNamespace(model_path=classifier_path))
    monkeypatch.setattr(app_module.LogisticClassifier, "load", classmethod(lambda cls, config=None: FakeClassifier()))
    monkeypatch.setattr(app_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(app_module, "MetaFeatureExtraction", FakeMetaFeatureExtraction)
    monkeypatch.setattr(app_module, "TopicFeatureExtraction", BrokenTopicFeatureExtraction)
    monkeypatch.setattr(app_module, "_get_similarity_features", lambda embeddings: pd.DataFrame({"max_similarity": [0.0]}))

    result = app_module.predict(EmailData(subject="Need this", body="Please review"))

    assert result.urgent is True
    assert result.proba == 0.75
    assert captured["features"]["topic"].tolist() == [-1]
    assert captured["features"]["topic_probability"].tolist() == [0.0]
