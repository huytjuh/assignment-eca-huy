from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_saved_classifier_schema_must_match_exact_feature_order():
    from scripts.run_train import has_matching_feature_schema

    logistic = SimpleNamespace(model=SimpleNamespace(feature_names_in_=np.array(["meta", "lora_0"])))

    assert has_matching_feature_schema(logistic, pd.DataFrame(columns=["meta", "lora_0"]))
    assert not has_matching_feature_schema(logistic, pd.DataFrame(columns=["lora_0", "meta"]))
    assert not has_matching_feature_schema(SimpleNamespace(model=object()), pd.DataFrame(columns=["meta"]))


def test_weak_labeler_pattern_and_semantic_features_are_synthetic():
    from src.emailurgency.models.labeler import WeakLabeler

    labeler = WeakLabeler.__new__(WeakLabeler)
    labeler.lexicons = {"urgency": pd.DataFrame({"prototype": ["asap"]})}
    labeler.semantic_embeddings = {"urgency": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}

    pattern = WeakLabeler._compile_pattern(["asap", "close of business"])
    enriched = labeler._add_semantic_features(
        pd.DataFrame({"subject": ["x", "y"], "body": ["a", "b"]}),
        np.array([[0.2, 0.8], [1.0, 0.0]], dtype=np.float32),
    )

    assert pattern.search("please handle this ASAP")
    assert not pattern.search("asphalt")
    assert enriched["urgency_semantic_similarity"].tolist() == pytest.approx([0.8, 1.0])


def test_lexicon_initializes_seed_files_only(tmp_path):
    from src.emailurgency.models.lexicon import Lexicon

    valid = tmp_path / "urgency.csv"
    invalid = tmp_path / "notes.csv"
    valid.write_text(
        "prototype,source,similarity\nASAP,Seed,1.0\nreview,Generated,0.9\n",
        encoding="utf-8",
    )
    invalid.write_text("term\nmissing-schema\n", encoding="utf-8")

    lexicon = Lexicon.__new__(Lexicon)
    lexicon.config = SimpleNamespace(lexicon_path=tmp_path)
    seeds = lexicon._initialize_seeds()

    assert list(seeds) == [valid]
    assert seeds[valid]["prototype"].tolist() == ["ASAP"]


def test_logistic_classifier_fits_predicts_and_saves_threshold(tmp_path):
    from src.emailurgency.models.classifier import LogisticClassifier

    config = SimpleNamespace(
        model_path=tmp_path / "logistic.joblib",
        model_params={"solver": "lbfgs", "max_iter": 200, "class_weight": None, "random_state": 42},
    )
    X = pd.DataFrame({"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})
    y = np.array([0, 0, 1, 1])

    clf = LogisticClassifier(config=config).fit(X, y)
    clf.save(threshold=0.7)
    loaded = LogisticClassifier.load(config=config)

    assert config.model_path.exists()
    assert loaded.threshold == 0.7
    assert loaded.predict_proba(X).shape == (4, 2)


def test_lora_dataset_tokenizes_text_and_casts_labels():
    from src.emailurgency.models.lora import LoRA

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2] for _ in texts], "attention_mask": [[1, 1] for _ in texts]}

    lora = LoRA.__new__(LoRA)
    lora.config = SimpleNamespace(max_length=8)
    lora.tokenizer = Tokenizer()

    dataset = lora._dataset(["a", "b"], np.array([0.0, 1.0]))

    assert dataset["labels"] == [0, 1]
    assert dataset["input_ids"] == [[1, 2], [1, 2]]


def test_lora_embeddings_requires_saved_merged_model(tmp_path):
    from src.emailurgency.models.lora import LoRA

    lora = LoRA.__new__(LoRA)
    lora.config = SimpleNamespace(merged_path=tmp_path / "missing")

    try:
        lora.embeddings(["hello"])
    except FileNotFoundError as exc:
        assert "Merged LoRA model not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

def test_save_training_embeddings_writes_npz(tmp_path):
    from scripts.run_train import save_training_embeddings

    path = tmp_path / "embeddings" / "train_embeddings.npz"
    embeddings = np.ones((2, 3), dtype=float)
    labels = np.array([0, 1])

    save_training_embeddings(path, embeddings, labels)

    artifact = np.load(path)
    assert artifact["embeddings"].dtype == np.float32
    assert artifact["embeddings"].shape == (2, 3)
    assert artifact["labels"].tolist() == [0, 1]

def test_similarity_extract_new_loads_saved_training_artifact(monkeypatch, tmp_path):
    from src.emailurgency.pipelines.feature_extraction import SimilarityFeatureExtraction

    artifact_path = tmp_path / "train_embeddings.npz"
    np.savez_compressed(
        artifact_path,
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.array([0, 1]),
    )
    monkeypatch.setattr(
        "configs.train_config.get_train_config",
        lambda: SimpleNamespace(embeddings_artifact_path=artifact_path),
    )

    extractor = SimilarityFeatureExtraction()
    features = extractor.extract_new(np.array([[1.0, 0.0]], dtype=np.float32))

    assert extractor.reference_embeddings is not None
    assert extractor.y.tolist() == [0, 1]
    assert features.shape[0] == 1
    assert "max_similarity" in features.columns
