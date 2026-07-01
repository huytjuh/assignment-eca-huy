from __future__ import annotations

import argparse

import pandas as pd
from sentence_transformers import SentenceTransformer

from configs.train_config import get_label_config, get_train_config
from src.emailurgency.models.classifier import LogisticClassifier
from src.emailurgency.models.labeler import WeakLabeler
from src.emailurgency.models.lora import LoRA
from src.emailurgency.pipelines.feature_extraction import MetaFeatureExtraction, SimilarityFeatureExtraction, TopicFeatureExtraction
from src.emailurgency.pipelines.preprocess import PreProcess

def parse_args():
    args = argparse.ArgumentParser(description="Evaluate Email Urgency Classifier")
    args.add_argument("--sample", type=int, default=None, help="Sample size")
    return args.parse_args()

def main() -> None:
    train_config = get_train_config()
    label_config = get_label_config()
    args = parse_args()

    test = pd.read_parquet(train_config.test_path, engine="pyarrow")
    if args.sample:
        test = test.head(args.sample)

    metafeatures = MetaFeatureExtraction().extract(test)
    test = test.merge(metafeatures, on="message_id", how="left")

    preprocess = PreProcess()
    test["subject"] = preprocess.preprocess_subject(test["subject"])
    test["body"] = preprocess.preprocess_body(test["body"]).str.slice(0, train_config.spacy_max_length)

    embeddings_model = SentenceTransformer(train_config.embeddings_model)
    test_text = test["body"].fillna("").astype(str).tolist()
    embeddings = embeddings_model.encode(test_text, **train_config.embeddings_params)

    weaklabeler = WeakLabeler.load(config=label_config)
    test["silver_label"] = weaklabeler.predict(test, embeddings)

    test["label"] = test["gold_labels"].where(test["gold_labels"].isin([0, 1]), test["silver_label"]).astype(int)

    simfeatures = SimilarityFeatureExtraction()
    test_sim = simfeatures.extract(embeddings, test["label"].to_numpy())

    topicfeatures = TopicFeatureExtraction()
    test_topics = topicfeatures.extract_new(test["body"], embeddings)

    lora = LoRA()
    test_lora = pd.DataFrame(lora.embeddings(test_text)).add_prefix("lora_")

    meta_cols = [col for col in metafeatures.columns if col != "message_id"]
    test_meta = test[meta_cols].reset_index(drop=True).fillna(0)

    X_test = pd.concat([test_meta, test_sim, test_topics, test_lora], axis=1).fillna(0)

    logistic = LogisticClassifier()
    logistic = LogisticClassifier.load(config=logistic.config)
    threshold = getattr(logistic, "threshold", 0.5)

    test["proba"] = logistic.predict_proba(X_test)[:, 1]
    test["pred"] = (test["proba"] >= threshold).astype(int)
    test["threshold"] = threshold

    dataset = pd.concat([test.reset_index(drop=True), test_sim, test_topics, test_lora], axis=1)
    dataset.to_parquet(train_config.eval_output_path, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
