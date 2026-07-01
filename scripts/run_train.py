from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold

from configs.train_config import get_label_config, get_train_config
from src.emailurgency.models.labeler import WeakLabeler
from src.emailurgency.models.lexicon import Lexicon
from src.emailurgency.models.lora import LoRA
from src.emailurgency.models.classifier import LogisticClassifier
from src.emailurgency.pipelines.feature_extraction import MetaFeatureExtraction, SimilarityFeatureExtraction, TopicFeatureExtraction
from src.emailurgency.pipelines.preprocess import PreProcess

def parse_args():
    args = argparse.ArgumentParser(description='Train Email Urgency Classifier')
    args.add_argument('--sample', type=int, default=None, help='Sample size')
    args.add_argument('--saved_model', action='store_true', help='Use saved model')
    args.add_argument('--use_lora', action='store_true', help='Use LoRA embeddings')
    return args.parse_args()

def main() -> None:
    train_config = get_train_config()
    args = parse_args()
    train = pd.read_parquet(train_config.train_path, engine='pyarrow')
    if args.sample:
        train = train.head(args.sample)
  
    metafeatures = MetaFeatureExtraction().extract(train)
    train = train.merge(metafeatures, on='message_id', how='left')
  
    preprocess = PreProcess()
    train['subject'] = preprocess.preprocess_subject(train['subject'])
    train['body'] = preprocess.preprocess_body(train['body']).str.slice(0, train_config.spacy_max_length)

    embeddings_model = SentenceTransformer(train_config.embeddings_model)
    embeddings = embeddings_model.encode(train['body'].fillna('').astype(str).tolist(), **train_config.embeddings_params)
    
    label_config = get_label_config()
    can_load_labeler = args.saved_model and label_config.model_path.exists()
    if not can_load_labeler:
        lexicon = Lexicon()
        lexicon.build_lexicon(embeddings, train['body'])

    if can_load_labeler:
        weaklabeler = WeakLabeler.load(config=label_config)
    else:
        weaklabeler = WeakLabeler(config=label_config)
        weaklabeler.fit(train.drop(columns='gold_labels'), train['gold_labels'], embeddings)
        weaklabeler.save()

    train['silver_label'] = weaklabeler.predict(train, embeddings)
    train['label'] = train['gold_labels'].where(train['gold_labels'].isin([0, 1]), train['silver_label']).astype(int)
    save_training_embeddings(train_config.embeddings_artifact_path, embeddings, train['label'].to_numpy())
    
    simfeatures = SimilarityFeatureExtraction()
    train_sim = simfeatures.extract(embeddings, train['label'].to_numpy())

    topicfeatures = TopicFeatureExtraction()
    train_topics = topicfeatures.extract(train['body'], embeddings)

    train_text = train['body'].fillna('').astype(str).tolist()
    if args.use_lora:
        lora = LoRA()
        if not args.saved_model or not lora.config.merged_path.exists():
            lora.fit(train_text, train['label'].to_numpy())
        train_lora = pd.DataFrame(lora.embeddings(train_text)).add_prefix('lora_')
    else:
        train_lora = pd.DataFrame(embeddings).add_prefix('lora_')

    meta_cols = [col for col in metafeatures.columns if col != 'message_id']
    train_meta = train[meta_cols].reset_index(drop=True).fillna(0)

    X_train = pd.concat([train_meta, train_sim, train_topics, train_lora], axis=1).fillna(0)

    kcv = StratifiedKFold(**train_config.kfold_params)
    y_train = train['label'].to_numpy()
    oof_proba = np.zeros(len(train), dtype=float)
    train['fold'] = -1

    for fold, (fit_idx, val_idx) in enumerate(kcv.split(X_train, y_train)):
        fold_logistic = LogisticClassifier()
        fold_logistic.fit(X_train.iloc[fit_idx], y_train[fit_idx])
        oof_proba[val_idx] = fold_logistic.predict_proba(X_train.iloc[val_idx])[:, 1]
        train.loc[train.index[val_idx], 'fold'] = fold

    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (oof_proba >= threshold).astype(int)
        tp = ((pred == 1) & (y_train == 1)).sum()
        fp = ((pred == 1) & (y_train == 0)).sum()
        fn = ((pred == 0) & (y_train == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logistic = LogisticClassifier()
    can_load_logistic = False
    if args.saved_model and logistic.config.model_path.exists():
        loaded_logistic = LogisticClassifier.load(config=logistic.config)
        if has_matching_feature_schema(loaded_logistic, X_train):
            logistic = loaded_logistic
            can_load_logistic = True

    if not can_load_logistic:
        logistic.fit(X_train, y_train)
        logistic.save(threshold=best_threshold)
        
    train['proba'] = logistic.predict_proba(X_train)[:, 1]
    logistic.threshold = best_threshold
    logistic.save()
    train['pred'] = (train['proba'] >= best_threshold).astype(int)
    train['threshold'] = best_threshold
    train['cv_proba'] = oof_proba

    dataset = pd.concat([train.reset_index(drop=True), train_sim, train_topics, train_lora], axis=1)
    dataset.to_parquet(train_config.output_path, engine='pyarrow', index=False)

def has_matching_feature_schema(logistic: LogisticClassifier, X: pd.DataFrame) -> bool:
    fitted_features = getattr(logistic.model, 'feature_names_in_', None)
    if fitted_features is None:
        return False
    return list(fitted_features) == list(X.columns)

def save_training_embeddings(path, embeddings: np.ndarray, labels: np.ndarray) -> None:
    """Persist training embeddings and labels for similarity features at eval/API time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=np.asarray(embeddings, dtype=np.float32), labels=np.asarray(labels, dtype=int))

if __name__ == "__main__":
    main()
