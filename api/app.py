from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from configs.train_config import get_label_config, get_logistic_regression_config, get_train_config
from src.emailurgency.models.classifier import LogisticClassifier
from src.emailurgency.models.labeler import WeakLabeler
from src.emailurgency.pipelines.feature_extraction import MetaFeatureExtraction, SimilarityFeatureExtraction, TopicFeatureExtraction
from src.emailurgency.pipelines.preprocess import PreProcess
from src.emailurgency.schemas.data import EmailData
from src.emailurgency.schemas.models import Prediction

app = FastAPI(title='Email Urgency Classifier', description='Trained on weak labels derived from email dataset', version='0.1.0')

def _build_feature_frame(x: EmailData) -> pd.DataFrame:
    payload = x.model_dump(exclude_none=True)
    payload.setdefault('message_id', f'api-{uuid4().hex}')
    payload.setdefault('sender', payload.get('x_from') or '')
    payload.setdefault('date', pd.Timestamp.now('UTC'))
    payload.setdefault('subject', '')
    payload.setdefault('body', '')
    payload.setdefault('to', [])
    payload.setdefault('cc', [])
    payload.setdefault('bcc', [])

    sender = str(payload.get('sender') or '')
    sender_domain = sender.split('@')[-1].lower() if '@' in sender else ''
    recipients = []
    for field in ('to', 'cc', 'bcc'):
        recipients.extend(payload.get(field) or [])
    recipient_domains = sorted({str(email).split('@')[-1].lower() for email in recipients if '@' in str(email)})
    payload.setdefault('sender_domain', sender_domain)
    payload.setdefault('recipient_domain', recipient_domains)

    return pd.DataFrame([payload])


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'logistic regression on LoRA embeddings + metafeatures'}

@app.post('/predict', response_model=Prediction)
def predict(x: EmailData):
    train_config = get_train_config()
    label_config = get_label_config()
    classifier_config = get_logistic_regression_config()

    test = _build_feature_frame(x)

    metafeatures = MetaFeatureExtraction().extract(test)
    test = test.merge(metafeatures, on='message_id', how='left')

    preprocess = PreProcess()
    test['subject'] = preprocess.preprocess_subject(test['subject'])
    test['body'] = preprocess.preprocess_body(test['body']).str.slice(0, train_config.spacy_max_length)

    embeddings_model = SentenceTransformer(train_config.embeddings_model)
    test_text = test['body'].fillna('').astype(str).tolist()
    embeddings = embeddings_model.encode(test_text, **train_config.embeddings_params)

    try:
        weaklabeler = WeakLabeler.load(config=label_config)
        test['silver_label'] = weaklabeler.predict(test, embeddings)
    except Exception:
        test['silver_label'] = 0

    gold_labels = test['gold_labels'] if 'gold_labels' in test.columns else pd.Series(np.nan, index=test.index)
    test['label'] = gold_labels.where(gold_labels.isin([0, 1]), test['silver_label']).astype(int)

    simfeatures = SimilarityFeatureExtraction()
    test_sim = simfeatures.extract_new(embeddings)

    try:
        topicfeatures = TopicFeatureExtraction(load_existing=True)
        test_topics = topicfeatures.extract_new(test['body'], embeddings)
    except Exception:
        test_topics = pd.DataFrame({'topic': [-1], 'topic_probability': [0.0]})

    test_lora = pd.DataFrame(embeddings).add_prefix('lora_')

    meta_cols = [col for col in metafeatures.columns if col != 'message_id']
    test_meta = test[meta_cols].reset_index(drop=True).fillna(0)

    X_test = pd.concat([test_meta, test_sim, test_topics, test_lora], axis=1).fillna(0)

    logistic = LogisticClassifier.load(config=classifier_config)
    expected_columns = list(getattr(logistic.model, 'feature_names_in_', []))
    if not expected_columns:
        raise HTTPException(status_code=500, detail='The loaded classifier does not expose its expected feature names.')

    X_test = X_test.reindex(columns=expected_columns, fill_value=0.0)

    threshold = float(getattr(logistic, 'threshold', 0.5))
    proba = float(logistic.predict_proba(X_test)[:, 1][0])
    return Prediction(urgent=bool(proba >= threshold), proba=proba, threshold=threshold)
