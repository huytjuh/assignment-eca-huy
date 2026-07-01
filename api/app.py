from __future__ import annotations

from src.emailurgency.schemas.data import EmailData
from src.emailurgency.schemas.models import Prediction
from src.emailurgency.models.classifier import LogisticClassifier

from fastapi import FastAPI

app = FastAPI(title='Email Urgency Classifier', description='Trained on weak labels derived from email dataset', version='0.1.0')

@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'logistic regression on LoRA embeddings + metafeatures'}

@app.post('/predict', response_model=Prediction)
def predict(x: EmailData):
    model = LogisticClassifier.load()
    return {'status': 'ok'}
