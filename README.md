# Email Urgency Classifier

This project builds an email urgency classifier for an unlabeled email dataset. Because most emails do not have human labels, the pipeline first creates weak labels from lexicons, semantic similarity, metadata, and topic/similarity signals. A logistic regression classifier is then trained on these weak labels based on meta-, topic-, similarity-, and embeddingsfeature (incl. a LoRA fine-tuning framework), while a small manually labeled golden-label set is used to tune the prediction threshold and calibrate the final urgent/not-urgent decision.

## How to Run the Model

Install dependencies with Poetry:

```powershell
make install
```

Prepare the dataset:

```powershell
make data
```

Train the weak-label classifier:

```powershell
make train
```

Evaluate on the test split:

```powershell
make val
```

Run monitoring outputs:

```powershell
make monitor
```

Start the FastAPI service:

```powershell
make api
```

The API is served on:

```text
http://localhost:8000
```

Docker build/run:

```powershell
docker build -t email-urgency-classifier .
docker run --rm -p 8000:8000 email-urgency-classifier
```

## Repo Structure

```text
api/                 FastAPI app and prediction endpoints
artifacts/           Saved labeler, classifier, transformer, LoRA, and model artifacts
configs/             Data, training, model, and API configuration
data/                Local datasets and generated parquet outputs
lexicon/             Seed lexicons and preprocessing rules
notebooks/           Exploration and experiment notebooks
scripts/             CLI entry points for data, train, eval, monitor, and API
src/emailurgency/    Core package: schemas, models, and pipelines
tests/               Unit tests
```

## Key Results & Findings

- Weak supervision is useful for bootstrapping labels when the email dataset is mostly unlabeled.
- Lexicon and semantic signals provide interpretable urgency cues, while embedding, similarity, topic, and metadata features improve coverage beyond exact keyword matches.
- Golden labels are kept separate from weak-label generation and used to tune the classifier threshold, reducing dependence on the default `0.5` cutoff.
- The trained output is saved to `artifacts/classifier/logistic.joblib`; enriched training/evaluation datasets are written under `data/`.
- Current training supports either saved weak-label/model artifacts or refitting when feature schemas change.

## Predict API

Health check:

```bash
curl http://localhost:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "example-1",
    "date": "2001-01-01T09:00:00Z",
    "from": "sender@example.com",
    "to": ["recipient@example.com"],
    "subject": "Need approval today",
    "body": "Can you review and approve this before close of business?"
  }'
```

Intended response schema:

```json
{
  "urgent": true,
  "proba": 0.82,
  "threshold": 0.57
}
```

## Notes and Next Steps

- Expand the manually labeled golden set to improve threshold tuning, calibration, and evaluation confidence.
- If more training time becomes available, fine-tune the embedding model with LoRA to better adapt it to the email domain.
- Continue deeper investigation into BERTopic to improve topic quality and interpretability.
- Create a clear action list for monitoring, stakeholder review, and ongoing model iteration.
- Define urgency more precisely by separating it from related concepts such as spam, bulk, and important emails.
- Strengthen the positive and negative signals used for interpretation so the model’s predictions are easier to explain and validate.

