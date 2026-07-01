from __future__ import annotations

from pydantic import BaseModel, ConfigDict

class MetaFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    message_id: str

    response_time: int | None = None
    time_since_last_email: int | None = None
    sent_hour: int
    sent_weekday: int
    is_after_hours: bool
    is_weekend: bool

    n_recipient: int
    n_recipient_domains: int
    n_words: int
    n_sentences: int

    is_internal: bool
    is_external: bool

class SimilarityFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    max_similarity: float
    mean_similarity: float
    std_similarity: float
    top_k_mean_similarity: float
    top_k_std_similarity: float
    top_k_pos_similarity: float
    top_k_neg_similarity: float
    gap_similarity: float

