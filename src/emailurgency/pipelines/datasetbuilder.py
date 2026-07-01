from __future__ import annotations

import pandas as pd

import re
import hashlib

from configs.global_config import DataConfig, get_data_config

from src.emailurgency.schemas.data import ThreadData
from src.emailurgency.pipelines.preprocess import PreProcess

class DatasetBuilder:
    """Dataset builder class"""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or get_data_config()
        self.preprocess = PreProcess(use_spacy=self.config.use_spacy)
        
        self.forward_patterns = re.compile(r'^\s*(fw|fwd)\s*:', re.IGNORECASE)
        self.reply_patterns = re.compile(r'^\s*re\s*:', re.IGNORECASE)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build dataset"""
        if not all(col in X.columns for col in ['message_id', 'from', 'date', 'subject', 'body']):
            raise ValueError('X must have columns: message_id, from, date, subject, body')
        
        X_out = pd.DataFrame(X).copy().fillna('')
        X_out['is_forward'] = self._forward(X_out['subject'])
        X_out['is_reply'] = self._reply(X_out['subject'])
        X_out['subject'] = self.preprocess.clean_subject(X_out['subject'])

        X_out['thread_id'] = self._thread(X_out)
        X_out = X_out.sort_values(['thread_id', 'date'])
        X_out['parent_id'] = self._parent(X_out)
        X_out['parent_id'] = X_out['parent_id'].astype(object).where(X_out['parent_id'].notna(), None)
        X_out['sequence'] = self._sequence(X_out)
        X_out['sender_domain'] = self._domain(X_out['from'])
        X_out['recipient_domain'] = self._domain_recipients(X_out.loc[:, ['to', 'cc', 'bcc']])

        X_out = X_out.astype(object).where(pd.notna(X_out), None)
        validated = [ThreadData.model_validate(x) for x in X_out.to_dict('records')]
        return pd.DataFrame([row.model_dump() for row in validated])

    def _forward(self, subject: pd.Series) -> pd.Series:
        return subject.str.match(self.forward_patterns)
    
    def _reply(self, subject: pd.Series) -> pd.Series:
        return subject.str.match(self.reply_patterns)

    def _thread(self, X: pd.DataFrame) -> pd.Series:
        """Create thread ids"""
        X = X.fillna({'subject': ''})

        thread_ids = []
        for i, row in X.iterrows():
            subject = str(row['subject']).strip()
            is_thread = row['is_forward'] + row['is_reply'] > 0

            key = subject if is_thread else f"missing-{i}"
            thread_ids.append(hashlib.sha256(key.encode("utf-8")).hexdigest())

        return pd.Series(thread_ids, index=X.index)

    def _sequence(self, X: pd.DataFrame) -> pd.Series:
        return X.groupby('thread_id').cumcount().add(1).reindex(X.index)

    def _parent(self, X: pd.DataFrame) -> pd.Series:
        return X.groupby('thread_id')['message_id'].shift(1).reindex(X.index)
    
    def _domain(self, email: pd.Series) -> pd.Series:
        return email.fillna('').astype(str).str.extract(r"@([^>\s]+)", expand=False)
    
    def _domain_recipients(self, email: pd.DataFrame) -> pd.Series:
        return email.map(' '.join).agg(' '.join, axis=1).str.findall(r"@([^>\s,;]+)").map(lambda x: sorted(set(x)))