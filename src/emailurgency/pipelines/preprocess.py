from __future__ import annotations

import re 
import pandas as pd

from pathlib import Path
from html import unescape
from bs4 import BeautifulSoup
import spacy

from sklearn.feature_extraction.text import CountVectorizer

from configs.global_config import PreProcessConfig, get_preprocess_config

class PreProcess:
    """Preprocess emails using spacy and regex to remove non-english words and punctuation"""

    def __init__(self, config: PreProcessConfig | None = None, use_spacy: bool = True) -> None:
        """Initialize PreProcess"""
        self.config = config or get_preprocess_config()
        self.nlp = spacy.load(self.config.spacy_model) if use_spacy else None

        self.html_patterns = re.compile(r"</?[a-z][\s\S]*?>|&(?:amp|lt|gt|quot|nbsp);", re.I)
        self.url_patterns = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
        self.email_patterns = re.compile(r'[\w\.-]+@[\w\.-]+', re.IGNORECASE)
        self.phone_patterns = re.compile(r'\d{3}-\d{3}-\d{4}', re.IGNORECASE)

        self.whitespace_patterns = re.compile(r'\s+', re.IGNORECASE)
        self.punctuation_patterns = re.compile(r"([!?.,;:])\1+", re.IGNORECASE)

        self.chain_patterns = re.compile(
            r"""(?imx)
            ^\s*(
                -{2,}\s*original\s+message\s*-{2,}
                |
                -{2,}\s*forwarded\s+by.*?-{2,}
                |
                -{2,}\s*forwarded\s+message\s*-{2,}
                |
                on\s+.+?\s+wrote:
                |
                from:\s*.+
                \n\s*(sent|date):\s*.+
                \n\s*to:\s*.+
            )
            """,
        )

    def preprocess_subject(self, subject: pd.Series) -> pd.Series:
        """Clean subject"""
        subject = subject.fillna('').astype(str).str.lower()
        subject = subject.str.replace(self.chain_patterns, "", regex=True)
        return subject.str.strip()
    
    def preprocess_body(self, body: pd.Series) -> list[str]:
        """Preprocess emails using spacy and regex to remove non-english words and punctuation"""
        if isinstance(body, list):
            body = pd.Series(body)

        body = self._clean_html(body)

        # basic preprocessing
        body = self._normalize_text(body)
        body = self._normalize_entities(body)

        # lexicon-based preprocessing
        body = self._remove_boilerplate(body)
        body = self._remove_signature(body)
        body = self._remove_thread(body)

        # final preprocessing 
        body = self._remove_whitespace(body)
        body = self._remove_punctuation(body)

        return body
    
    def _clean_html(self, corpus: pd.Series) -> pd.Series:
        """Clean HTML"""
        def clean(text):
            if pd.isna(text):
                return ""
            
            if not self.html_patterns.search(text):
                return text

            text = unescape(str(text))
            soup = BeautifulSoup(text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            return soup.get_text(separator=" ", strip=True)

        return corpus.apply(clean)
        
    def _normalize_text(self, corpus: pd.Series) -> pd.Series:
        """Normalize text"""
        corpus = corpus.fillna('').str.lower()
        corpus = corpus.str.replace('\n', ' ')
        return corpus
    
    def _normalize_entities(self, corpus: pd.Series) -> pd.Series:
        """Normalize entities"""
        corpus = corpus.str.replace(self.url_patterns, '', regex=True)
        corpus = corpus.str.replace(self.email_patterns, '', regex=True)
        corpus = corpus.str.replace(self.phone_patterns, '', regex=True)
        corpus = corpus.str.replace('[\U0001F600-\U0001F64F]', '', regex=True)
        return corpus
    
    def _remove_boilerplate(self, corpus: pd.Series) -> pd.Series:
        """Remove boilerplate based on lexicon patterns"""
        boilerplate_file = self._resolve_lexicon_file(self.config.boilerplate_file)
        if not boilerplate_file.exists() or boilerplate_file.stat().st_size == 0:
            self._build_lexicon_boilerplate(boilerplate_file, corpus)

        boilerplate_re = self._compile_lexicon_regex(boilerplate_file)
        return corpus.apply(lambda x: boilerplate_re.split(str(x), maxsplit=1)[0].strip() if pd.notna(x) else '')
    
    def _remove_signature(self, corpus: pd.Series) -> pd.Series:
        """Remove signature based on lexicon patterns"""
        signature_file = self._resolve_lexicon_file(self.config.signature_file)
        if not signature_file.exists() or signature_file.stat().st_size == 0:
            return corpus

        signature_re = self._compile_lexicon_regex(signature_file, line_end=True)
        
        def remove(text: str) -> str:
            if pd.isna(text):
                return ''
            
            lines = str(text).splitlines()
            start = max(0, len(lines) - 12)

            for i in range(start, len(lines)):
                if signature_re.match(lines[i]):
                    return "\n".join(lines[:i]).strip()

            return str(text).strip()
        
        return corpus.apply(remove)

    def _remove_thread(self, corpus: pd.Series) -> pd.Series:
        return corpus.apply(lambda x: self.chain_patterns.split(str(x), maxsplit=1)[0].strip() if pd.notna(x) else '')
    
    def _remove_whitespace(self, corpus: pd.Series) -> pd.Series:
        return corpus.str.replace(self.whitespace_patterns, ' ', regex=True)
    
    def _remove_punctuation(self, corpus: pd.Series) -> pd.Series:
        return corpus.str.replace(self.punctuation_patterns, r'\1', regex=True)
    
    def _remove_stopwords(self, corpus: pd.Series) -> pd.Series:
        return corpus.apply(lambda x: ' '.join([token.text for token in self.nlp(x) if not token.is_stop]))
    
    def _compile_lexicon_regex(self, file: Path, line_end: bool = False) -> re.Pattern:
        """Compile lexicon regex"""
        if file.suffix == '.parquet':
            terms = pd.read_parquet(file).iloc[:, 0].tolist()
        else:
            terms = pd.read_csv(file).iloc[:, 0].tolist()
        pattern = '|'.join(re.escape(t) for t in terms)
        pattern = rf"^\s*(?:{pattern})" + (r"\s*$" if line_end else "")
        return re.compile(pattern, flags=re.I | re.M)

    @staticmethod
    def _resolve_lexicon_file(file: Path) -> Path:
        """Use an existing same-stem parquet lexicon when the configured CSV is absent."""
        if file.exists():
            return file

        parquet_file = file.with_suffix('.parquet')
        return parquet_file if parquet_file.exists() else file

    @staticmethod
    def _build_lexicon_boilerplate(file: Path, corpus: pd.Series) -> None:
        """Build lexicon preprocess"""
        vectorizer = CountVectorizer(ngram_range=(3, 6), min_df=10, max_df=0.8, max_features=50000, stop_words='english')
        corpus = corpus.fillna('').astype(str).str[:1000] + ' ' + corpus.fillna('').astype(str).str[-1000:]
        X = vectorizer.fit_transform(corpus.str.lower())
        out = pd.DataFrame({'ngram': vectorizer.get_feature_names_out(), 'count': X.sum(0).A1 / X.shape[0]}).sort_values('count', ascending=False)
        out[out['count'] > .01].to_csv(file, index=False)
