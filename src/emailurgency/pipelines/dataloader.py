from __future__ import annotations

from pathlib import Path
import logging
from typing import Any
from pydantic import ValidationError

import pandas as pd
import requests
from tqdm import tqdm

from configs.fastapi_config import FastAPIConfig
from src.emailurgency.schemas.data import EmailData

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class using FastAPI"""

    def __init__(self, config: FastAPIConfig | None = None) -> None:
        """Initialize DataLoader"""
        self.config = config or FastAPIConfig()
    
    def fetch(self, offset: int) -> list[dict[str, Any]]:
        """Fetch emails from FastAPI"""
        params = {'offset': offset, 'limit': self.config.page_size}
        response = requests.get(f'{self.config.base_url}/emails', params=params, timeout=30)
        response.raise_for_status()
        return response.json()['emails']
    
    def parse(self, data: list[dict[str, Any]]) -> list[EmailData]:
        """Parse emails"""
        parsed: list[EmailData] = []
        for idx, email in enumerate(data):
            try:
                parsed.append(EmailData.model_validate(email))
            except ValidationError:
                logger.exception(f'Failed to parse email at index {idx}')
                raise

        logger.info(f'Parsing {len(data)} rows')
        return parsed
    
    def save(self, parsed: list[EmailData], path: Path) -> None:
        """Save emails to parquet file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([email.model_dump(by_alias=True) for email in parsed])
        df.to_parquet(path, engine='pyarrow', compression='zstd', index=False)
        logger.info(f'Saved {len(parsed)} emails to {path}')

    def get_statistics(self) -> int:
        """Get statistics from FastAPI"""
        response = requests.get(f'{self.config.base_url}/stats', timeout=30)
        response.raise_for_status()
        return response.json()['total_emails']
    
    def combine_chunks(self) -> None:
        """Combine chunks"""
        list_chunks = [pd.read_parquet(chunk, engine='pyarrow') for chunk in self.config.data_path.glob('chunks/*.parquet')]
        output = pd.concat(list_chunks, ignore_index=True)
        output.to_parquet(self.config.data_path / 'data.parquet', engine='pyarrow', compression='zstd')

    def run(self) -> pd.DataFrame:
        """Run data loader"""
        n_samples = self.get_statistics()
        logger.info(f'Number of emails: {n_samples}')

        for chunk_idx, offset in enumerate(tqdm(range(0, n_samples, self.config.page_size), desc='Fetching emails'), start=1):
            path_output = self.config.data_path / f'chunks/emails_{chunk_idx:03d}.parquet'
            if path_output.exists():
                logger.info(f'Skipping emails_{chunk_idx:03d}.parquet')
                continue

            try:
                batch = self.fetch(offset)
            except requests.exceptions.RequestException:
                logger.exception(f'Failed to fetch offset {offset}')
                raise

            if not batch:
                logger.warning(f'Empty batch {path_output}')
                break

            parsed = self.parse(batch)
            self.save(parsed, path_output)

        self.combine_chunks()
        logger.info(f'Download completed')

        return pd.read_parquet(self.config.data_path / 'data.parquet', engine='pyarrow')