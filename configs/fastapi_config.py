from __future__ import annotations

from pathlib import Path

from functools import lru_cache
from pydantic_settings import BaseSettings

class FastAPIConfig(BaseSettings):
    model_config = {'env_file': '.env'}

    data_path: Path = Path('data')

    base_url: str = 'http://localhost:8000'
    page_size: int = 1000

    host: str = '0.0.0.0'
    port: int = 8000

@lru_cache
def get_fastapi_config() -> FastAPIConfig:
    return FastAPIConfig()