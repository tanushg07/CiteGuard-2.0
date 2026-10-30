"""Validated configuration shared by services and command-line evaluation."""
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / '.env', env_file_encoding='utf-8', extra='ignore',
        env_ignore_empty=True,
    )
    groq_api_key: SecretStr = SecretStr('')
    groq_timeout_seconds: float = Field(default=5, gt=0, le=30)
    crossref_enabled: bool = True
    api_contact_email: str = 'citeguard@example.com'
    crossref_timeout_seconds: float = Field(default=3, gt=0, le=10)
    openalex_enabled: bool = True
    arxiv_enabled: bool = True
    openalex_api_key: SecretStr = SecretStr('')
    # Additional legally accessible providers used by the v2.1 identification / open-access pipeline.
    semantic_scholar_enabled: bool = True
    semantic_scholar_api_key: SecretStr = SecretStr('')
    unpaywall_enabled: bool = True
    europepmc_enabled: bool = True
    # Timeout for one scholarly-metadata API request (identification); separate from full-text downloads.
    provider_timeout_seconds: float = Field(default=10, gt=0, le=60)
    # Persistent reference-identification cache lifetime; 0 disables the cache.
    identification_cache_days: float = Field(default=7, ge=0, le=90)
    # OCR fallback for scanned PDFs: 'auto' uses an installed engine (rapidocr / tesseract), 'off' disables it.
    ocr_mode: Literal['auto', 'off'] = 'auto'
    ocr_max_pages: int = Field(default=30, ge=1, le=300)
    source_timeout_seconds: float = Field(default=15, gt=0, le=60)
    source_max_bytes: int = Field(default=25 * 1024 * 1024, gt=0, le=100 * 1024 * 1024)
    source_concurrency: int = Field(default=4, ge=1, le=8)
    citeguard_mode: Literal['neural', 'baseline'] = 'neural'
    citeguard_allow_download: bool = False
    citeguard_warmup: bool = True  # load + probe models in the background at server start and log readiness
    # Must be identical for scripts/prepare_models.py and the server, otherwise weights are "missing" at runtime.
    model_cache_dir: str = ''
    # Hard ceiling on the retrieval WORK for one reference (retries included). The clock starts when the reference
    # acquires a concurrency slot - time spent queued behind other references never counts against it.
    source_reference_deadline_seconds: float = Field(default=90.0, gt=0, le=600)
    # Ceiling for one full-text location (redirects + retries), so a single slow host cannot consume the deadline.
    source_url_budget_seconds: float = Field(default=40.0, gt=0, le=300)
    # At most this many candidate full-text locations are tried per reference.
    source_max_locations: int = Field(default=6, ge=1, le=12)
    source_max_retries: int = Field(default=2, ge=0, le=5)
    source_backoff_seconds: float = Field(default=1.0, ge=0, le=30)
    source_retry_after_cap_seconds: float = Field(default=20.0, ge=0, le=120)


def get_settings() -> Settings:
    return Settings()
