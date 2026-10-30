"""Test-session defaults: unit tests must never trigger model loading or network access.

Every scholarly provider is disabled and the persistent identification cache is off unless a test opts in with
explicit settings and a mocked httpx transport. Live provider tests live in tests/live and only run when
CITEGUARD_LIVE_TESTS=1 (they read the real .env instead).
"""
import os

LIVE = os.environ.get('CITEGUARD_LIVE_TESTS') == '1'

os.environ.setdefault('CITEGUARD_WARMUP', 'false')
if not LIVE:
    for name in ('CROSSREF_ENABLED', 'OPENALEX_ENABLED', 'ARXIV_ENABLED', 'SEMANTIC_SCHOLAR_ENABLED',
                 'UNPAYWALL_ENABLED', 'EUROPEPMC_ENABLED'):
        os.environ[name] = 'false'
    os.environ['IDENTIFICATION_CACHE_DAYS'] = '0'
    os.environ['GROQ_API_KEY'] = ''


def pytest_collection_modifyitems(config, items):
    import pytest
    skip = pytest.mark.skip(reason='live provider test; set CITEGUARD_LIVE_TESTS=1 to run')
    for item in items:
        if 'live' in item.keywords and not LIVE:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line('markers', 'live: real network calls to scholarly providers (opt-in)')
