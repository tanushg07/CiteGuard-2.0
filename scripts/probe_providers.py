"""Live connectivity probe for the scholarly providers CiteGuard uses (diagnostic, not a test).

Usage: python scripts/probe_providers.py
Prints HTTP status, latency and a short summary per provider for one DOI and one title query.
"""
import time

import httpx

from backend.config import get_settings

settings = get_settings()
UA = {'User-Agent': f'CiteGuard/2.1 (mailto:{settings.api_contact_email})'}
DOI = '10.1038/nature02236'
TITLE = 'Functional genomic hypothesis generation and experimentation by a robot scientist'
CHECKS = [
    ('crossref doi', f'https://api.crossref.org/works/{DOI}', {}),
    ('crossref biblio', 'https://api.crossref.org/works', {'query.bibliographic': TITLE + ' King 2004', 'rows': 3}),
    ('openalex doi', f'https://api.openalex.org/works/https://doi.org/{DOI}', {}),
    ('openalex search', 'https://api.openalex.org/works', {'search': TITLE, 'per-page': 3}),
    ('semantic scholar', 'https://api.semanticscholar.org/graph/v1/paper/search/match',
     {'query': TITLE, 'fields': 'title,year,externalIds,openAccessPdf'}),
    ('unpaywall', f'https://api.unpaywall.org/v2/{DOI}', {'email': settings.api_contact_email}),
    ('arxiv', 'https://export.arxiv.org/api/query', {'id_list': '2310.01798', 'max_results': 1}),
    ('europepmc', 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
     {'query': f'DOI:{DOI}', 'format': 'json', 'resultType': 'core'}),
]
with httpx.Client(timeout=20, follow_redirects=True, headers=UA) as client:
    for name, url, params in CHECKS:
        started = time.monotonic()
        try:
            response = client.get(url, params=params)
            body = response.text[:160].replace('\n', ' ')
            print(f'{name:18} HTTP {response.status_code} {time.monotonic() - started:5.2f}s {body}')
        except httpx.HTTPError as exc:
            print(f'{name:18} ERROR {type(exc).__name__}: {exc}')
