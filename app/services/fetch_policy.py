"""Retrieval attempt bookkeeping, retry/backoff and outcome classification.

Pure functions only (no network, no third-party imports) so every branch is unit-testable.

Principles
- Never bypass access controls: 401/403 is recorded as ACCESS_DENIED and is not retried or worked around.
- "Inaccessible" (denied, rate-limited, timed out) is distinct from "does not exist" (404/410, unregistered DOI).
- Transient failures are retried a bounded number of times; the final outcome is still recorded.
- Every attempt is logged with provider, URL, timestamp, outcome and reason.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

TRANSIENT = frozenset({'TIMEOUT', 'NETWORK_ERROR', 'RATE_LIMITED', 'TRANSIENT_ERROR'})
DEFINITIVE_NEGATIVE = frozenset({'ACCESS_DENIED', 'NOT_FOUND', 'NOT_A_PDF', 'BLOCKED_ADDRESS', 'TOO_LARGE',
                                 'IDENTITY_MISMATCH', 'PARSE_FAILED'})
SUCCESS = frozenset({'FETCHED', 'CACHED'})

DOI_SYNTAX = re.compile(r'^10\.\d{4,9}/\S+$')
ARXIV_SYNTAX = re.compile(r'^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$', re.I)


@dataclass
class Attempt:
    provider: str
    url: str
    outcome: str
    reason: str = ''
    http_status: int | None = None
    timestamp: str = ''
    duration_seconds: float | None = None
    attempt_number: int = 1
    cached: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat(timespec='seconds')

    def as_dict(self) -> dict:
        return asdict(self)


def provider_for(url: str) -> str:
    host = re.sub(r'^https?://', '', url).split('/')[0].lower()
    if host.endswith('arxiv.org'):
        return 'arxiv'
    if host in ('doi.org', 'dx.doi.org'):
        return 'doi_resolver'
    if host.endswith('openalex.org'):
        return 'openalex'
    if host.endswith('crossref.org'):
        return 'crossref'
    return 'publisher_or_repository:' + host


def classify_http(status: int) -> tuple[str, bool]:
    """(outcome, retryable)."""
    if status == 200:
        return 'FETCHED', False
    if status in (401, 403):
        return 'ACCESS_DENIED', False
    if status in (404, 410):
        return 'NOT_FOUND', False
    if status == 429:
        return 'RATE_LIMITED', True
    if status in (408, 425, 500, 502, 503, 504):
        return 'TRANSIENT_ERROR', True
    return 'HTTP_ERROR', False


def parse_retry_after(value: str | None, cap: float, now: datetime | None = None) -> float | None:
    """Seconds to wait, capped; None if absent/unparseable. Supports delta-seconds and HTTP-date."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = (when - (now or datetime.now(UTC))).total_seconds()
    return max(0.0, min(seconds, cap))


def backoff_delay(retry_number: int, base: float, retry_after: float | None = None, cap: float = 20.0) -> float:
    """retry_number starts at 0. A server-provided Retry-After wins over exponential backoff."""
    if retry_after is not None:
        return min(retry_after, cap)
    return min(base * (2 ** retry_number), cap)


def classify_identifier(doi: str | None = None, arxiv_id: str | None = None) -> tuple[str, str]:
    """Syntactic check before any network call. ('VALID'|'MALFORMED'|'ABSENT', explanation)."""
    if doi:
        d = doi.strip().removeprefix('https://doi.org/').removeprefix('http://doi.org/')
        return ('VALID', 'DOI is syntactically valid.') if DOI_SYNTAX.match(d) else \
               ('MALFORMED', f'"{doi}" is not a syntactically valid DOI; nothing was requested for it.')
    if arxiv_id:
        return ('VALID', 'arXiv id is syntactically valid.') if ARXIV_SYNTAX.match(arxiv_id.strip()) else \
               ('MALFORMED', f'"{arxiv_id}" is not a syntactically valid arXiv identifier.')
    return 'ABSENT', 'No DOI or arXiv identifier available.'


def summarize_attempts(attempts: list[dict], source_identified: bool) -> tuple[str, str]:
    """Decide the source status when NO usable full text was obtained (callers invoke this only on failure;
    a FETCHED attempt can still be a wrong paper or unreadable PDF, so it is not treated as success here).

    - any transient failure and nothing definitive => RETRIEVAL_FAILED (could not determine; retry later)
    - identified but only definitive negatives     => SOURCE_METADATA_ONLY (exists; text inaccessible)
    - not identified                               => SOURCE_UNAVAILABLE
    """
    outcomes = [a.get('outcome') for a in attempts]
    transient = [a for a in attempts if a.get('outcome') in TRANSIENT]
    if transient and not any(o in DEFINITIVE_NEGATIVE for o in outcomes) and source_identified:
        last = transient[-1]
        return 'RETRIEVAL_FAILED', (f'Retrieval did not complete ({last["outcome"]} from {last["provider"]}); '
                                    'the source may be accessible on retry. Support was not assessed.')
    if source_identified:
        reasons = sorted({a.get('outcome', '?') for a in attempts})
        tail = f' Attempt outcomes: {", ".join(reasons)}.' if reasons else ' No open full-text location was found.'
        return 'SOURCE_METADATA_ONLY', 'Source identified, but full text could not be retrieved.' + tail + ' Metadata is not evidence.'
    return 'SOURCE_UNAVAILABLE', 'The cited source could not be reliably identified.'
