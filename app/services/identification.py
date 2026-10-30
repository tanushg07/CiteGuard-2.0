"""Reference identification: bibliography entry -> one validated scholarly work, or an explicit failure state.

Identification is deliberately separate from full-text availability: a correctly identified but paywalled paper
is IDENTIFIED_*, never "nonexistent".

Pipeline for one reference
1. Parse structured fields (title, authors, year, venue, volume, issue, pages, DOI, arXiv id).
2. Query legally accessible metadata providers, exact identifiers first:
      arXiv id  -> arXiv API (+ Semantic Scholar ARXIV: lookup)
      DOI       -> Crossref /works/{doi} + OpenAlex /works/doi
      otherwise -> Crossref query.bibliographic + OpenAlex title search, then Semantic Scholar title match,
                   then arXiv title search. A DOI-less reference is fully discoverable.
3. Score every candidate on several metadata signals. Title similarity alone never accepts a match: authors and
   publication year must agree (or the reference must lack them, which lowers confidence).
4. Merge candidates that are the same work (DOI / arXiv id / near-identical title + first author), detect
   genuinely different works with similar scores (AMBIGUOUS_SOURCE), and record every provider attempt.

Statuses: IDENTIFIED_HIGH_CONFIDENCE, IDENTIFIED_NEEDS_REVIEW, AMBIGUOUS_SOURCE, SOURCE_NOT_FOUND,
PROVIDER_ERROR, RATE_LIMITED. No paper, journal or DOI is hardcoded.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from urllib.parse import quote

import httpx

from app.services.document_parser import reference_record
from app.services.fetch_policy import Attempt, parse_retry_after
from backend.config import ROOT, get_settings

IDENTIFIED_HIGH = 'IDENTIFIED_HIGH_CONFIDENCE'
IDENTIFIED_REVIEW = 'IDENTIFIED_NEEDS_REVIEW'
AMBIGUOUS = 'AMBIGUOUS_SOURCE'
NOT_FOUND = 'SOURCE_NOT_FOUND'
PROVIDER_ERROR = 'PROVIDER_ERROR'
RATE_LIMITED = 'RATE_LIMITED'
IDENTIFIED = frozenset({IDENTIFIED_HIGH, IDENTIFIED_REVIEW})
CONCLUSIVE = frozenset({IDENTIFIED_HIGH, IDENTIFIED_REVIEW, AMBIGUOUS, NOT_FOUND})

CACHE_DIR = ROOT / 'data' / 'cache' / 'identification'
CACHE_VERSION = 'v2.1-2'  # bump whenever parsing/scoring changes (cache key)
STOPWORDS = frozenset('a an the of and or for in on to with by from at as is are via its their into using'.split())


# ---------------------------------------------------------------------------------------------- normalisation

def fold(text) -> str:
    text = unicodedata.normalize('NFKD', html.unescape(str(text or '')))
    return ''.join(c for c in text if not unicodedata.combining(c))


def norm_title(text) -> str:
    text = re.sub(r'<[^>]+>', ' ', fold(text).lower())
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return ' '.join(text.split())


def title_tokens(text) -> list[str]:
    return [t for t in norm_title(text).split() if t not in STOPWORDS]


def norm_surname(name) -> str:
    return re.sub(r'[^a-z]', '', fold(name).lower())


def title_similarity(reference_titles: list[str], candidate_title: str, reference_text: str = '') -> float:
    """Max over reference title readings of a character ratio and a token Dice score (0..1)."""
    cand = norm_title(candidate_title)
    if not cand:
        return 0.0
    best = 0.0
    cand_main = norm_title(re.split(r'[:?]', fold(candidate_title))[0])
    for ref in [t for t in reference_titles if t]:
        r = norm_title(ref)
        a, b = set(title_tokens(r)), set(title_tokens(cand))
        dice = 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0
        best = max(best, SequenceMatcher(None, r, cand).ratio(), dice)
        # Main titles (before ':' / '?') agree but one side omits the subtitle. Capped below the 0.92 acceptance
        # level, so a shared main title alone can never accept a match without an exact identifier.
        ref_main = norm_title(re.split(r'[:?]', fold(ref))[0])
        if len(ref_main) >= 12 and ref_main == cand_main:
            best = max(best, .9)
    if not reference_titles and reference_text:
        # No parsed title: fall back to containment of the candidate title in the whole reference string.
        whole = norm_title(reference_text)
        if len(cand) >= 15 and cand in whole:
            return 0.95
        toks = title_tokens(cand)
        if toks:
            best = max(best, .9 * sum(t in whole.split() for t in toks) / len(toks))
    return round(best, 4)


# ---------------------------------------------------------------------------------------------- data model

@dataclass
class Candidate:
    provider: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmcid: str | None = None
    abstract: str | None = None
    work_type: str | None = None
    provider_id: str | None = None
    oa_locations: list[dict] = field(default_factory=list)   # {'url', 'kind': pdf|landing|jats, 'provider', ...}
    match_score: float | None = None                         # provider-native score, informational only

    def surnames(self) -> list[str]:
        out = []
        for name in self.authors:
            parts = [p for p in re.split(r'[\s,]+', fold(name)) if p]
            if not parts:
                continue
            # 'Family, Given' is rarer in API output than 'Given Family'; providers here return the latter.
            out.append(norm_surname(parts[-1]))
        return [s for s in out if s]


@dataclass
class MatchScore:
    title_similarity: float
    first_author_match: bool | None
    author_overlap: float | None
    year: str            # exact | adjacent | mismatch | unknown
    venue_similarity: float | None
    volume_match: bool | None
    pages_match: bool | None
    doi_match: bool | None
    arxiv_match: bool | None
    total: float
    decision: str        # HIGH | REVIEW | REJECT
    reasons: list[str]

    def as_dict(self):
        return asdict(self)


def _first_page(pages):
    m = re.match(r'[A-Za-z]?(\d+)', str(pages or ''))
    return m[1] if m else None


def _clean_doi(doi):
    if not doi:
        return None
    d = str(doi).strip().lower()
    d = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', d)
    return d.rstrip('.,;') or None


def _clean_arxiv(aid):
    return re.sub(r'v\d+$', '', str(aid).strip().lower()) if aid else None


def score_candidate(ref: dict, cand: Candidate) -> MatchScore:
    reasons = []
    titles = [ref.get('title'), *ref.get('title_variants', [])]
    titles = [t for t in titles if t]
    tsim = title_similarity(titles, cand.title or '', ref.get('normalized_reference') or ref.get('raw_reference', ''))

    ref_surnames = [norm_surname(s) for s in ref.get('author_surnames', []) if norm_surname(s)]
    cand_surnames = cand.surnames()
    first_match = overlap = None
    if ref_surnames and cand_surnames:
        first_match = ref_surnames[0] == cand_surnames[0] or ref_surnames[0] in cand_surnames[:2]
        considered = ref_surnames[:8]
        overlap = round(sum(s in cand_surnames for s in considered) / len(considered), 3)
    elif cand_surnames and not ref_surnames:
        # Reference without parsed authors: look for candidate surnames anywhere in the raw entry.
        whole = norm_surname(ref.get('normalized_reference', ''))
        hits = [s for s in cand_surnames[:5] if len(s) > 2 and s in whole]
        if hits:
            first_match, overlap = cand_surnames[0] in hits, round(len(hits) / min(5, len(cand_surnames)), 3)

    year = 'unknown'
    if ref.get('year') and cand.year:
        delta = abs(int(ref['year']) - int(cand.year))
        year = 'exact' if delta == 0 else 'adjacent' if delta == 1 else 'mismatch'

    venue_sim = None
    if ref.get('venue') and cand.venue:
        a, b = set(title_tokens(ref['venue'])), set(title_tokens(cand.venue))
        venue_sim = round(2 * len(a & b) / (len(a) + len(b)), 3) if a and b else 0.0

    volume_match = (str(ref['volume']) == str(cand.volume)) if ref.get('volume') and cand.volume else None
    rp, cp = _first_page(ref.get('pages')), _first_page(cand.pages)
    pages_match = (rp == cp) if rp and cp else None

    rdoi, cdoi = _clean_doi(ref.get('doi')), _clean_doi(cand.doi)
    doi_match = (rdoi == cdoi) if rdoi and cdoi else None
    rarx, carx = _clean_arxiv(ref.get('arxiv_id')), _clean_arxiv(cand.arxiv_id)
    arxiv_match = (rarx == carx) if rarx and carx else None

    authors_ok = first_match is True or (overlap or 0) >= .5
    authors_bad = first_match is False and (overlap or 0) == 0
    year_ok = year in ('exact', 'adjacent')
    total = (.45 * tsim + .2 * (overlap if overlap is not None else (.5 if not ref_surnames else 0))
             + .15 * {'exact': 1, 'adjacent': .7, 'unknown': .4, 'mismatch': 0}[year]
             + .05 * (venue_sim or 0) + .05 * bool(volume_match) + .05 * bool(pages_match)
             + .05 * bool(doi_match or arxiv_match))

    if doi_match is False and not (cdoi or '').startswith('10.48550/'):
        reasons.append(f'DOI conflict: reference {rdoi} vs candidate {cdoi}.')
        decision = 'REJECT'
    elif doi_match or arxiv_match:
        ident = 'DOI' if doi_match else 'arXiv identifier'
        if not titles or tsim >= .6:
            decision = 'HIGH'
            reasons.append(f'Exact {ident} match' + (f'; title similarity {tsim:.2f}.' if titles else '; reference title not parsed.'))
        else:
            decision = 'REVIEW'
            reasons.append(f'Exact {ident} match but the title disagrees (similarity {tsim:.2f}); possible identifier typo.')
        if authors_bad:
            decision = 'REVIEW'
            reasons.append('Author lists do not overlap.')
    elif tsim >= .92 and authors_ok and year_ok:
        decision = 'HIGH'
        reasons.append(f'Title similarity {tsim:.2f}, author agreement, year {year}.')
    elif tsim >= .92 and not ref_surnames and year == 'exact' and (venue_sim or 0) >= .5:
        decision = 'HIGH'
        reasons.append(f'Title similarity {tsim:.2f}, year and venue agree; the reference lists no authors.')
    elif tsim >= .92 and authors_bad:
        decision = 'REJECT'
        reasons.append(f'Title similarity {tsim:.2f} but no author overlap: likely a different work with a similar title.')
    elif tsim >= .92 and authors_ok and year == 'mismatch':
        decision = 'REVIEW'
        reasons.append(f'Title and authors agree but publication year differs ({ref.get("year")} vs {cand.year}).')
    elif tsim >= .92 and year_ok:
        decision = 'REVIEW'
        reasons.append(f'Title similarity {tsim:.2f} and year {year}; authors could not be compared.')
    elif tsim >= .8 and authors_ok and year_ok:
        decision = 'REVIEW'
        reasons.append(f'Title partly matches ({tsim:.2f}); authors and year agree.')
    else:
        decision = 'REJECT'
        reasons.append(f'Insufficient agreement (title {tsim:.2f}, first author {first_match}, year {year}).')
    return MatchScore(tsim, first_match, overlap, year, venue_sim, volume_match, pages_match, doi_match, arxiv_match,
                      round(total, 4), decision, reasons)


# ---------------------------------------------------------------------------------------------- providers

class ProviderOutcome:
    OK, NOT_FOUND, ERROR, RATE_LIMITED, DISABLED = 'OK', 'NOT_FOUND', 'ERROR', 'RATE_LIMITED', 'DISABLED'
    SKIPPED = 'SKIPPED'  # no query was possible (e.g. no parsed title); neither a success nor a failure


class _Pacer:
    """Minimum interval between requests to one provider (politeness / documented rate limits)."""
    def __init__(self, interval):
        self.interval, self.last, self.lock = interval, 0.0, asyncio.Lock()

    async def wait(self):
        async with self.lock:
            delay = self.interval - (time.monotonic() - self.last)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last = time.monotonic()


class Providers:
    """Thin async clients for documented scholarly APIs. Each call returns (outcome, candidates, attempt)."""

    def __init__(self, settings=None, transport=None, pace=True):
        self.s = settings or get_settings()
        self.transport = transport
        email = self.s.api_contact_email
        self.headers = {'User-Agent': f'CiteGuard/2.1 (mailto:{email})'}
        self.email_usable = bool(email) and not email.endswith('@example.com')
        intervals = {'crossref': .05, 'openalex': .1, 'semantic_scholar': 1.1, 'arxiv': 3.0, 'unpaywall': .1, 'europepmc': .1}
        self.pacers = {k: _Pacer(v if pace else 0) for k, v in intervals.items()}
        self.circuit: dict[str, str] = {}   # provider -> reason it is switched off for the rest of the job
        self.request_counts: dict[str, int] = {}
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.s.provider_timeout_seconds, follow_redirects=True,
                                         headers=self.headers, transport=self.transport)
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    def enabled(self, name):
        return {'crossref': self.s.crossref_enabled, 'openalex': self.s.openalex_enabled, 'arxiv': self.s.arxiv_enabled,
                'semantic_scholar': self.s.semantic_scholar_enabled,
                'unpaywall': self.s.unpaywall_enabled and self.email_usable,
                'europepmc': self.s.europepmc_enabled}[name]

    async def _get(self, provider, url, params=None, headers=None, query_kind=''):
        """GET with pacing, one bounded Retry-After retry on 429, and an Attempt record."""
        if not self.enabled(provider):
            reason = ('Unpaywall requires a real contact e-mail (API_CONTACT_EMAIL).' if provider == 'unpaywall'
                      else f'{provider} disabled by configuration.')
            return ProviderOutcome.DISABLED, None, Attempt(provider, url, 'DISABLED', reason)
        if provider in self.circuit:
            return ProviderOutcome.RATE_LIMITED, None, Attempt(provider, url, 'RATE_LIMITED', self.circuit[provider])
        for number in (1, 2):
            await self.pacers[provider].wait()
            started = time.monotonic()
            self.request_counts[provider] = self.request_counts.get(provider, 0) + 1
            try:
                response = await self._client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                return ProviderOutcome.ERROR, None, Attempt(provider, url, 'TIMEOUT', f'{query_kind}: {type(exc).__name__}',
                                                             duration_seconds=round(time.monotonic() - started, 3))
            except httpx.HTTPError as exc:
                return ProviderOutcome.ERROR, None, Attempt(provider, url, 'NETWORK_ERROR', f'{query_kind}: {type(exc).__name__}',
                                                             duration_seconds=round(time.monotonic() - started, 3))
            took = round(time.monotonic() - started, 3)
            if response.status_code == 429:
                wait = parse_retry_after(response.headers.get('retry-after'), 10.0)
                if number == 1:
                    await asyncio.sleep(wait if wait is not None else 2.0)
                    continue
                self.circuit[provider] = f'{provider} rate-limited this job (HTTP 429); later lookups were skipped.'
                return ProviderOutcome.RATE_LIMITED, None, Attempt(provider, url, 'RATE_LIMITED',
                                                                    f'{query_kind}: HTTP 429 after retry.', 429,
                                                                    duration_seconds=took, attempt_number=number)
            if response.status_code == 404:
                return ProviderOutcome.NOT_FOUND, response, Attempt(provider, url, 'NOT_FOUND', f'{query_kind}: HTTP 404.',
                                                                     404, duration_seconds=took, attempt_number=number)
            if response.status_code in (401, 403):
                return ProviderOutcome.ERROR, None, Attempt(provider, url, 'ACCESS_DENIED', f'{query_kind}: HTTP {response.status_code}.',
                                                             response.status_code, duration_seconds=took)
            if response.status_code >= 400:
                return ProviderOutcome.ERROR, None, Attempt(provider, url, 'HTTP_ERROR', f'{query_kind}: HTTP {response.status_code}.',
                                                             response.status_code, duration_seconds=took)
            return ProviderOutcome.OK, response, Attempt(provider, url, 'OK', query_kind, 200, duration_seconds=took,
                                                          attempt_number=number)
        raise AssertionError('unreachable')

    @staticmethod
    def _json(response):
        try:
            return response.json()
        except ValueError:
            return None

    # -- Crossref ------------------------------------------------------------------------------------
    @staticmethod
    def _crossref(work) -> Candidate | None:
        if not isinstance(work, dict):
            return None
        dates = (work.get('published') or work.get('issued') or work.get('published-print') or {}).get('date-parts') or []
        abstract = work.get('abstract')
        links = [{'url': link['URL'], 'kind': 'pdf' if link.get('content-type') == 'application/pdf' else 'landing',
                  'provider': 'crossref_link', 'note': link.get('intended-application')}
                 for link in work.get('link') or [] if isinstance(link, dict) and link.get('URL')]
        title = (work.get('title') or [None])[0]
        subtitle = (work.get('subtitle') or [None])[0]
        if title and subtitle and norm_title(subtitle) not in norm_title(title):
            title = f'{title}: {subtitle}'  # Crossref stores subtitles separately ('False-Positive Psychology')
        return Candidate(
            provider='crossref', title=title, year=dates[0][0] if dates and dates[0] else None,
            authors=[' '.join(filter(None, [a.get('given'), a.get('family')])) or a.get('name', '')
                     for a in work.get('author') or [] if isinstance(a, dict)],
            venue=(work.get('container-title') or [None])[0], volume=work.get('volume'), issue=work.get('issue'),
            pages=work.get('page'), doi=work.get('DOI'), work_type=work.get('type'),
            abstract=html.unescape(re.sub(r'<[^>]+>', ' ', abstract)).strip() if abstract else None,
            oa_locations=links, match_score=work.get('score'))

    async def crossref_doi(self, doi):
        out, resp, att = await self._get('crossref', f'https://api.crossref.org/works/{quote(doi, safe="")}',
                                         {'mailto': self.s.api_contact_email}, query_kind='DOI lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        data = self._json(resp) or {}
        cand = self._crossref(data.get('message'))
        return (ProviderOutcome.OK, [cand], att) if cand else (ProviderOutcome.ERROR, [], att)

    async def crossref_search(self, ref):
        query = ' '.join(filter(None, [ref.get('title'), ' '.join(ref.get('author_surnames', [])[:3]),
                                       ref.get('venue'), str(ref.get('year') or '')])) or ref['normalized_reference']
        out, resp, att = await self._get('crossref', 'https://api.crossref.org/works',
                                         {'query.bibliographic': query[:600], 'rows': 5, 'mailto': self.s.api_contact_email},
                                         query_kind='bibliographic search')
        if out != ProviderOutcome.OK:
            return out, [], att
        items = ((self._json(resp) or {}).get('message') or {}).get('items')
        if not isinstance(items, list):
            return ProviderOutcome.ERROR, [], att
        return ProviderOutcome.OK, [c for c in map(self._crossref, items) if c], att

    # -- OpenAlex ------------------------------------------------------------------------------------
    @staticmethod
    def _openalex(work) -> Candidate | None:
        if not isinstance(work, dict):
            return None
        abstract = None
        inv = work.get('abstract_inverted_index')
        if isinstance(inv, dict) and inv:
            size = max((max(p) for p in inv.values() if p), default=-1) + 1
            if 0 < size <= 20000:
                words = [''] * size
                for word, positions in inv.items():
                    for pos in positions:
                        words[pos] = word
                abstract = ' '.join(words).strip() or None
        ids = work.get('ids') or {}
        biblio = work.get('biblio') or {}
        primary = work.get('primary_location') or {}
        locations, arxiv_id = [], None
        for loc in [work.get('best_oa_location'), primary, *(work.get('locations') or [])]:
            if not isinstance(loc, dict):
                continue
            landing = loc.get('landing_page_url') or ''
            m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', landing + ' ' + (loc.get('pdf_url') or ''))
            if m:
                arxiv_id = arxiv_id or m[1]
            if loc.get('is_oa') and loc.get('pdf_url'):
                locations.append({'url': loc['pdf_url'], 'kind': 'pdf', 'provider': 'openalex',
                                  'version': loc.get('version'), 'license': loc.get('license')})
            if loc.get('is_oa') and landing:
                locations.append({'url': landing, 'kind': 'landing', 'provider': 'openalex',
                                  'version': loc.get('version'), 'license': loc.get('license')})
        pages = '–'.join(filter(None, [biblio.get('first_page'), biblio.get('last_page')])) or None
        pmcid = ids.get('pmcid')
        return Candidate(
            provider='openalex', title=work.get('title') or work.get('display_name'), year=work.get('publication_year'),
            authors=[(a.get('author') or {}).get('display_name', '') for a in work.get('authorships') or [] if isinstance(a, dict)],
            venue=(primary.get('source') or {}).get('display_name'), volume=biblio.get('volume'), issue=biblio.get('issue'),
            pages=pages, doi=_clean_doi(work.get('doi')), arxiv_id=arxiv_id, work_type=work.get('type'),
            pmcid=pmcid.rsplit('/', 1)[-1] if pmcid else None, abstract=abstract, provider_id=work.get('id'),
            oa_locations=locations, match_score=work.get('relevance_score'))

    def _openalex_params(self, extra):
        params = {'mailto': self.s.api_contact_email, **extra}
        key = self.s.openalex_api_key.get_secret_value()
        if key:
            params['api_key'] = key
        return params

    async def openalex_doi(self, doi):
        out, resp, att = await self._get('openalex', f'https://api.openalex.org/works/https://doi.org/{doi}',
                                         self._openalex_params({}), query_kind='DOI lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        cand = self._openalex(self._json(resp))
        return (ProviderOutcome.OK, [cand], att) if cand else (ProviderOutcome.ERROR, [], att)

    async def openalex_search(self, ref):
        title = ref.get('title') or ref['normalized_reference'][:300]
        out, resp, att = await self._get('openalex', 'https://api.openalex.org/works',
                                         self._openalex_params({'search': norm_title(title)[:300], 'per-page': 5}),
                                         query_kind='title search')
        if out != ProviderOutcome.OK:
            return out, [], att
        results = (self._json(resp) or {}).get('results')
        if not isinstance(results, list):
            return ProviderOutcome.ERROR, [], att
        return ProviderOutcome.OK, [c for c in map(self._openalex, results) if c], att

    # -- Semantic Scholar ------------------------------------------------------------------------------
    S2_FIELDS = 'title,year,authors,venue,journal,externalIds,openAccessPdf,abstract,publicationTypes'

    @staticmethod
    def _s2(paper) -> Candidate | None:
        if not isinstance(paper, dict) or not paper.get('title'):
            return None
        ext = paper.get('externalIds') or {}
        journal = paper.get('journal') or {}
        oa = paper.get('openAccessPdf') or {}
        locations = [{'url': oa['url'], 'kind': 'pdf', 'provider': 'semantic_scholar', 'note': oa.get('status')}] if oa.get('url') else []
        return Candidate(provider='semantic_scholar', title=paper.get('title'), year=paper.get('year'),
                         authors=[a.get('name', '') for a in paper.get('authors') or [] if isinstance(a, dict)],
                         venue=journal.get('name') or paper.get('venue'), volume=journal.get('volume'),
                         pages=(journal.get('pages') or '').strip() or None, doi=_clean_doi(ext.get('DOI')),
                         arxiv_id=ext.get('ArXiv'), pmcid=('PMC' + str(ext['PubMedCentral'])) if ext.get('PubMedCentral') else None,
                         abstract=paper.get('abstract'), provider_id=paper.get('paperId'), oa_locations=locations,
                         match_score=paper.get('matchScore'))

    def _s2_headers(self):
        key = self.s.semantic_scholar_api_key.get_secret_value()
        return {'x-api-key': key} if key else None

    async def s2_lookup(self, ident):
        out, resp, att = await self._get('semantic_scholar', f'https://api.semanticscholar.org/graph/v1/paper/{ident}',
                                         {'fields': self.S2_FIELDS}, self._s2_headers(), query_kind=f'{ident.split(":")[0]} lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        cand = self._s2(self._json(resp))
        return (ProviderOutcome.OK, [cand], att) if cand else (ProviderOutcome.ERROR, [], att)

    async def s2_match(self, ref):
        title = ref.get('title')
        if not title:
            return ProviderOutcome.SKIPPED, [], Attempt('semantic_scholar', '', 'SKIPPED', 'No parsed title to match.')
        out, resp, att = await self._get('semantic_scholar', 'https://api.semanticscholar.org/graph/v1/paper/search/match',
                                         {'query': title[:300], 'fields': self.S2_FIELDS}, self._s2_headers(),
                                         query_kind='title match')
        if out != ProviderOutcome.OK:
            return out, [], att
        data = (self._json(resp) or {}).get('data')
        if not isinstance(data, list):
            return ProviderOutcome.ERROR, [], att
        return ProviderOutcome.OK, [c for c in map(self._s2, data) if c], att

    # -- arXiv ---------------------------------------------------------------------------------------
    @staticmethod
    def _arxiv_entries(text) -> list[Candidate] | None:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
        out = []
        for entry in root.findall('a:entry', ns):
            aid = (entry.findtext('a:id', '', ns) or '').split('/abs/')[-1]
            title = ' '.join((entry.findtext('a:title', '', ns) or '').split())
            if not aid or not title or title.lower() == 'error':
                continue
            published = entry.findtext('a:published', '', ns) or ''
            base = re.sub(r'v\d+$', '', aid)
            out.append(Candidate(provider='arxiv', title=title, year=int(published[:4]) if published[:4].isdigit() else None,
                                 authors=[n.text or '' for n in entry.findall('a:author/a:name', ns)], venue='arXiv',
                                 doi=_clean_doi(entry.findtext('x:doi', None, ns)), arxiv_id=base,
                                 abstract=' '.join((entry.findtext('a:summary', '', ns) or '').split()) or None,
                                 oa_locations=[{'url': f'https://arxiv.org/pdf/{base}', 'kind': 'pdf', 'provider': 'arxiv'}]))
        return out

    async def arxiv_id(self, aid):
        out, resp, att = await self._get('arxiv', 'https://export.arxiv.org/api/query',
                                         {'id_list': re.sub(r'v\d+$', '', aid), 'max_results': 1}, query_kind='arXiv id lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        entries = self._arxiv_entries(resp.text)
        return (ProviderOutcome.OK, entries, att) if entries is not None else (ProviderOutcome.ERROR, [], att)

    async def arxiv_search(self, ref):
        words = title_tokens(ref.get('title') or '')[:10]
        if len(words) < 3:
            return ProviderOutcome.SKIPPED, [], Attempt('arxiv', '', 'SKIPPED', 'Title too short for an arXiv title search.')
        query = ' AND '.join(f'ti:{w}' for w in words)
        out, resp, att = await self._get('arxiv', 'https://export.arxiv.org/api/query',
                                         {'search_query': query, 'max_results': 5}, query_kind='arXiv title search')
        if out != ProviderOutcome.OK:
            return out, [], att
        entries = self._arxiv_entries(resp.text)
        return (ProviderOutcome.OK, entries, att) if entries is not None else (ProviderOutcome.ERROR, [], att)

    # -- open-access location services (full-text discovery, not identification) --------------------------
    async def unpaywall(self, doi):
        out, resp, att = await self._get('unpaywall', f'https://api.unpaywall.org/v2/{quote(doi, safe="/")}',
                                         {'email': self.s.api_contact_email}, query_kind='open-access lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        data = self._json(resp) or {}
        locs = []
        for loc in [data.get('best_oa_location'), *(data.get('oa_locations') or [])]:
            if not isinstance(loc, dict):
                continue
            meta = {'provider': 'unpaywall', 'host_type': loc.get('host_type'), 'version': loc.get('version'),
                    'license': loc.get('license')}
            if loc.get('url_for_pdf'):
                locs.append({'url': loc['url_for_pdf'], 'kind': 'pdf', **meta})
            if loc.get('url_for_landing_page') or loc.get('url'):
                locs.append({'url': loc.get('url_for_landing_page') or loc['url'], 'kind': 'landing', **meta})
        return ProviderOutcome.OK, locs, att

    async def europepmc(self, doi):
        out, resp, att = await self._get('europepmc', 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                                         {'query': f'DOI:"{doi}"', 'format': 'json', 'resultType': 'lite'},
                                         query_kind='open-access lookup')
        if out != ProviderOutcome.OK:
            return out, [], att
        results = (((self._json(resp) or {}).get('resultList') or {}).get('result')) or []
        locs = []
        for r in results:
            if isinstance(r, dict) and r.get('pmcid') and r.get('isOpenAccess') == 'Y':
                locs.append({'url': f'https://www.ebi.ac.uk/europepmc/webservices/rest/{r["pmcid"]}/fullTextXML',
                             'kind': 'jats', 'provider': 'europepmc', 'pmcid': r['pmcid']})
        return ProviderOutcome.OK, locs, att


# ---------------------------------------------------------------------------------------------- identification

@dataclass
class Identification:
    status: str
    reason: str
    reference: dict
    best: dict | None = None                 # chosen candidate (asdict) incl. merged OA locations
    score: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    providers_queried: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)   # other accepted works when AMBIGUOUS
    cached: bool = False
    duration_seconds: float = 0.0

    def as_dict(self):
        return asdict(self)


def _same_work(a: Candidate, b: Candidate) -> bool:
    da, db = _clean_doi(a.doi), _clean_doi(b.doi)
    if da and db:
        if da == db:
            return True
        # Two different registered DOIs are two works (unless one is an arXiv DataCite DOI for a preprint version).
        if not (da.startswith('10.48550/') or db.startswith('10.48550/')):
            return False
    if a.arxiv_id and b.arxiv_id and _clean_arxiv(a.arxiv_id) == _clean_arxiv(b.arxiv_id):
        return True
    ta, tb = norm_title(a.title), norm_title(b.title)
    if ta and ta == tb or (ta and tb and SequenceMatcher(None, ta, tb).ratio() >= .97):
        sa, sb = a.surnames(), b.surnames()
        return not sa or not sb or sa[0] == sb[0]
    return False


def _merge(primary: Candidate, other: Candidate) -> Candidate:
    for name in ('title', 'year', 'venue', 'volume', 'issue', 'pages', 'doi', 'arxiv_id', 'pmcid', 'abstract'):
        if not getattr(primary, name) and getattr(other, name):
            setattr(primary, name, getattr(other, name))
    if not primary.authors and other.authors:
        primary.authors = other.authors
    seen = {loc['url'] for loc in primary.oa_locations}
    primary.oa_locations += [loc for loc in other.oa_locations if loc['url'] not in seen]
    return primary


class ReferenceIdentifier:
    """Resolve bibliography entries to works. One instance per job; results are cached per reference."""

    def __init__(self, settings=None, transport=None, pace=True, cache_dir=None):
        self.settings = settings or get_settings()
        self.transport = transport
        self.pace = pace
        # None -> default persistent cache; False -> no cache at all (cold measurements)
        self.cache_dir = CACHE_DIR if cache_dir is None else (cache_dir or None)
        self._memo: dict[str, asyncio.Task] = {}
        self.providers: Providers | None = None

    async def __aenter__(self):
        self.providers = await Providers(self.settings, self.transport, self.pace).__aenter__()
        return self

    async def __aexit__(self, *exc):
        if self.providers:
            await self.providers.__aexit__(*exc)

    # -- cache -------------------------------------------------------------------------------------
    def _key(self, text):
        return hashlib.sha256((CACHE_VERSION + '|' + ' '.join(text.lower().split())).encode()).hexdigest()

    def _load(self, key):
        days = self.settings.identification_cache_days
        if not days or self.cache_dir is None:
            return None
        path = self.cache_dir / f'{key}.json'
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if time.time() - data.get('stored_at', 0) > days * 86400 or data.get('status') not in CONCLUSIVE:
            return None
        data.pop('stored_at', None)
        data['cached'] = True
        return Identification(**data)

    def _store(self, key, result: Identification):
        if not self.settings.identification_cache_days or self.cache_dir is None or result.status not in CONCLUSIVE:
            return  # provider errors and rate limits are never cached: a retry may succeed
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f'{key}.json').write_text(json.dumps({**result.as_dict(), 'stored_at': time.time()}),
                                                        encoding='utf-8')
        except OSError:
            pass

    # -- public ------------------------------------------------------------------------------------
    async def identify(self, raw: str) -> Identification:
        ref = reference_record(raw, 1)
        key = self._key(ref['normalized_reference'])
        if key not in self._memo:
            self._memo[key] = asyncio.ensure_future(self._identify_cached(key, ref))
        return await self._memo[key]

    async def _identify_cached(self, key, ref):
        cached = self._load(key)
        if cached:
            cached.attempts = [Attempt('identification_cache', '', 'CACHED',
                                       f'Identification served from the local cache ({cached.status}).', cached=True).as_dict()]
            return cached
        started = time.monotonic()
        result = await self._identify(ref)
        result.duration_seconds = round(time.monotonic() - started, 3)
        self._store(key, result)
        return result

    async def _identify(self, ref) -> Identification:
        p = self.providers
        attempts: list[dict] = []
        outcomes: dict[str, list[str]] = {}
        pool: list[tuple[Candidate, MatchScore]] = []

        def take(provider, outcome, cands, attempt):
            outcomes.setdefault(provider, []).append(outcome)
            a = attempt.as_dict()
            a['candidates'] = len(cands)
            for cand in cands:
                pool.append((cand, score_candidate(ref, cand)))
            if cands:
                best = max((score_candidate(ref, c) for c in cands), key=lambda s: s.total)
                a['best_candidate_decision'] = best.decision
                a['best_title_similarity'] = best.title_similarity
            attempts.append(a)

        def best_decision():
            accepted = [s.decision for _, s in pool if s.decision != 'REJECT']
            return 'HIGH' if 'HIGH' in accepted else 'REVIEW' if accepted else None

        stages = []
        if ref.get('arxiv_id'):
            stages.append([('arxiv', p.arxiv_id(ref['arxiv_id'])), ('semantic_scholar', p.s2_lookup('ARXIV:' + _clean_arxiv(ref['arxiv_id'])))])
        if ref.get('doi'):
            stages.append([('crossref', p.crossref_doi(ref['doi'])), ('openalex', p.openalex_doi(ref['doi']))])
        stages.append([('crossref', p.crossref_search(ref)), ('openalex', p.openalex_search(ref))])
        stages.append([('semantic_scholar', p.s2_match(ref))])
        stages.append([('arxiv', p.arxiv_search(ref))])
        for index, stage in enumerate(stages):
            if best_decision() == 'HIGH' and index > 0:
                for _, coro in stage:
                    coro.close()  # never awaited: identification already conclusive
                continue
            results = await asyncio.gather(*(coro for _, coro in stage))
            for (provider, _), (outcome, cands, attempt) in zip(stage, results):
                take(provider, outcome, cands, attempt)

        # ---- group accepted candidates into distinct works
        accepted = sorted([(c, s) for c, s in pool if s.decision != 'REJECT'], key=lambda cs: (cs[1].decision != 'HIGH', -cs[1].total))
        works: list[tuple[Candidate, MatchScore, list[str]]] = []
        for cand, sc in accepted:
            for i, (wc, ws, provs) in enumerate(works):
                if _same_work(wc, cand):
                    works[i] = (_merge(wc, cand), ws, provs + [cand.provider])
                    break
            else:
                works.append((Candidate(**asdict(cand)), sc, [cand.provider]))
        # OA locations from rejected-but-same-work candidates are not merged: only accepted metadata is used.
        cand_dump = [{'provider': c.provider, 'title': c.title, 'doi': c.doi, 'arxiv_id': c.arxiv_id, 'year': c.year,
                      'authors': c.authors[:6], 'venue': c.venue, 'score': s.as_dict()} for c, s in pool]
        queried = sorted(outcomes)
        failed = {k: v for k, v in outcomes.items() if any(o in (ProviderOutcome.ERROR, ProviderOutcome.RATE_LIMITED) for o in v)}
        succeeded = [k for k, v in outcomes.items() if any(o in (ProviderOutcome.OK, ProviderOutcome.NOT_FOUND) for o in v)]

        if works:
            top, top_score, provs = works[0]
            rivals = [w for w in works[1:] if w[1].decision == top_score.decision and top_score.total - w[1].total < .05]
            if rivals and not (top_score.doi_match or top_score.arxiv_match):
                return Identification(AMBIGUOUS, f'{len(rivals) + 1} different works match this reference about equally well '
                                      f'(best score {top_score.total:.2f}); none was selected.', ref, None, top_score.as_dict(),
                                      cand_dump, attempts, queried,
                                      alternatives=[{'title': w[0].title, 'doi': w[0].doi, 'year': w[0].year,
                                                     'score': w[1].total} for w in works[:4]])
            status = IDENTIFIED_HIGH if top_score.decision == 'HIGH' else IDENTIFIED_REVIEW
            agreeing = sorted(set(provs))
            reason = ' '.join(top_score.reasons) + f' Confirmed by: {", ".join(agreeing)}.'
            return Identification(status, reason, ref, asdict(top), top_score.as_dict(), cand_dump, attempts, queried)
        if outcomes and all(o in (ProviderOutcome.DISABLED, ProviderOutcome.SKIPPED) for v in outcomes.values() for o in v):
            return Identification(PROVIDER_ERROR, 'All metadata providers are disabled by configuration; the reference '
                                  'was not searched.', ref, None, None, cand_dump, attempts, queried)
        if failed and len(succeeded) < 2:
            limited = any(ProviderOutcome.RATE_LIMITED in v for v in failed.values())
            status = RATE_LIMITED if limited else PROVIDER_ERROR
            return Identification(status, f'Provider failures ({", ".join(sorted(failed))}) prevented a reliable search; '
                                  f'successful providers: {", ".join(succeeded) or "none"}. The source may exist.',
                                  ref, None, None, cand_dump, attempts, queried)
        near = max((s for _, s in pool), key=lambda s: s.total, default=None)
        detail = f' Closest rejected candidate: {near.reasons[0]}' if near else ' No candidates were returned.'
        return Identification(NOT_FOUND, f'No provider returned a work whose title, authors and year agree with this reference '
                              f'(searched: {", ".join(succeeded)}).' + detail, ref, None, None, cand_dump, attempts, queried)
