"""Full-text discovery, retrieval, extraction and identity validation for an IDENTIFIED reference.

Order of locations (legally accessible only; access controls are never bypassed):
  1. exact arXiv PDF                       (arXiv id from the reference or from provider metadata)
  2. Europe PMC open-access JATS full text (PMCID, open-access subset only)
  3. Unpaywall / OpenAlex / Semantic Scholar open-access PDF and landing locations
  4. publisher links from Crossref and the DOI landing page (citation_pdf_url, or open HTML full text)

Every location produces attempt records. A document is accepted as full text only if
  - it was extracted to readable text (PDF text layer, OCR fallback, HTML article body or JATS XML), and
  - it passes identity validation against the identified metadata (identifier, title, authors), and
  - it is substantive (an abstract-only landing page is not full text).
Rejected documents are never used as evidence. Search snippets and generated text are never used.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from app.services.document_parser import DocumentParser
from app.services.downloader import fetch_pdf
from app.services.fetch_policy import Attempt, provider_for
from app.services.identification import fold, norm_surname, norm_title, title_tokens
from backend.config import ROOT

PARSED_CACHE = ROOT / 'data' / 'cache' / 'parsed'
PARSER_VERSION = 'v2.1-2'  # bump whenever extraction output changes (cache key)
MIN_FULLTEXT_WORDS = 1200
NON_EVIDENCE_SECTIONS = re.compile(r'^(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|works cited|literature cited|'
                                   r'acknowledg(?:e)?ments?|funding|author contributions|competing interests|'
                                   r'data availability|code availability|additional information|supplementary information)\b', re.I)


# ---------------------------------------------------------------------------------------------- extraction

class _ArticleHTML(HTMLParser):
    """Collect headings and paragraph text from an HTML article, skipping navigation and reference lists."""
    SKIP = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'noscript', 'svg', 'button', 'figure', 'table'}
    SKIP_HINT = re.compile(r'(?:^|[-_ ])(?:nav|menu|footer|header|cookie|banner|sidebar|related|recommend|share|'
                           r'references?|bibliography|citation-list|ref-list|comments?|ads?)(?:$|[-_ ])', re.I)
    BLOCK = {'p', 'h1', 'h2', 'h3', 'h4', 'li', 'blockquote'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta, self.blocks, self.stack, self.skip_depth, self.buf, self.tag = {}, [], [], 0, [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta':
            name = (a.get('name') or a.get('property') or '').lower()
            if name and a.get('content'):
                self.meta.setdefault(name, []).append(a['content'])
            return
        if tag in ('br', 'img', 'hr', 'input', 'link'):
            return
        hint = ' '.join(filter(None, [a.get('class'), a.get('id'), a.get('role')]))
        skipping = tag in self.SKIP or bool(hint and self.SKIP_HINT.search(hint))
        self.stack.append((tag, skipping))
        if skipping:
            self.skip_depth += 1
        if tag in self.BLOCK and not self.skip_depth:
            self._flush()
            self.tag = tag

    def handle_endtag(self, tag):
        while self.stack:
            t, skipping = self.stack.pop()
            if skipping:
                self.skip_depth -= 1
            if t == tag:
                break
        if tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self.skip_depth and self.tag:
            self.buf.append(data)

    def _flush(self):
        text = ' '.join(''.join(self.buf).split())
        if text and self.tag:
            self.blocks.append((self.tag, text))
        self.buf, self.tag = [], None


def _paragraphs_from_blocks(blocks, name):
    paragraphs, section = [], 'Body'
    for tag, text in blocks:
        if tag in ('h1', 'h2', 'h3', 'h4'):
            section = text[:120]
            continue
        if NON_EVIDENCE_SECTIONS.match(section) or len(text.split()) < 8:
            continue
        paragraphs.append({'text': text, 'page_number': None, 'paragraph': len(paragraphs) + 1, 'section': section})
    return paragraphs


def extract_html(content: bytes, name: str) -> dict:
    parser = _ArticleHTML()
    parser.feed(content[:8_000_000].decode('utf-8', errors='replace'))
    parser._flush()
    meta = parser.meta
    title = (meta.get('citation_title') or meta.get('dc.title') or meta.get('og:title') or [None])[0]
    paragraphs = _paragraphs_from_blocks(parser.blocks, name)
    words = sum(len(p['text'].split()) for p in paragraphs)
    front = ' '.join([title or '', *meta.get('citation_author', []), *meta.get('citation_doi', []),
                      *[t for tag, t in parser.blocks[:40]]])
    return {'document_name': name, 'format': 'html', 'paragraphs': paragraphs, 'words': words, 'front_text': front,
            'declared': {'title': title, 'doi': (meta.get('citation_doi') or [None])[0],
                         'authors': meta.get('citation_author', [])},
            'abstract_only': words < MIN_FULLTEXT_WORDS, 'total_pages': None,
            'extraction': {'method': 'html_article', 'paragraphs': len(paragraphs), 'words': words}}


def extract_jats(content: bytes, name: str) -> dict:
    root = ET.fromstring(content)
    title = ' '.join(''.join(root.find('.//article-title').itertext()).split()) if root.find('.//article-title') is not None else None
    doi = next((e.text for e in root.iter('article-id') if e.get('pub-id-type') == 'doi'), None)
    surnames = [e.text or '' for e in root.iter('surname')][:30]
    blocks = []
    body = root.find('.//body')
    abstract = root.find('.//abstract')
    for part, label in ((abstract, 'Abstract'), (body, None)):
        if part is None:
            continue
        if label:
            blocks.append(('h2', label))
        for el in part.iter():
            if el.tag == 'title':
                blocks.append(('h2', ' '.join(''.join(el.itertext()).split())))
            elif el.tag == 'p':
                blocks.append(('p', ' '.join(''.join(el.itertext()).split())))
    paragraphs = _paragraphs_from_blocks(blocks, name)
    words = sum(len(p['text'].split()) for p in paragraphs)
    return {'document_name': name, 'format': 'jats', 'paragraphs': paragraphs, 'words': words,
            'front_text': ' '.join(filter(None, [title, doi, *surnames])),
            'declared': {'title': title, 'doi': doi, 'authors': surnames},
            'abstract_only': body is None or words < MIN_FULLTEXT_WORDS, 'total_pages': None,
            'extraction': {'method': 'jats_xml', 'paragraphs': len(paragraphs), 'words': words}}


def _parsed_cache_path(digest):
    return PARSED_CACHE / f'{digest}.{PARSER_VERSION}.json'


def extract_pdf(path: str, digest: str | None = None, parser: DocumentParser | None = None) -> dict:
    """Parse a PDF (text layer, OCR fallback) with a persistent per-document cache keyed by SHA-256."""
    digest = digest or hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cache = _parsed_cache_path(digest)
    if cache.exists():
        try:
            doc = json.loads(cache.read_text(encoding='utf-8'))
            doc['extraction'] = {**doc.get('extraction', {}), 'cached': True}
            return doc
        except (OSError, ValueError):
            pass
    parsed = (parser or DocumentParser()).parse(path)
    # First page only: in a short document page 2 may already hold the bibliography, and a matching entry in a
    # document's own reference list is not evidence of identity.
    front = '\n'.join(p['text'] for p in parsed['pages'][:1])
    paragraphs = [p for p in parsed['paragraphs'] if not NON_EVIDENCE_SECTIONS.match(p.get('section') or '')]
    words = sum(len(p['text'].split()) for p in paragraphs)
    doc = {'document_name': parsed['document_name'], 'format': 'pdf', 'paragraphs': paragraphs, 'words': words,
           'front_text': front, 'declared': {'title': parsed['metadata'].get('title'), 'doi': None,
                                             'authors': parsed['metadata'].get('authors', [])},
           'abstract_only': False, 'total_pages': parsed['total_pages'],
           'extraction': {**parsed.get('extraction', {'method': 'pdf_text'}), 'paragraphs': len(paragraphs), 'words': words}}
    try:
        PARSED_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(doc), encoding='utf-8')
    except OSError:
        pass
    return doc


# ---------------------------------------------------------------------------------------------- identity

def validate_identity(meta: dict, document: dict) -> dict:
    """Is this document the identified work? Uses front matter only - a matching entry in the document's own
    bibliography is not identity. Returns {'valid', 'method', 'checks', 'reason'}."""
    front = fold(document.get('front_text') or '')
    front_lower = front.lower()
    compact = re.sub(r'[^a-z0-9]', '', front_lower)
    declared = document.get('declared') or {}
    checks = {}
    doi = (meta.get('doi') or '').lower()
    if doi:
        checks['doi_in_front_matter'] = doi in front_lower or (declared.get('doi') or '').lower() == doi
    arxiv = re.sub(r'v\d+$', '', (meta.get('arxiv_id') or '').lower())
    if arxiv:
        checks['arxiv_id_in_front_matter'] = arxiv in front_lower
    title = meta.get('title') or ''
    toks = title_tokens(title)
    if toks:
        words = set(norm_title(front).split())
        coverage = sum(t in words for t in toks) / len(toks)
        checks['title_token_coverage'] = round(coverage, 3)
        checks['title_exact_compact'] = len(re.sub(r'[^a-z0-9]', '', norm_title(title))) >= 15 and \
            re.sub(r'[^a-z0-9]', '', norm_title(title)) in compact
    surnames = [norm_surname(a.split()[-1]) for a in meta.get('authors', [])[:6] if a.split()]
    if surnames:
        found = [s for s in surnames if len(s) > 2 and s in compact]
        checks['authors_found'] = f'{len(found)}/{len(surnames)}'
        author_ok = bool(found)
    else:
        author_ok = None
    if checks.get('doi_in_front_matter') or checks.get('arxiv_id_in_front_matter'):
        return {'valid': True, 'method': 'exact_identifier', 'checks': checks,
                'reason': 'The identified DOI/arXiv identifier appears in the document front matter.'}
    title_ok = checks.get('title_exact_compact') or checks.get('title_token_coverage', 0) >= .9
    if title_ok and author_ok is not False:
        return {'valid': True, 'method': 'title_and_authors' if author_ok else 'title_only', 'checks': checks,
                'reason': 'The identified title' + (' and author surnames appear' if author_ok else ' appears')
                          + ' in the document front matter.'}
    return {'valid': False, 'method': 'none', 'checks': checks,
            'reason': 'Downloaded content does not match the identified work (title/authors/identifier not found in front matter).'}


# ---------------------------------------------------------------------------------------------- discovery

def discover_locations(best: dict, ref: dict, extra: list[dict], limit: int) -> list[dict]:
    """Ordered, de-duplicated candidate locations for one identified work."""
    locs = []
    arxiv = best.get('arxiv_id') or ref.get('arxiv_id')
    if arxiv:
        base = re.sub(r'v\d+$', '', arxiv)
        locs.append({'url': f'https://arxiv.org/pdf/{base}', 'kind': 'pdf', 'provider': 'arxiv'})
    jats = [x for x in extra if x.get('kind') == 'jats']
    oa = [x for x in extra + best.get('oa_locations', []) if x.get('kind') in ('pdf', 'landing') and x.get('provider') != 'crossref_link']
    oa.sort(key=lambda x: (x['kind'] != 'pdf', x.get('provider') != 'unpaywall'))
    links = [x for x in best.get('oa_locations', []) if x.get('provider') == 'crossref_link'
             and x.get('note') != 'similarity-checking']
    locs += jats + oa + links
    if best.get('doi'):
        locs.append({'url': 'https://doi.org/' + best['doi'], 'kind': 'landing', 'provider': 'doi_resolver'})
    for url in ref.get('urls', []):
        locs.append({'url': url, 'kind': 'landing', 'provider': 'reference_url'})
    seen, out = set(), []
    for loc in locs:
        key = re.sub(r'^https?://(www\.)?', '', loc['url']).rstrip('/').lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(loc)
    return out[:limit]


async def fetch_jats(providers, loc):
    outcome, response, attempt = await providers._get('europepmc', loc['url'], query_kind='JATS full text')
    if outcome != 'OK':
        return None, [attempt.as_dict()]
    try:
        return extract_jats(response.content, loc.get('pmcid') or 'Europe PMC'), [attempt.as_dict()]
    except ET.ParseError:
        return None, [attempt.as_dict(), Attempt('europepmc', loc['url'], 'PARSE_FAILED', 'JATS XML could not be parsed.').as_dict()]


async def retrieve_full_text(best: dict, ref: dict, locations: list[dict], settings, providers=None,
                             blocked_digests: frozenset = frozenset()) -> dict:
    """Try locations in order. Returns {'status', 'document'?, 'url', 'provider', 'format', 'sha256', 'identity',
    'extraction', 'failure_reason', 'attempts', 'abstract_only_pages'}."""
    attempts, outcome_summary = [], []
    extraction_failures, identity_failures = [], []
    for loc in locations:
        url, provider = loc['url'], loc.get('provider') or provider_for(loc['url'])
        if loc.get('kind') == 'jats':
            if providers is None:
                continue
            document, att = await fetch_jats(providers, loc)
            attempts += att
            digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest() if document else None
        else:
            fetched = await fetch_pdf(url, settings)
            for a in fetched.get('attempts', []):
                a.setdefault('location_provider', provider)
            attempts += fetched.get('attempts', [])
            if not fetched.get('path'):
                outcome_summary.append(fetched.get('reason') or 'not retrievable')
                continue
            digest = fetched['sha256']
            if digest in blocked_digests:
                attempts.append(Attempt(provider, fetched['url'], 'SELF_REFERENCE',
                                        'This document is the submitted manuscript; it cannot verify itself.').as_dict())
                continue
            try:
                if fetched.get('format') == 'html':
                    document = await asyncio.to_thread(extract_html, Path(fetched['path']).read_bytes(),
                                                       urlsplit(fetched['url']).hostname or 'web page')
                else:
                    document = await asyncio.wait_for(asyncio.to_thread(extract_pdf, fetched['path'], digest), 120)
            except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                message = str(exc)
                cause = message.split(':', 1)[0] if re.match(r'^[A-Z_]+:', message) else 'PARSE_FAILED'
                attempts.append(Attempt(provider, fetched['url'], 'EXTRACTION_FAILED', f'{cause}: {message[:300]}').as_dict())
                extraction_failures.append({'url': fetched['url'], 'cause': cause, 'detail': message[:300]})
                continue
            url = fetched['url']
        if document is None:
            outcome_summary.append('JATS not available')
            continue
        identity = validate_identity(best, document)
        if not identity['valid']:
            attempts.append(Attempt(provider, url, 'IDENTITY_MISMATCH',
                                    identity['reason'] + ' The document was NOT used as evidence.').as_dict())
            identity_failures.append({'url': url, **identity})
            continue
        if document.get('abstract_only') or document['words'] < 200:
            attempts.append(Attempt(provider, url, 'ABSTRACT_ONLY_PAGE',
                                    f'Validated page holds only {document["words"]} words of article text; not full text.').as_dict())
            outcome_summary.append('abstract-only page')
            continue
        attempts.append(Attempt(provider, url, 'FULL_TEXT_ACCEPTED',
                                f'{document["format"].upper()} with {document["words"]} words; identity: {identity["method"]}.').as_dict())
        version = loc.get('version') or ('preprint (arXiv)' if provider == 'arxiv' else None)
        return {'status': 'FULL_TEXT_RETRIEVED', 'document': document, 'url': url, 'provider': provider,
                'version': version, 'format': document['format'], 'sha256': digest, 'identity': identity,
                'extraction': {'status': 'EXTRACTED', **document.get('extraction', {})},
                'failure_reason': None, 'attempts': attempts}
    outcomes = {a.get('outcome') for a in attempts}
    if extraction_failures and not identity_failures:
        status, reason = 'EXTRACTION_FAILED', ('A source document was downloaded but no readable text could be extracted ('
                                               + extraction_failures[0]['cause'] + ').')
    elif identity_failures:
        status, reason = 'IDENTITY_MISMATCH', 'Downloaded documents did not match the identified work and were discarded.'
    elif not locations:
        status, reason = 'NO_OPEN_ACCESS_LOCATION', 'No open-access full-text location is known for this work.'
    elif outcomes & {'TIMEOUT', 'NETWORK_ERROR', 'RATE_LIMITED', 'TRANSIENT_ERROR'} and not outcomes & {'ACCESS_DENIED', 'NOT_A_PDF', 'ABSTRACT_ONLY_PAGE', 'NOT_FOUND'}:
        status, reason = 'RETRIEVAL_FAILED', 'Full-text locations could not be reached (timeouts or network errors); retry may succeed.'
    elif 'ACCESS_DENIED' in outcomes:
        status, reason = 'ACCESS_RESTRICTED', 'Known full-text locations require a subscription or login; access controls were respected.'
    else:
        status, reason = 'NO_OPEN_ACCESS_LOCATION', 'Locations tried did not provide open full text (' + '; '.join(sorted(set(outcome_summary))[:4]) + ').'
    return {'status': status, 'document': None, 'url': None, 'provider': None, 'format': None, 'sha256': None,
            'identity': identity_failures[0] if identity_failures else None,
            'extraction': {'status': 'FAILED', **extraction_failures[0]} if extraction_failures else {'status': 'NOT_ATTEMPTED'},
            'failure_reason': reason, 'attempts': attempts}
