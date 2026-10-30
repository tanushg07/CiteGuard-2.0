"""Resolve each bibliography entry once: identification -> full-text retrieval -> extraction -> passages.

Each stage has its own explicit state so failures are attributed to the stage that failed:

  identification.status : IDENTIFIED_HIGH_CONFIDENCE | IDENTIFIED_NEEDS_REVIEW | AMBIGUOUS_SOURCE |
                          SOURCE_NOT_FOUND | PROVIDER_ERROR | RATE_LIMITED | NO_BIBLIOGRAPHY_ENTRY
  full_text.status      : FULL_TEXT_RETRIEVED | ACCESS_RESTRICTED | NO_OPEN_ACCESS_LOCATION | RETRIEVAL_FAILED |
                          IDENTITY_MISMATCH | EXTRACTION_FAILED | NOT_ATTEMPTED
  status (pipeline)     : FULL_TEXT | ABSTRACT_ONLY | SOURCE_METADATA_ONLY | SOURCE_UNAVAILABLE |
                          RETRIEVAL_FAILED | EXTRACTION_FAILED     (consumed by app.core.verdicts.decide)

The per-reference deadline starts when the reference acquires a concurrency slot, never while it is queued.
"""
import asyncio
import hashlib
import time

from app.core.schemas import ReferenceMetadata
from app.services.document_parser import reference_record
from app.services.fetch_policy import Attempt
from app.services.fulltext import discover_locations, retrieve_full_text, validate_identity
from app.services.identification import (
    IDENTIFIED,
    IDENTIFIED_REVIEW,
    PROVIDER_ERROR,
    RATE_LIMITED,
    Providers,
    ReferenceIdentifier,
)
from backend.config import get_settings

IDENTIFICATION_BUDGET_SECONDS = 35.0


def _attempt(provider, outcome, reason, stage, url=''):
    return {**Attempt(provider, url, outcome, reason).as_dict(), 'stage': stage}


def _metadata(raw, ident, best):
    if not best:
        status = {'SOURCE_NOT_FOUND': 'not_found', 'AMBIGUOUS_SOURCE': 'ambiguous_match', 'RATE_LIMITED': 'rate_limited',
                  'PROVIDER_ERROR': 'api_unavailable'}.get(ident.get('status'), 'not_found')
        if 'disabled by configuration' in (ident.get('reason') or ''):
            status = 'disabled'
        return ReferenceMetadata(raw_reference=raw, status=status, identification_status=ident.get('status'),
                                 identification=ident)
    return ReferenceMetadata(
        raw_reference=raw, provider=best.get('provider') or 'unknown', status='enriched',
        identification_status=ident.get('status'), title=best.get('title'), doi=best.get('doi'),
        abstract=best.get('abstract'), venue=best.get('venue'), authors=best.get('authors') or [],
        year=best.get('year'), volume=best.get('volume'), issue=best.get('issue'), pages=best.get('pages'),
        arxiv_id=best.get('arxiv_id'),
        pdf_url=next((loc['url'] for loc in best.get('oa_locations', []) if loc.get('kind') == 'pdf'), None),
        urls=[loc['url'] for loc in best.get('oa_locations', [])][:10], identification=ident)


class SourceResolver:
    def __init__(self, settings=None, transport=None, pace=True, cache_dir=None):
        self.config = settings or get_settings()
        self.semaphore = asyncio.Semaphore(self.config.source_concurrency)
        self.cache = {}
        self.digests = {}                       # sha256 -> source_id (duplicate-document detection)
        self.identifier = ReferenceIdentifier(self.config, transport, pace, cache_dir)
        self.blocked_digests = frozenset()
        self._entered = False
        self.stage_seconds = {'identification': 0.0, 'full_text_retrieval': 0.0}

    async def __aenter__(self):
        await self.identifier.__aenter__()
        self._entered = True
        return self

    async def __aexit__(self, *exc):
        await self.identifier.__aexit__(*exc)

    async def resolve(self, raw):
        if not self._entered:
            await self.__aenter__()
        key = ' '.join(raw.lower().split())  # trivially different spellings of one reference share one resolution
        if key not in self.cache:
            self.cache[key] = asyncio.ensure_future(self._slot(raw))
        return await self.cache[key]

    async def _slot(self, raw):
        async with self.semaphore:
            started = time.monotonic()
            try:
                return await asyncio.wait_for(self._resolve(raw), self.config.source_reference_deadline_seconds)
            except TimeoutError:
                record = self._empty(raw)
                record.update(status='RETRIEVAL_FAILED', failure_stage=record.get('failure_stage') or 'full_text_retrieval',
                              reason=f'Per-reference work deadline ({self.config.source_reference_deadline_seconds:g}s) exceeded '
                                     'after the reference started processing; retry may succeed. Support was not assessed.')
                record['attempts'].append(_attempt('resolver', 'TIMEOUT', 'Per-reference deadline exceeded.', 'resolver'))
                record['timings'] = {'total': round(time.monotonic() - started, 3)}
                return record, ReferenceMetadata(raw_reference=raw, status='api_unavailable')

    @staticmethod
    def _empty(raw):
        return {'source_id': 'SRC-' + hashlib.sha256(raw.encode()).hexdigest()[:12], 'raw_reference': raw,
                'status': 'SOURCE_UNAVAILABLE', 'reason': 'Source could not be reliably identified.', 'attempts': [],
                'passages': [], 'abstract_passages': [], 'source_identified': False, 'full_text_retrieved': False,
                'identification': {'status': 'NOT_ATTEMPTED'}, 'full_text': {'status': 'NOT_ATTEMPTED'},
                'failure_stage': None, 'reference': {}, 'timings': {}}

    async def _resolve(self, raw):
        record = self._empty(raw)
        source_id = record['source_id']
        if not raw:
            record['reason'] = 'No unambiguous bibliography entry could be matched to this citation marker.'
            record['identification'] = {'status': 'NO_BIBLIOGRAPHY_ENTRY', 'reason': record['reason']}
            record['failure_stage'] = 'reference_matching'
            record['attempts'].append(_attempt('reference_matching', 'NO_BIBLIOGRAPHY_ENTRY', record['reason'], 'reference_matching'))
            return record, ReferenceMetadata(status='no_bibliography_entry', identification_status='NO_BIBLIOGRAPHY_ENTRY')
        ref = reference_record(raw, 1)
        record['reference'] = {k: ref.get(k) for k in ('title', 'authors', 'year', 'venue', 'volume', 'issue', 'pages',
                                                       'doi', 'arxiv_id', 'urls', 'normalized_reference')}
        # ---- 1. identification
        t0 = time.monotonic()
        try:
            ident = await asyncio.wait_for(self.identifier.identify(raw), IDENTIFICATION_BUDGET_SECONDS)
            ident_dict = ident.as_dict()
        except TimeoutError:
            ident_dict = {'status': PROVIDER_ERROR, 'reason': f'Identification exceeded {IDENTIFICATION_BUDGET_SECONDS:g}s.',
                          'best': None, 'attempts': [], 'candidates': []}
        took = time.monotonic() - t0
        self.stage_seconds['identification'] += took
        record['timings']['identification'] = round(took, 3)
        best = ident_dict.pop('best', None)
        record['attempts'] += [{**a, 'stage': 'identification'} for a in ident_dict.pop('attempts', [])]
        ident_dict['candidates'] = ident_dict.get('candidates', [])[:8]
        ident_dict.pop('reference', None)
        record['identification'] = ident_dict
        metadata = _metadata(raw, ident_dict, best)
        record['metadata'] = metadata.model_dump()
        identified = ident_dict['status'] in IDENTIFIED

        # ---- 2. full-text discovery and retrieval
        t1 = time.monotonic()
        if identified:
            extra = []
            doi = best.get('doi')
            if doi:
                providers: Providers = self.identifier.providers
                for coro in (providers.unpaywall(doi), providers.europepmc(doi)):
                    outcome, locs, attempt = await coro
                    record['attempts'].append({**attempt.as_dict(), 'stage': 'full_text_discovery', 'locations': len(locs)})
                    extra += locs
            locations = discover_locations(best, ref, extra, self.config.source_max_locations)
            target = best
        elif ref.get('urls') or ref.get('arxiv_id'):
            # Unidentified reference with an explicit URL / arXiv id: the document itself may establish identity.
            locations = discover_locations({}, ref, [], self.config.source_max_locations)
            target = {'title': ref.get('title'), 'authors': ref.get('authors') or [], 'doi': ref.get('doi'),
                      'arxiv_id': ref.get('arxiv_id')}
        else:
            locations, target = [], None
        record['full_text'] = {'status': 'NOT_ATTEMPTED', 'locations_considered': [
            {k: loc.get(k) for k in ('url', 'kind', 'provider', 'version')} for loc in locations]}
        fetched = None
        if identified and not locations:
            record['full_text'].update(status='NO_OPEN_ACCESS_LOCATION', failure_reason=(
                'No open-access full-text location is known for this work (no DOI, arXiv id, repository copy or URL '
                'was found by the providers).'))
        if locations and target:
            fetched = await retrieve_full_text(target, ref, locations, self.config, self.identifier.providers,
                                               self.blocked_digests)
            record['attempts'] += [{**a, 'stage': a.get('stage') or 'full_text_retrieval'} for a in fetched.pop('attempts')]
            document = fetched.pop('document')
            record['full_text'].update(fetched)
            if document:
                record['full_text'].update(total_pages=document.get('total_pages'), words=document.get('words'))
        took = time.monotonic() - t1
        self.stage_seconds['full_text_retrieval'] += took
        record['timings']['full_text'] = round(took, 3)

        # ---- 3. outcome
        if fetched and fetched['status'] == 'FULL_TEXT_RETRIEVED':
            digest = fetched['sha256']
            if digest in self.digests and self.digests[digest] != source_id:
                record['full_text']['duplicate_of'] = self.digests[digest]
                record['attempts'].append(_attempt('duplicate_check', 'DUPLICATE_DOCUMENT',
                                                   f'Same document already retrieved for {self.digests[digest]}.', 'full_text_retrieval'))
            self.digests.setdefault(digest, source_id)
            if not identified:
                ident_dict.update(status=IDENTIFIED_REVIEW, reason='Identified from the document at the URL given in the '
                                  'reference (title/authors in its front matter); no metadata provider confirmed it.')
                metadata = ReferenceMetadata(raw_reference=raw, provider='source_document', status='enriched',
                                             identification_status=IDENTIFIED_REVIEW, title=ref.get('title'),
                                             authors=ref.get('authors') or [], year=ref.get('year'), identification=ident_dict)
                record['metadata'] = metadata.model_dump()
            name = metadata.title or metadata.doi or metadata.arxiv_id or source_id
            record.update(status='FULL_TEXT', source_identified=True, full_text_retrieved=True, url=fetched['url'],
                          sha256=digest, reason=f'Full text retrieved from {fetched["provider"]} ({fetched["format"]}); '
                                                f'{fetched["identity"]["reason"]}')
            record['passages'] = [{**p, 'name': name, 'source_id': source_id, 'source_url': fetched['url'],
                                   'evidence_tier': 'full_text', 'extraction_method': fetched['extraction'].get('method')}
                                  for p in document['paragraphs']]
            return record, metadata

        abstract = metadata.abstract if identified and ident_dict['status'] != IDENTIFIED_REVIEW else None
        if abstract and len(abstract.split()) >= 30:
            record['abstract_passages'] = [{'text': abstract, 'name': (metadata.title or source_id) + ' (abstract)',
                                            'source_id': source_id, 'source_url': None, 'section': 'Abstract (provider metadata)',
                                            'page_number': None, 'paragraph': 0, 'evidence_tier': 'abstract',
                                            'extraction_method': f'metadata:{metadata.provider}'}]
        ft = record['full_text']
        ft_status = ft.get('status')
        if not identified and not fetched:
            status = {RATE_LIMITED: 'RETRIEVAL_FAILED', PROVIDER_ERROR: 'RETRIEVAL_FAILED'}.get(ident_dict['status'], 'SOURCE_UNAVAILABLE')
            record.update(status=status, failure_stage='identification', reason=ident_dict.get('reason', record['reason']))
            return record, metadata
        if not identified:
            record.update(status='SOURCE_UNAVAILABLE', failure_stage='identification',
                          reason=ident_dict.get('reason', '') + ' A document at the reference URL could not be validated: '
                          + (ft.get('failure_reason') or ''))
            return record, metadata
        record['source_identified'] = True
        failure = ft.get('failure_reason') or 'No full-text location was tried.'
        if ft_status == 'EXTRACTION_FAILED':
            record.update(status='EXTRACTION_FAILED', failure_stage='extraction',
                          reason='Source identified and a document was downloaded, but readable text could not be extracted. ' + failure)
        elif ft_status == 'RETRIEVAL_FAILED':
            record.update(status='RETRIEVAL_FAILED', failure_stage='full_text_retrieval',
                          reason='Source identified; full-text retrieval did not complete. ' + failure)
        else:
            record.update(status='SOURCE_METADATA_ONLY', failure_stage='full_text_retrieval',
                          reason='Source identified, but full text could not be retrieved. ' + failure + ' Metadata is not evidence.')
        if record['abstract_passages']:
            record['status_without_abstract'] = record['status']
            record['status'] = 'ABSTRACT_ONLY'
        return record, metadata


def identity_matches(metadata, parsed):
    """Compatibility wrapper: (valid, reason) for ReferenceMetadata vs a DocumentParser result (first page only)."""
    meta = metadata.model_dump() if hasattr(metadata, 'model_dump') else dict(metadata)
    front = '\n'.join(p['text'] for p in parsed['pages'][:1])
    verdict = validate_identity(meta, {'front_text': front, 'declared': {}})
    return verdict['valid'], verdict['reason']


__all__ = ['SourceResolver', 'validate_identity', 'identity_matches']
