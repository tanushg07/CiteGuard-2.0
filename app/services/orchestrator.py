"""Whole-paper analysis followed by independently traced citation verification.

Manuscript -> reference identification -> full-text retrieval -> text extraction -> atomic claim decomposition
-> citation association -> evidence passage retrieval -> cross-encoder ranking -> NLI inference
-> evidence-grounded verdict (app.core.verdicts.decide) -> transparent report (app.core.stats).

This module wires stages together, records timings and model usage, and surfaces failures with the stage at
which they happened - it does not decide what counts as support.
"""
import asyncio
import copy
import hashlib
import logging
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from app.core.models import model_health
from app.core.schemas import (
    AnalysisResponse,
    DocumentSummary,
    PipelineStepUpdate,
    ReferenceMetadata,
    VerificationLabel,
)
from app.core.stats import PairRecord, StatsInconsistency, compute_stats, reference_stats
from app.core.verdicts import (
    RESOLVED,
    UNCALIBRATED,
    DecisionInput,
    Verdict,
    assert_decision_invariants,
    decide,
    next_action,
)
from app.services.document_parser import DocumentParser
from app.services.extractor import Extractor
from app.services.reference_metadata import reference_for_marker
from app.services.retriever import Retriever
from app.services.source_resolver import SourceResolver
from app.services.sources import sources_for_claim
from app.services.verifier import Verifier
from backend.config import get_settings

logger = logging.getLogger(__name__)
JOB_STORE = {}

LEGACY_LABEL = {Verdict.VERIFIED_SUPPORTED: VerificationLabel.SUPPORTED,
                Verdict.VERIFIED_CONTRADICTED: VerificationLabel.CONTRADICTED}
SOURCE_TEXT = ('FULL_TEXT', 'ABSTRACT_ONLY')


def _failure_stage(decision, source):
    if decision.verdict in RESOLVED or decision.verdict is Verdict.NOT_VERIFIABLE:
        return None
    if decision.reason_code in ('NO_SOURCE_TEXT', 'EXTRACTION_FAILED', 'ABSTRACT_NOT_ENTAILED'):
        return source.get('failure_stage') or ('extraction' if decision.reason_code == 'EXTRACTION_FAILED' else 'identification')
    if decision.reason_code in ('PASSAGE_RETRIEVAL_ERROR', 'NO_RELEVANT_PASSAGE'):
        return 'evidence_retrieval'
    if decision.verdict is Verdict.NLI_UNAVAILABLE:
        return 'nli'
    return None  # semantic outcome (not entailed / needs review): the pipeline completed


def finalize(result, claim, source, retrieval_error=False, components=()):
    """Attach provenance and the centrally decided verdict to a verifier result. Idempotent."""
    result.claim_group_id = claim.claim_group_id
    result.citation_id = claim.citation_id
    result.section = claim.section
    result.sentence_index = claim.sentence_index
    result.association_confidence = claim.association_confidence
    result.association_type = claim.association_type
    result.ambiguity_status = claim.ambiguity_status
    result.reference_metadata = claim.reference_metadata
    result.source_identification = {k: v for k, v in source.items() if k not in ('passages', 'abstract_passages')}
    result.retrieval_attempts = list(source.get('attempts', []))
    result.evidence_state = 'PASSAGES' if result.evidence_list else 'NONE'
    result.retrieval = {'status': 'SUCCESS' if result.evidence_list else ('ERROR' if retrieval_error else 'NO_EVIDENCE'),
                        'top_k': len(result.evidence_list),
                        'candidate_count': result.evidence_list[0].candidate_count if result.evidence_list else 0,
                        'ranking_method': result.evidence_list[0].ranking_method if result.evidence_list else None,
                        'top_rerank_score': max((e.rerank_score for e in result.evidence_list if e.rerank_score is not None), default=None)}
    nli = result.nli or {}
    ident = (source.get('identification') or {}).get('status')
    inputs = DecisionInput(
        claim_type=claim.claim_type, source_status=source.get('status', 'SOURCE_UNAVAILABLE'),
        source_reason=source.get('reason', ''), fallback_status=source.get('status_without_abstract', ''),
        passage_count=len(result.evidence_list), retrieval_error=retrieval_error,
        nli_status=nli.get('status'), nli_label=nli.get('decision_label'), nli_truncated=bool(nli.get('truncated')),
        nli_conflict=bool(nli.get('conflict')), contradiction_grounded=nli.get('contradiction_grounded', True),
        numerical_status=result.numerical_comparison.status if result.numerical_comparison else None,
        association=claim.association_type, identification_status=ident,
        decomposition_uncertain=claim.decomposition_uncertain, component_verdicts=tuple(components))
    decision = decide(inputs)
    assert_decision_invariants(decision, inputs)
    result.final_verdict, result.decision_reason = decision.verdict.value, decision.reason
    result.reason_code = decision.reason_code
    result.review_required = decision.review_required
    result.review_reasons = list(decision.review_reasons)
    result.limitations = list(decision.limitations)
    result.disagreements = list(decision.disagreements)
    result.semantic_verification_completed = decision.semantic_verification_completed
    result.evidence_tier = decision.evidence_tier if result.evidence_list else None
    result.failure_stage = _failure_stage(decision, source)
    result.next_action = next_action(decision.verdict, decision.review_required)
    result.confidence_kind = UNCALIBRATED
    # The legacy label mirrors the centralized verdict; it can never be more favourable than the policy allows.
    result.status = (VerificationLabel.NUMERICAL_MISMATCH if decision.reason_code == 'NUMERICAL_DISAGREEMENT'
                     else LEGACY_LABEL.get(decision.verdict, VerificationLabel.INSUFFICIENT))
    if decision.verdict not in RESOLVED and not decision.semantic_verification_completed:
        result.confidence = None  # no completed semantic check => no score to report
    return result


class Timings:
    """Wall-clock per pipeline stage, so performance claims can be measured rather than assumed."""
    def __init__(self):
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = round(self.seconds.get(name, 0.0) + time.monotonic() - start, 3)

    def add(self, name, value):
        self.seconds[name] = round(self.seconds.get(name, 0.0) + value, 3)


def model_warnings(health):
    out = []
    for key, label, tail in (('nli', 'NLI verification', 'lexical matches cannot establish support.'),
                             ('cross_encoder', 'cross-encoder', 'lexical retrieval only proposes candidate passages.')):
        st = health.get(key, {})
        if st.get('status') != 'READY':
            out.append(f"MODEL_UNAVAILABLE: {label} unavailable ({st.get('cause') or 'unknown cause'}"
                       f"{' at ' + st['stage'] if st.get('stage') else ''}): {st.get('error') or 'not initialised'}. "
                       f"{st.get('remedy') or ''} {tail}".replace('  ', ' ').strip())
    return out


def _reference_row(record):
    md = record.get('metadata') or {}
    ft = record.get('full_text') or {}
    ident = record.get('identification') or {}
    return {'source_id': record.get('source_id'), 'raw_reference': record.get('raw_reference'),
            'status': record.get('status'), 'failure_stage': record.get('failure_stage'), 'reason': record.get('reason'),
            'identification_status': ident.get('status'), 'identification_reason': ident.get('reason'),
            'match_score': ident.get('score'), 'alternatives': ident.get('alternatives', []),
            'title': md.get('title'), 'doi': md.get('doi'), 'arxiv_id': md.get('arxiv_id'), 'year': md.get('year'),
            'venue': md.get('venue'), 'authors': (md.get('authors') or [])[:8], 'provider': md.get('provider'),
            'full_text_status': ft.get('status'), 'full_text_provider': ft.get('provider'), 'full_text_url': ft.get('url'),
            'full_text_format': ft.get('format'), 'full_text_version': ft.get('version'),
            'identity_validation': ft.get('identity'), 'extraction': ft.get('extraction'),
            'full_text_failure_reason': ft.get('failure_reason'), 'duplicate_of': ft.get('duplicate_of'),
            'passages': len(record.get('passages') or []), 'abstract_available': bool(record.get('abstract_passages')),
            'attempts': len(record.get('attempts') or []), 'timings': record.get('timings', {}),
            'locations_considered': ft.get('locations_considered', [])}


class Orchestrator:
    def __init__(self, use_models=None):
        self.parser = DocumentParser()
        self.extractor = Extractor()
        self.use_models = use_models if use_models is not None else get_settings().citeguard_mode == 'neural'
        self.retriever = Retriever(use_reranker=False, chunk_size=250, top_k=3)
        self.verifier = Verifier(use_nli=False)

    async def run_pipeline(self, target_path=None, target_text=None, source_paths=None, source_names=None,
                           source_texts=None, document_title='Uploaded Document', progress_callback=None, job_id=None,
                           resolver=None):
        started = time.monotonic()
        timings = Timings()
        job_id = job_id or f'job_{uuid.uuid4().hex[:10]}'
        warnings = []
        async def notify(step, name, status, detail, count=0, pct=0):
            if progress_callback:
                await progress_callback(PipelineStepUpdate(step_id=step, name=name, status=status,
                    detail=detail, claims_found=count, progress_percentage=pct))
        await notify(1, 'Analyzing whole paper', 'loading', 'Reading all pages and constructing the document structure.', pct=5)
        parser = DocumentParser()
        with timings.stage('pdf_parse_and_structure'):
            if target_path:
                document = await asyncio.wait_for(asyncio.to_thread(parser.parse, target_path), 120)
                target_digest = await asyncio.to_thread(lambda: hashlib.sha256(Path(target_path).read_bytes()).hexdigest())
            elif target_text:
                document = await asyncio.to_thread(parser.parse_text, target_text, document_title)
                target_digest = None
            else:
                raise ValueError('Either a PDF or target text is required.')
        manuscript_id = target_digest or hashlib.sha256((document_title + (target_text or '')).encode()).hexdigest()
        await notify(1, 'Analyzing whole paper', 'complete', f"Read {document['total_pages']} pages and {len(document['sentences'])} sentences.", pct=15)
        await notify(2, 'Extracting citations', 'loading', 'Decomposing cited sentences into atomic propositions.', pct=20)
        with timings.stage('claim_extraction_and_decomposition'):
            claims = self.extractor.extract(document['paragraphs'], document['references'], enrich=False,
                                            total_pages=document['total_pages'], manuscript_id=manuscript_id)
        bad_pages = sum(c.page_mapping != 'ok' for c in claims)
        if bad_pages:
            warnings.append(f'PAGE_MAPPING: {bad_pages} claim-citation pairs have a missing or impossible manuscript page; '
                            'their page is shown as unknown rather than guessed.')
        await notify(2, 'Extracting citations', 'complete', f'{len(claims)} claim-citation pairs detected.', len(claims), 25)
        if not claims:
            warnings.append('No supported citation patterns were detected.')
        supplied = list(source_texts or [])
        for index, path in enumerate(source_paths or []):
            parsed = await asyncio.to_thread(DocumentParser().parse, path)
            supplied.extend({**p, 'name': source_names[index] if source_names else parsed['document_name'],
                             'evidence_tier': 'full_text', 'extraction_method': (parsed.get('extraction') or {}).get('method')}
                            for p in parsed['paragraphs'])
        source_records, claim_sources, source_states = {}, {}, {}
        await notify(3, 'Identifying source papers', 'loading', 'Identifying references and retrieving open full text.', len(claims), 30)
        raw_by_claim = {c.id: reference_for_marker(c.citation_marker, document['references']) for c in claims}
        pending = []
        for c in claims:
            local = sources_for_claim(c, supplied, document['references'])
            if local:
                claim_sources[c.id] = local
                source_states[c.id] = {'status': 'FULL_TEXT', 'source_id': 'USER-SUPPLIED', 'reason': 'Explicitly supplied source.',
                                       'passages': local, 'source_identified': True, 'full_text_retrieved': True,
                                       'identification': {'status': 'USER_SUPPLIED'},
                                       'full_text': {'status': 'FULL_TEXT_RETRIEVED', 'provider': 'user_supplied'},
                                       'attempts': [{'provider': 'user_supplied', 'url': '', 'outcome': 'CACHED',
                                                     'reason': 'Source text was supplied by the user.', 'timestamp': '',
                                                     'stage': 'full_text_retrieval'}]}
            elif raw_by_claim[c.id] not in pending:
                pending.append(raw_by_claim[c.id])
        own_resolver = resolver is None
        resolver = resolver or SourceResolver()
        resolver.blocked_digests = frozenset({target_digest} if target_digest else set())
        completed = verified_count = 0

        def init_models():
            t0 = time.monotonic()
            health = model_health() if self.use_models else {
                'nli': {'status': 'DISABLED', 'cause': 'DISABLED', 'error': 'baseline mode'},
                'cross_encoder': {'status': 'DISABLED', 'cause': 'DISABLED', 'error': 'baseline mode'}}
            retriever = Retriever(self.use_models, self.retriever.chunk_size, self.retriever.top_k)
            verifier = Verifier(self.use_models, self.verifier.entailment_threshold, self.verifier.contradiction_threshold)
            return health, retriever, verifier, time.monotonic() - t0

        # Models load while the network-bound reference resolution runs; each pair is ranked and verified as soon
        # as its source is ready, so CPU-bound work overlaps with downloads instead of waiting for all of them.
        models_task = asyncio.ensure_future(asyncio.to_thread(init_models))
        indexes, results_by_id, retrieval_errors = {}, {}, set()
        cpu_lock = asyncio.Lock()  # one CPU-bound verification at a time (models are shared, torch is multi-threaded)

        def verify_claim(c, retriever, verifier):
            """Index the claim's source passages, rank them, run NLI and the verdict policy. Runs in a worker thread."""
            selected = claim_sources.get(c.id, [])
            key = tuple((s.get('source_id'), s.get('name'), s['text'], s.get('page_number')) for s in selected)
            err = False
            t0 = time.monotonic()
            try:
                if key not in indexes:
                    index = copy.copy(retriever)
                    index.add_sources(selected)
                    indexes[key] = index
                index = indexes[key]
                evidence = index.retrieve(c, index.top_k, 30) if index.chunks else []
            except Exception:  # noqa: BLE001 - recorded as an evidence-retrieval failure for this pair
                logger.exception('Evidence retrieval failed for %s', c.id)
                evidence, err = [], True
            t1 = time.monotonic()
            state = source_states[c.id]
            try:
                result = verifier.verify(c, evidence, False)
                finalize(result, c, state, err)
                # Coordinated predicates: each derived component is checked; the policy rolls them up.
                if result.reason_code in ('NOT_ENTAILED', 'ABSTRACT_NOT_ENTAILED') and result.evidence_list and c.predicate_components:
                    component_results = verifier.component_checks(c.predicate_components, result.evidence_list)
                    result.nli['components'] = component_results
                    finalize(result, c, state, err, components=[r['verdict'] for r in component_results])
                result = verifier.reasoner.explain(result, verifier.use_nli)
            except Exception as exc:  # noqa: BLE001
                logger.exception('Verification failed for %s', c.id)
                result = verifier.verify(c, [], False)
                result.evidence_list = evidence
                result.nli = {'status': 'INFERENCE_FAILED', 'operational_state': 'INFERENCE_FAILED', 'label': None,
                              'score': None, 'decision_label': None, 'reason': f'Verification raised {type(exc).__name__}.'}
                finalize(result, c, state, err)
                result = verifier.reasoner.explain(result, False)
            return result, err, t1 - t0, time.monotonic() - t1

        async def process(c):
            nonlocal verified_count
            _, retriever, verifier, _ = await models_task
            async with cpu_lock:
                result, err, ranking_s, nli_s = await asyncio.to_thread(verify_claim, c, retriever, verifier)
            timings.add('passage_indexing_and_cross_encoder_ranking', ranking_s)
            timings.add('nli_verification_and_explanations', nli_s)
            if err:
                retrieval_errors.add(c.id)
                warnings.append(f'RETRIEVAL_FAILED: passage retrieval raised an error for {c.id}.')
            results_by_id[c.id] = result
            verified_count += 1
            await notify(6, 'Verifying claims', 'loading', f'Verified {verified_count}/{len(claims)} pairs against source passages.',
                         len(claims), 55 + int(40 * verified_count / max(1, len(claims))))

        work = [asyncio.ensure_future(process(c)) for c in claims if c.id in claim_sources]  # user-supplied sources

        async def identified(raw):
            try:
                return raw, await resolver.resolve(raw)
            except Exception as exc:  # noqa: BLE001 - preserved as an explicit failure, never swallowed silently
                logger.exception('Source resolution crashed for a reference')
                record = SourceResolver._empty(raw)
                record.update(status='RETRIEVAL_FAILED', failure_stage='identification',
                              reason=f'Source resolution raised {type(exc).__name__}: {exc}; support was not assessed.')
                record['attempts'].append({'provider': 'resolver', 'url': '', 'outcome': 'INTERNAL_ERROR',
                                           'reason': f'{type(exc).__name__}: {exc}'[:300], 'timestamp': '', 'stage': 'resolver'})
                return raw, (record, ReferenceMetadata(raw_reference=raw, status='api_unavailable'))
        try:
            if own_resolver:
                await resolver.__aenter__()
            with timings.stage('source_resolution_wall_clock'):
                for task in asyncio.as_completed([identified(raw) for raw in pending]):
                    raw, (record, metadata) = await task
                    source_records[raw] = record
                    for c in claims:
                        if c.id not in claim_sources and raw_by_claim[c.id] == raw:
                            c.reference_metadata = metadata
                            claim_sources[c.id] = record.get('passages') or record.get('abstract_passages') or []
                            source_states[c.id] = record
                            work.append(asyncio.ensure_future(process(c)))
                    completed += 1
                    ident = (record.get('identification') or {}).get('status', '?')
                    await notify(3, 'Identifying source papers', 'loading',
                                 f'Resolved {completed}/{len(pending)} references: {ident} / {record["status"]}.',
                                 len(claims), 30 + int(25 * completed / max(1, len(pending))))
        finally:
            if own_resolver:
                await resolver.__aexit__(None, None, None)
        timings.add('identification_total_across_references', resolver.stage_seconds['identification'])
        timings.add('full_text_retrieval_total_across_references', resolver.stage_seconds['full_text_retrieval'])
        records_list = list(source_records.values())
        full = sum(r['status'] == 'FULL_TEXT' for r in records_list)
        await notify(3, 'Identifying source papers', 'complete',
                     f'{sum((r.get("identification") or {}).get("status", "").startswith("IDENTIFIED") for r in records_list)}'
                     f'/{len(records_list)} references identified; {full} full-text sources retrieved.', len(claims), 55)
        with timings.stage('verification_tail_after_resolution_wall_clock'):
            await asyncio.gather(*work)
        health, retriever, verifier, model_seconds = await models_task
        timings.add('model_init_and_health_check', model_seconds)
        if self.use_models:
            warnings.extend(model_warnings(health))
        else:
            warnings.append('MODEL_UNAVAILABLE: models are disabled (baseline mode); no semantic verification will run.')
        results = [results_by_id[c.id] for c in claims]
        await notify(6, 'Verifying claims', 'complete', 'NLI and numerical decisions recorded.', len(claims), 95)

        records = []
        for r in results:
            src = r.source_identification or {}
            full = src.get('status') == 'FULL_TEXT'
            ident = bool(src.get('source_identified')) or full
            records.append(PairRecord(pair_id=r.id, sentence_id=r.claim_group_id, atomic_claim_id=r.atomic_claim_id or r.id,
                                      verdict=r.final_verdict, source_identified=ident, full_text=full,
                                      evidence_count=len(r.evidence_list), semantic_completed=r.semantic_verification_completed,
                                      review_required=r.review_required, evidence_tier=r.evidence_tier,
                                      failure_stage=r.failure_stage, reference_id=src.get('source_id')))
        try:
            stats = compute_stats(records)
        except StatsInconsistency as exc:
            logger.error('Statistics consistency check failed: %s', exc)
            warnings.append(f'INTERNAL_CONSISTENCY_ERROR: {exc}. Reported totals are unvalidated; do not rely on them.')
            stats = compute_stats(records, validate=False)
            stats['consistency_error'] = str(exc)
        stats['references'] = reference_stats(records_list, len(document['references']))
        counts = Counter(r.final_verdict for r in results)
        scored = [r.confidence for r in results if r.confidence is not None]
        usage = {'nli': {**verifier.usage, 'health_status': health.get('nli', {}).get('status')},
                 'cross_encoder': {'model': health.get('cross_encoder', {}).get('model_id'),
                                   'health_status': health.get('cross_encoder', {}).get('status'),
                                   'state': ('INFERENCE_COMPLETED' if retriever.stats['rerank_calls'] > retriever.stats['rerank_failures']
                                             else 'INFERENCE_FAILED' if retriever.stats['rerank_failures']
                                             else 'MODEL_LOADED' if retriever.use_reranker else 'NOT_LOADED'),
                                   **retriever.stats}}
        summary = DocumentSummary(document_name=document_title, total_claims=len(results),
            total_pages=document['total_pages'], total_references=len(document['references']),
            unique_claims=stats['distinct_cited_sentences'], verdict_counts=dict(counts), stats=stats,
            supported_count=counts[Verdict.VERIFIED_SUPPORTED], contradicted_count=counts[Verdict.VERIFIED_CONTRADICTED],
            unrelated_count=counts[Verdict.INSUFFICIENT_EVIDENCE],
            numerical_mismatch_count=sum(r.numerical_comparison is not None and r.numerical_comparison.status == 'FAILED' for r in results),
            supported_percentage=stats['support_rate']['percent'],
            average_confidence=round(sum(scored) / len(scored), 1) if scored else None,
            processing_time_seconds=round(time.monotonic() - started, 2))
        timings.add('total_wall_clock', time.monotonic() - started)
        response = AnalysisResponse(job_id=job_id, summary=summary, claims=results, document=document,
            sources=[{k: v for k, v in r.items() if k not in ('passages', 'abstract_passages')} for r in records_list],
            references=[_reference_row(r) for r in records_list], warnings=warnings,
            model_health=health, model_usage=usage, timings=timings.seconds,
            engines={'retrieval': 'sentence-window passages; bm25+tfidf candidates',
                     'reranker': 'cross-encoder' if retriever.use_reranker else 'lexical (semantic reranker unavailable)',
                     'verification': 'nli' if verifier.use_nli else 'MODEL_UNAVAILABLE'})
        JOB_STORE[job_id] = response
        await notify(7, 'Report complete', 'complete', f'Analyzed {len(results)} citations across {document["total_pages"]} pages.', len(claims), 100)
        return response
