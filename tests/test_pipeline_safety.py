"""End-to-end safety properties at the pipeline layer. Needs the full dependency set; skipped otherwise.
Models are MOCKED here - these tests prove control flow, not model quality."""
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip('pydantic')
pytest.importorskip('pydantic_settings')

from app.core.model_health import ModelStatus, ModelUnavailable  # noqa: E402
from app.core.schemas import ClaimCitationPair, RetrievedEvidence  # noqa: E402
from app.services.extractor import Extractor  # noqa: E402
from app.services.orchestrator import finalize, model_warnings  # noqa: E402
from app.services.verifier import Verifier  # noqa: E402

FULL = {'status': 'FULL_TEXT', 'reason': 'Retrieved', 'source_identified': True, 'attempts': []}


def claim(text='The dose was 2 mg.', **kw):
    return ClaimCitationPair(id='c', text=text, citation_marker='[1]', **kw)


def ev(text, **kw):
    return RetrievedEvidence(claim_id='c', source_document='src', evidence_text=text, relevance_score=.9,
                             retrieval_score=kw.pop('retrieval_score', .6), **kw)


def nli_mock(entail=0.0, neutral=0.0, contra=0.0):
    m = MagicMock()
    m.return_value = [{'label': 'ENTAILMENT', 'score': entail}, {'label': 'NEUTRAL', 'score': neutral}, {'label': 'CONTRADICTION', 'score': contra}]
    return m


def verify(text, passage, model_scores, **kw):
    with patch('app.services.verifier.pipeline') as p:
        p.return_value = nli_mock(**model_scores)
        v = Verifier(use_nli=True)
        c = claim(text, **kw)
        r = v.verify(c, [ev(passage)], False)
    return finalize(r, c, FULL)


def test_supported_requires_entailment_and_passage():                   # case 1
    r = verify('The dose was 2 mg.', 'Patients received a 2 mg dose.', dict(entail=.97, neutral=.02, contra=.01))
    assert r.final_verdict == 'VERIFIED_SUPPORTED' and r.semantic_verification_completed and r.evidence_state == 'PASSAGES'


def test_topical_overlap_is_insufficient_not_supported():               # case 2
    r = verify('The dose was 2 mg.', 'The dose escalation protocol is described in Section 4.', dict(entail=.05, neutral=.9, contra=.05))
    assert r.final_verdict == 'INSUFFICIENT_EVIDENCE'


def test_contradicting_passage():                                       # case 4
    r = verify('The dose was 2 mg.', 'The dose was never administered.', dict(entail=.02, neutral=.08, contra=.9))
    assert r.final_verdict == 'VERIFIED_CONTRADICTED' and r.review_required


def test_conflicting_passages_are_preserved():                          # case 13
    def model(pair):
        # 'passage two' contradicts the claim in both directions (a grounded, symmetric contradiction)
        e, n, c = (.02, .03, .95) if 'passage two' in pair['text'] + pair['text_pair'] else (.95, .03, .02)
        return [{'label': 'ENTAILMENT', 'score': e}, {'label': 'NEUTRAL', 'score': n}, {'label': 'CONTRADICTION', 'score': c}]
    with patch('app.services.verifier.pipeline') as p:
        p.return_value = MagicMock(side_effect=model)
        v = Verifier(use_nli=True)
        c = claim()
        r = finalize(v.verify(c, [ev('passage one'), ev('passage two')], False), c, FULL)
    assert r.final_verdict == 'NEEDS_HUMAN_REVIEW' and 'nli_passage_conflict' in r.disagreements and r.review_required


def test_unavailable_nli_cannot_verify():                               # case 7
    status = ModelStatus('nli', 'x', status='UNAVAILABLE', stage='load', cause='WEIGHTS_NOT_CACHED', error='no cache')
    with patch('app.services.verifier.pipeline', side_effect=ModelUnavailable(status)):
        v = Verifier(use_nli=True)
    assert v.nli_unavailable['cause'] == 'WEIGHTS_NOT_CACHED'
    c = claim()
    r = finalize(v.verify(c, [ev('The dose was 2 mg.')], False), c, FULL)
    assert r.final_verdict == 'NLI_UNAVAILABLE' and r.confidence is None and not r.semantic_verification_completed


def test_unavailable_models_produce_specific_warnings():                # cases 7/8
    out = model_warnings({'nli': {'status': 'UNAVAILABLE', 'cause': 'WEIGHTS_NOT_CACHED', 'stage': 'load', 'error': 'x', 'remedy': 'run prepare_models'},
                          'cross_encoder': {'status': 'UNAVAILABLE', 'cause': 'MISSING_PACKAGE', 'stage': 'import', 'error': 'y'}})
    assert len(out) == 2 and 'WEIGHTS_NOT_CACHED' in out[0] and 'MISSING_PACKAGE' in out[1]
    assert model_warnings({'nli': {'status': 'READY'}, 'cross_encoder': {'status': 'READY'}}) == []


def test_missing_passage_explanation_does_not_imply_examination():      # case 12
    v = Verifier(use_nli=False)
    c = claim()
    r = v.verify(c, [], True)
    finalize(r, c, FULL)
    r = v.reasoner.explain(r, False)
    assert 'No source passage was examined' in r.reasoning and 'listed below' not in r.reasoning


@pytest.mark.parametrize('page,total,expected_page,status', [(None, 10, None, 'missing'), (11, 10, None, 'out_of_range'), (3, 10, 3, 'ok')])
def test_page_mapping_never_defaults_to_page_one(page, total, expected_page, status):   # case 14
    para = {'text': 'The model achieved 95% accuracy on the held-out test set [4].', 'section': 'Results'}
    if page is not None:
        para['page_number'] = page
    pairs = Extractor().extract([para], enrich=False, total_pages=total)
    assert pairs[0].page_number == expected_page and pairs[0].page_mapping == status


def test_extractor_populates_structured_fields_and_scope():             # cases 10, 11
    text = 'BLEU improved on the English-German task over prior systems [8]; the decoder attends to encoder outputs via cross-attention.'
    pairs = Extractor().extract([{'text': text, 'page_number': 2}], references=['[8] Ref eight. Title of work here. 2017.'],
                                enrich=False, total_pages=9, manuscript_id='M1')
    assert [p.atomic_claim_id for p in pairs] == ['CLM-0001.1', 'CLM-0001.2']
    assert {p.sentence_id for p in pairs} == {'CLM-0001'} and all(p.sentence_text == text for p in pairs)
    assert [p.association_type for p in pairs] == ['direct', 'ambiguous_scope']
    assert all(p.manuscript_id == 'M1' and p.resolved_reference_id == 'REF-001' for p in pairs)


def test_summary_stats_consistent_on_text_job():                        # case 16
    import asyncio

    from app.services.orchestrator import Orchestrator
    r = asyncio.run(Orchestrator(use_models=False).run_pipeline(
        target_text='The dose was 2 mg [1]. Efficacy was high in all cohorts [2].', source_texts=[{'name': 'S1', 'text': 'The dose was 2 mg.', 'markers': ['[1]']}]))
    st = r.summary.stats
    assert st['claim_citation_pairs'] == len(r.claims) == sum(st['verdict_counts'].values())
    assert st['supported'] == 0 and st['model_failures'] >= 1           # baseline mode can never verify
    assert 'consistency_error' not in st
    assert r.timings['total_wall_clock'] > 0
