"""Evidence ranking, NLI gating and end-to-end aggregation with MOCKED models (cases 13-20). No network.

The NLI double below is deterministic and keyed on the premise/hypothesis text, so each test states exactly which
model output it assumes. These tests prove control flow and policy, not model quality.
"""
import asyncio
from unittest.mock import MagicMock, patch

import httpx

from app.core.model_health import ModelStatus, ModelUnavailable
from app.core.schemas import ClaimCitationPair, RetrievedEvidence
from app.services.orchestrator import Orchestrator, finalize
from app.services.retriever import Retriever
from app.services.source_resolver import SourceResolver
from app.services.verifier import Verifier
from tests.helpers import Router, crossref_work, install_http, make_pdf, openalex_work, settings

FULL = {'status': 'FULL_TEXT', 'reason': 'Retrieved', 'source_identified': True, 'attempts': [],
        'identification': {'status': 'IDENTIFIED_HIGH_CONFIDENCE'}}


def scores(e, n, c):
    return [{'label': 'ENTAILMENT', 'score': e}, {'label': 'NEUTRAL', 'score': n}, {'label': 'CONTRADICTION', 'score': c}]


def nli_double(rules):
    """rules: list of (substring present in premise+hypothesis, (e, n, c)); default neutral."""
    def model(pair):
        text = (pair['text'] + ' || ' + pair['text_pair']).lower()
        for key, value in rules:
            if key.lower() in text:
                return scores(*value)
        return scores(.02, .96, .02)
    return MagicMock(side_effect=model)


def verifier_with(rules):
    with patch('app.services.verifier.pipeline') as p:
        p.return_value = nli_double(rules)
        return Verifier(use_nli=True)


def ev(text, rerank=2.0, lexical=.5, eid='EVD-1'):
    return RetrievedEvidence(claim_id='c', source_document='src', evidence_text=text, relevance_score=rerank,
                             rerank_score=rerank, retrieval_score=lexical, ranking_method='cross_encoder', evidence_id=eid)


def run_claim(verifier, text, evidence, components=()):
    claim = ClaimCitationPair(id='c', text=text, citation_marker='[1]', predicate_components=list(components))
    result = verifier.verify(claim, evidence, False)
    finalize(result, claim, FULL)
    if result.reason_code == 'NOT_ENTAILED' and components:
        comp = verifier.component_checks(list(components), result.evidence_list)
        result.nli['components'] = comp
        finalize(result, claim, FULL, components=[r['verdict'] for r in comp])
    return result


def test_correct_supporting_evidence_records_full_nli_trace():           # case 13
    v = verifier_with([('2 mg dose', (.97, .02, .01))])
    r = run_claim(v, 'Patients received a 2 mg dose daily.', [ev('All patients received a 2 mg dose every day.')])
    assert r.final_verdict == 'VERIFIED_SUPPORTED' and r.semantic_verification_completed
    passage = r.evidence_list[0].nli
    assert passage['operational_state'] == 'INFERENCE_COMPLETED' and passage['model']
    assert set(passage['scores']) == {'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'} and passage['raw']
    assert passage['seconds'] >= 0 and v.usage['passages_scored'] >= 1


def test_partially_supporting_evidence_uses_components():                # case 14
    rules = [('can search the literature.', (.95, .04, .01))]   # matches the component hypothesis only
    v = verifier_with(rules)
    components = ['LLM systems can search the literature.', 'LLM systems can plan experiments.']
    r = run_claim(v, 'LLM systems can search the literature and plan experiments.',
                  [ev('Our LLM system can search the literature for relevant work.')], components)
    assert r.final_verdict == 'VERIFIED_PARTIALLY_SUPPORTED' and r.review_required
    assert [c['verdict'] for c in r.nli['components']] == ['VERIFIED_SUPPORTED', 'INSUFFICIENT_EVIDENCE']


def test_grounded_contradiction_requires_symmetry():                      # case 15
    v = verifier_with([('never administered', (.01, .04, .95))])
    r = run_claim(v, 'The dose was 2 mg.', [ev('The dose was never administered.')])
    assert r.final_verdict == 'VERIFIED_CONTRADICTED'
    assert r.evidence_list[0].nli['reverse_contradiction'] >= .5


def test_one_directional_contradiction_is_not_a_verdict():
    def model(pair):
        # contradiction only when the claim is the hypothesis (typical MNLI "extra detail" artefact)
        return scores(.01, .04, .95) if pair['text_pair'].startswith('LLMs have challenged') else scores(.05, .9, .05)
    with patch('app.services.verifier.pipeline') as p:
        p.return_value = MagicMock(side_effect=model)
        v = Verifier(use_nli=True)
    r = run_claim(v, 'LLMs have challenged hypotheses and revised interpretations.',
                  [ev('This work shows that an LLM agent can support autonomous experiments by forming hypotheses.')])
    assert r.final_verdict == 'NEEDS_HUMAN_REVIEW' and r.reason_code == 'UNGROUNDED_CONTRADICTION'


def test_topically_similar_but_irrelevant_evidence_is_not_support():      # case 16
    v = verifier_with([('dose escalation', (.10, .85, .05))])
    r = run_claim(v, 'The dose was 2 mg.', [ev('The dose escalation protocol is described in Section 4.')])
    assert r.final_verdict == 'INSUFFICIENT_EVIDENCE' and r.review_required


def test_entailment_from_an_off_topic_passage_is_gated_out():
    v = verifier_with([('prompt design', (.95, .04, .01))])
    r = run_claim(v, 'Separating generation and evaluation reduces self-confirmation bias.',
                  [ev('To illustrate prompt design issues we take the constrained generation task.', rerank=-3.7)])
    assert r.final_verdict == 'INSUFFICIENT_EVIDENCE'
    assert r.nli['gated_out'] and 'weakly relevant' in r.nli['gated_out'][0]['gate']


def test_nli_unavailable_never_verifies():                                # case 17
    status = ModelStatus('nli', 'x', status='UNAVAILABLE', stage='load', cause='WEIGHTS_NOT_CACHED', error='no cache')
    with patch('app.services.verifier.pipeline', side_effect=ModelUnavailable(status)):
        v = Verifier(use_nli=True)
    r = run_claim(v, 'The dose was 2 mg.', [ev('The dose was 2 mg.')])
    assert r.final_verdict == 'NLI_UNAVAILABLE' and r.failure_stage == 'nli' and r.review_required


def test_cross_encoder_unavailable_falls_back_to_lexical_ranking():       # case 18
    status = ModelStatus('cross_encoder', 'x', status='UNAVAILABLE', stage='load', cause='WEIGHTS_NOT_CACHED', error='no cache')
    with patch('app.services.retriever.CrossEncoder', side_effect=ModelUnavailable(status)):
        retriever = Retriever(use_reranker=True, chunk_size=250)
    assert retriever.use_reranker is False and retriever.reranker_unavailable['cause'] == 'WEIGHTS_NOT_CACHED'
    retriever.add_sources([{'name': 'S', 'text': 'Patients received a 2 mg dose. Unrelated sentence about weather.', 'section': 'Results'},
                           {'name': 'S', 'text': '[1] A. Author. A cited paper about doses. 2001.', 'section': 'References'}])
    claim = ClaimCitationPair(id='c', text='Patients received a 2 mg dose.', citation_marker='[1]')
    out = retriever.retrieve(claim)
    assert out and out[0].ranking_method == 'lexical' and out[0].rerank_score is None
    assert all('cited paper' not in e.evidence_text for e in out)          # bibliography is never evidence
    v = verifier_with([('2 mg dose', (.97, .02, .01))])
    r = run_claim(v, claim.text, out)
    assert r.final_verdict == 'VERIFIED_SUPPORTED'                        # NLI decides; lexical ranking only proposed


def test_passages_keep_location_context_and_ranks():
    retriever = Retriever(use_reranker=False, chunk_size=250)
    para = ' '.join(f'Sentence number {i} talks about calibration of spin ensembles.' for i in range(8))
    retriever.add_sources([{'name': 'S', 'text': para, 'page_number': 4, 'section': '2 Methods', 'paragraph': 7,
                            'source_id': 'SRC-1', 'evidence_tier': 'full_text', 'extraction_method': 'pdf_text:pdfium'}])
    out = retriever.retrieve(ClaimCitationPair(id='c', text='calibration of spin ensembles', citation_marker='[1]'), top_k=3)
    assert out[0].page_number == 4 and out[0].section == '2 Methods' and out[0].paragraph == 7
    assert out[0].context == para and out[0].lexical_rank == 1 and out[0].extraction_method == 'pdf_text:pdfium'
    assert len({e.evidence_text for e in out}) == len(out)                 # no duplicate passages


# ---- end to end through the orchestrator with mocked providers and models (cases 19, 20) -----------------------

TITLE_A = 'Autonomous laboratories select experiments from feedback'
TITLE_B = 'Closed-loop experimentation with mobile robots'
BODY_A = ['The autonomous laboratory used experimental feedback to select and carry out further experiments.'] * 30
BODY_B = ['The mobile robot used feedback from each result to choose the next experiment without human input.'] * 30
MANUSCRIPT = ('Autonomous laboratories have used experimental feedback to select and carry out further experiments [1, 2].\n'
              'References\n'
              f'[1] Ada Lovelace and Charles Babbage. {TITLE_A}. Journal of Automation, 3(1):1–9, 2020. doi: 10.5555/a.\n'
              f'[2] Grace Hopper. {TITLE_B}. Robotics Letters, 4(2):10–19, 2021. doi: 10.5555/b.\n'
              '[3] Alan Turing. A reference that nobody cites. 1950.')


def test_end_to_end_two_references_supporting_one_claim(monkeypatch, tmp_path):   # cases 19 + 20
    router = Router()
    for doi, title, authors, body, venue in (('10.5555/a', TITLE_A, ['Ada Lovelace', 'Charles Babbage'], BODY_A, 'Journal of Automation'),
                                             ('10.5555/b', TITLE_B, ['Grace Hopper'], BODY_B, 'Robotics Letters')):
        router.add(f'api.crossref.org/works/{doi.replace("/", "%2F")}', {'message': crossref_work(title, authors, 2020 if doi.endswith('a') else 2021, doi, venue=venue)})
        router.add(f'doi.org/{doi}', openalex_work(title, authors, 2020 if doi.endswith('a') else 2021, doi, venue=venue,
                                                   pdf=f'https://repo.example/{doi[-1]}.pdf'))
        router.add(f'repo.example/{doi[-1]}.pdf', httpx.Response(200, content=make_pdf([title, ', '.join(authors), f'doi:{doi}'] + body)))
    install_http(monkeypatch, router, tmp_path)
    cfg = settings()
    monkeypatch.setattr('app.services.source_resolver.get_settings', lambda: cfg)
    monkeypatch.setattr('app.services.orchestrator.model_health', lambda: {'nli': {'status': 'READY'}, 'cross_encoder': {'status': 'READY'}})
    reranker = MagicMock()
    reranker.predict.side_effect = lambda pairs: [3.0 if 'feedback' in doc else -6.0 for _, doc in pairs]
    rules = [('further experiments. || autonomous laboratories', (.97, .02, .01))]   # only source A's passages entail

    async def run():
        with patch('app.services.retriever.CrossEncoder', return_value=reranker), \
             patch('app.services.verifier.pipeline', return_value=nli_double(rules)):
            async with SourceResolver(cfg, pace=False, cache_dir=None) as resolver:
                return await Orchestrator(use_models=True).run_pipeline(target_text=MANUSCRIPT, resolver=resolver)
    result = asyncio.run(run())
    verdicts = {c.citation_marker: c.final_verdict for c in result.claims}
    assert verdicts == {'[1]': 'VERIFIED_SUPPORTED', '[2]': 'INSUFFICIENT_EVIDENCE'}
    st = result.summary.stats
    assert st['claim_citation_pairs'] == 2 == sum(st['verdict_counts'].values())
    assert st['proposition_rollup'] == {'SUPPORTED_BY_AT_LEAST_ONE_CITATION': 1}
    assert st['pairs_full_text_retrieved'] == 2 and st['pairs_with_full_text_passages'] == 2
    assert st['support_rate'] == {**st['support_rate'], 'numerator': 1, 'denominator': 2}
    assert st['review_required'] == 1                                   # the unresolved pair is in the queue
    refs = st['references']
    assert (refs['total_references'], refs['identified'], refs['full_text_retrieved'], refs['usable_extracted_text']) == (3, 2, 2, 2)
    assert refs['references_not_cited_or_not_matched'] == 1
    assert result.model_usage['nli']['state'] == 'INFERENCE_COMPLETED' and result.model_usage['nli']['passages_scored'] > 0
    assert result.model_usage['cross_encoder']['rerank_pairs'] > 0
    supported = next(c for c in result.claims if c.citation_marker == '[1]')
    assert supported.evidence_list[0].page_number == 1 and supported.evidence_list[0].rerank_score == 3.0
    assert 'consistency_error' not in st and len(result.references) == 2
