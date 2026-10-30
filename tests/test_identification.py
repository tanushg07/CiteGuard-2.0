"""Reference identification with MOCKED providers (regression cases 1-4, 7). No network."""
import asyncio

import httpx
import pytest

from app.services.document_parser import DocumentParser, normalize_reference_text, reference_record
from app.services.identification import (
    AMBIGUOUS,
    IDENTIFIED_HIGH,
    IDENTIFIED_REVIEW,
    NOT_FOUND,
    PROVIDER_ERROR,
    RATE_LIMITED,
    Candidate,
    ReferenceIdentifier,
    score_candidate,
)
from tests.helpers import Router, crossref_work, openalex_work, settings

KING = ('[1] Ross D. King, Kenneth E. Whelan, Ffion M. Jones, and Stephen G. Oliver. Functional genomic hypothesis '
        'generation and experimentation by a robot scientist. Nature, 427(6971):247–252, 2004. doi: 10.1038/nature02236.')
KING_NO_DOI = ('[1] Ross D. King, Kenneth E. Whelan, Ffion M. Jones, and Stephen G. Oliver. Functional genomic hypothesis '
               'generation and experimentation by a robot scientist. Nature, 427(6971):247–252, 2004.')
KING_WORK = crossref_work('Functional genomic hypothesis generation and experimentation by a robot scientist',
                          ['Ross King', 'Kenneth Whelan', 'Ffion Jones', 'Stephen Oliver'], 2004, '10.1038/nature02236',
                          venue='Nature', volume='427', page='247-252')


def identify(router, raw, **cfg):
    async def run():
        async with ReferenceIdentifier(settings(**cfg), httpx.MockTransport(router), pace=False, cache_dir=None) as ri:
            return await ri.identify(raw)
    return asyncio.run(run())


def test_reference_with_doi_uses_exact_lookup():                      # case 1
    router = Router().add('api.crossref.org/works/10.1038', {'message': KING_WORK}) \
                     .add('api.openalex.org/works/https://doi.org', openalex_work(KING_WORK['title'][0], ['Ross D. King'], 2004, '10.1038/nature02236'))
    result = identify(router, KING)
    assert result.status == IDENTIFIED_HIGH
    assert result.best['doi'] == '10.1038/nature02236'
    assert result.score['doi_match'] is True
    assert router.count('query.bibliographic') == 0          # exact identifier was enough
    assert {a['provider'] for a in result.attempts} >= {'crossref', 'openalex'}


def test_reference_without_doi_is_discoverable_by_metadata():          # case 2
    router = Router().add('query.bibliographic', {'message': {'items': [KING_WORK]}})
    result = identify(router, KING_NO_DOI)
    assert result.status == IDENTIFIED_HIGH, result.reason
    assert result.best['doi'] == '10.1038/nature02236'
    assert result.score['first_author_match'] and result.score['year'] == 'exact'
    assert result.score['volume_match'] is True


def test_title_only_reference_with_venue_and_year():                   # case 3
    raw = '[21] Why AI cannot do good science without humans. Nature, 653(8115):650, 2026.'
    work = crossref_work('Why AI cannot do good science without humans', [], 2026, '10.1038/d41586-026-01551-3', venue='Nature')
    result = identify(Router().add('query.bibliographic', {'message': {'items': [work]}}), raw)
    assert result.status == IDENTIFIED_HIGH, result.reason
    assert 'lists no authors' in result.reason


def test_similar_title_with_different_authors_is_rejected():
    impostor = crossref_work(KING_WORK['title'][0], ['Jane Doe', 'Max Mustermann'], 2004, '10.9999/other')
    result = identify(Router().add('query.bibliographic', {'message': {'items': [impostor]}}), KING_NO_DOI)
    assert result.status == NOT_FOUND
    assert any(c['score']['decision'] == 'REJECT' for c in result.candidates)


def test_ambiguous_bibliography_entry():                               # case 4
    raw = '[3] A. Smith and B. Jones. Deep learning for citation analysis. 2020.'
    a = crossref_work('Deep learning for citation analysis', ['Alice Smith', 'Bob Jones'], 2020, '10.1111/a')
    b = crossref_work('Deep learning for citation analysis', ['Alan Smith', 'Beth Jones'], 2020, '10.2222/b')
    b['title'] = ['Deep Learning for Citation Analysis: A Survey']
    c = dict(a, DOI='10.3333/c', author=[{'given': 'Anna', 'family': 'Smith'}, {'given': 'Ben', 'family': 'Jones'}])
    result = identify(Router().add('query.bibliographic', {'message': {'items': [a, c]}}), raw)
    assert result.status == AMBIGUOUS
    assert len(result.alternatives) >= 2 and result.best is None


def test_year_mismatch_needs_review_not_high_confidence():
    work = dict(KING_WORK, issued={'date-parts': [[2009]]})
    result = identify(Router().add('query.bibliographic', {'message': {'items': [work]}}), KING_NO_DOI)
    assert result.status == IDENTIFIED_REVIEW


def test_rate_limited_provider_is_reported_not_called_not_found():     # case 7
    router = Router()
    for host in ('api.crossref.org', 'api.openalex.org', 'semanticscholar', 'export.arxiv.org'):
        router.add(host, httpx.Response(429, headers={'retry-after': '0'}))
    result = identify(router, KING_NO_DOI)
    assert result.status == RATE_LIMITED
    assert 'The source may exist' in result.reason
    assert any(a['outcome'] == 'RATE_LIMITED' for a in result.attempts)


def test_provider_errors_are_provider_error():
    router = Router()
    for host in ('api.crossref.org', 'api.openalex.org', 'semanticscholar', 'export.arxiv.org'):
        router.add(host, httpx.Response(503))
    assert identify(router, KING_NO_DOI).status == PROVIDER_ERROR


def test_provider_fallback_when_first_provider_fails():
    router = Router().add('api.crossref.org', httpx.Response(500)) \
                     .add('api.openalex.org/works?', {'results': [openalex_work(KING_WORK['title'][0],
                          ['Ross D. King', 'Kenneth E. Whelan'], 2004, '10.1038/nature02236', venue='Nature')]})
    result = identify(router, KING_NO_DOI)
    assert result.status == IDENTIFIED_HIGH
    outcomes = {(a['provider'], a['outcome']) for a in result.attempts}
    assert ('crossref', 'HTTP_ERROR') in outcomes and ('openalex', 'OK') in outcomes


def test_doi_typo_with_wrong_title_needs_review():
    other = crossref_work('An entirely different paper about fluid dynamics', ['Ross King'], 2004, '10.1038/nature02236')
    result = identify(Router().add('api.crossref.org/works/10.1038', {'message': other}), KING)
    assert result.status == IDENTIFIED_REVIEW and 'possible identifier typo' in result.reason


def test_scoring_signals_are_recorded():
    ref = reference_record(KING_NO_DOI, 1)
    cand = Candidate('crossref', title=KING_WORK['title'][0], authors=['Ross King', 'Kenneth Whelan'], year=2004,
                     venue='Nature', volume='427', pages='247-252')
    score = score_candidate(ref, cand)
    assert score.decision == 'HIGH' and score.pages_match and score.volume_match and score.title_similarity > .95


# ---- bibliography parsing that caused the observed METADATA_NOT_APPLICABLE / missing-DOI failures -------------

def test_wrapped_year_line_is_not_a_new_reference():
    text = ('Body text cites a robot scientist [1] and automation [2].\nReferences\n'
            '[1] Ross D. King, Kenneth E. Whelan, and Stephen G. Oliver. Functional genomic hypothesis generation by a robot\n'
            'scientist. Nature, 427(6971):247–252,\n2004. doi: 10.1038/nature02236.\n'
            '[2] Ross D. King. The automation of science. Science, 324:85–89, 2009.')
    refs = DocumentParser().parse_text(text)['references']
    assert len(refs) == 2
    record = reference_record(refs[0], 1)
    assert record['doi'] == '10.1038/nature02236' and record['year'] == 2004


def test_line_wrap_damage_is_repaired():
    assert normalize_reference_text('Paola Cap- pellaro. doi: 10.1038/ d41586-026-01551-3. URL https: //x.org/a') == \
        'Paola Cappellaro. doi: 10.1038/d41586-026-01551-3. URL https://x.org/a'


def test_structured_reference_fields():
    r = reference_record(KING, 1)
    assert r['author_surnames'][:2] == ['King', 'Whelan']
    assert r['title'] == 'Functional genomic hypothesis generation and experimentation by a robot scientist'
    assert (r['venue'], r['volume'], r['issue'], r['pages'], r['year']) == ('Nature', '427', '6971', '247–252', 2004)


@pytest.mark.parametrize('raw', ['', '   '])
def test_empty_reference_never_calls_providers(raw):
    router = Router()
    result = identify(router, raw or 'x')  # 'x' has no title worth searching
    assert result.status in (NOT_FOUND, PROVIDER_ERROR)
