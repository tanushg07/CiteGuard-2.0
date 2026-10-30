"""Full-text retrieval, extraction, identity validation and OCR with MOCKED HTTP (cases 5, 6, 8, 9). No network."""
import asyncio

import httpx
import pytest

from app.services.document_parser import DocumentParser
from app.services.fulltext import extract_html, validate_identity
from app.services.source_resolver import SourceResolver
from tests.helpers import Router, crossref_work, install_http, make_pdf, openalex_work, settings

TITLE = 'Robot scientists close the loop between hypotheses and experiments'
AUTHORS = ['Ada Lovelace', 'Charles Babbage']
RAW = f'[1] Ada Lovelace and Charles Babbage. {TITLE}. Journal of Automation, 12(3):45–67, 2021. doi: 10.5555/robot.2021.'
WORK = crossref_work(TITLE, AUTHORS, 2021, '10.5555/robot.2021', venue='Journal of Automation', volume='12', page='45-67')
BODY = ['The robot scientist generated hypotheses about gene function and tested them with automated experiments.',
        'Each experiment was selected using feedback from the previous results, closing the scientific loop.',
        'The system ran for several weeks without human intervention and recorded every measurement.']


def article_pdf(title=TITLE, repeat=8):
    lines = [title, ', '.join(AUTHORS), 'doi:10.5555/robot.2021', 'Abstract'] + BODY * repeat
    return make_pdf(lines)


def resolve(monkeypatch, tmp_path, router, raw=RAW, **cfg):
    install_http(monkeypatch, router, tmp_path)

    async def run():
        async with SourceResolver(settings(**cfg), pace=False, cache_dir=None) as resolver:
            return await resolver.resolve(raw)
    return asyncio.run(run())


def base_router(**oa):
    return Router().add('api.crossref.org/works/10.5555', {'message': WORK}) \
                   .add('api.openalex.org/works/https://doi.org', openalex_work(TITLE, AUTHORS, 2021, '10.5555/robot.2021', **oa))


def test_open_access_pdf_is_retrieved_validated_and_extracted(monkeypatch, tmp_path):
    router = base_router(pdf='https://repo.example/robot.pdf').add('repo.example/robot.pdf', httpx.Response(200, content=article_pdf()))
    record, metadata = resolve(monkeypatch, tmp_path, router)
    assert record['status'] == 'FULL_TEXT' and record['source_identified'] and record['full_text_retrieved']
    ft = record['full_text']
    assert ft['status'] == 'FULL_TEXT_RETRIEVED' and ft['provider'] == 'openalex' and ft['format'] == 'pdf'
    assert ft['identity']['valid'] and ft['identity']['method'] == 'exact_identifier'
    assert ft['extraction']['method'].startswith('pdf_text') and ft['url'] == 'https://repo.example/robot.pdf'
    assert record['passages'] and all(p['evidence_tier'] == 'full_text' and p['page_number'] == 1 for p in record['passages'])
    assert record['identification']['status'] == 'IDENTIFIED_HIGH_CONFIDENCE'
    stages = {a['stage'] for a in record['attempts']}
    assert {'identification', 'full_text_discovery', 'full_text_retrieval'} <= stages


def test_identified_but_no_full_text_is_metadata_only_not_nonexistent(monkeypatch, tmp_path):   # case 5
    record, metadata = resolve(monkeypatch, tmp_path, base_router().add('doi.org/10.5555', httpx.Response(404)))
    assert record['identification']['status'] == 'IDENTIFIED_HIGH_CONFIDENCE'
    assert record['status'] == 'SOURCE_METADATA_ONLY' and record['source_identified'] is True
    assert record['failure_stage'] == 'full_text_retrieval'
    assert metadata.status == 'enriched' and metadata.doi == '10.5555/robot.2021'


def test_publisher_access_denied_is_respected(monkeypatch, tmp_path):                          # case 6
    router = base_router(pdf='https://publisher.example/paywalled.pdf') \
        .add('publisher.example', httpx.Response(403)).add('doi.org/10.5555', httpx.Response(403))
    record, _ = resolve(monkeypatch, tmp_path, router)
    assert record['full_text']['status'] == 'ACCESS_RESTRICTED'
    assert record['status'] == 'SOURCE_METADATA_ONLY'
    assert 'access controls were respected' in record['reason']
    assert router.count('publisher.example') == 1                       # no retry or workaround on 403


def test_institutional_login_redirect_is_not_followed(monkeypatch, tmp_path):
    router = base_router(pdf='https://publisher.example/a.pdf') \
        .add('publisher.example/a.pdf', httpx.Response(303, headers={'location': 'https://idp.publisher.example/login'})) \
        .add('doi.org/10.5555', httpx.Response(404))
    record, _ = resolve(monkeypatch, tmp_path, router)
    assert router.count('idp.publisher.example') == 0
    assert any(a['outcome'] == 'ACCESS_DENIED' and 'login' in a['reason'] for a in record['attempts'])


def test_pdf_extraction_failure_is_attributed_to_extraction(monkeypatch, tmp_path):            # case 8
    router = base_router(pdf='https://repo.example/broken.pdf') \
        .add('repo.example/broken.pdf', httpx.Response(200, content=b'%PDF-1.4\nthis is not a parseable pdf'))
    monkeypatch.setenv('OCR_MODE', 'off')
    record, _ = resolve(monkeypatch, tmp_path, router)
    assert record['status'] == 'EXTRACTION_FAILED' and record['failure_stage'] == 'extraction'
    assert record['full_text']['status'] == 'EXTRACTION_FAILED'
    assert any(a['outcome'] == 'EXTRACTION_FAILED' for a in record['attempts'])


def test_wrong_document_is_rejected_by_identity_validation(monkeypatch, tmp_path):
    router = base_router(pdf='https://repo.example/other.pdf').add(
        'repo.example/other.pdf', httpx.Response(200, content=make_pdf(['A study of ocean currents', 'Jane Doe'] + BODY * 8)))
    record, _ = resolve(monkeypatch, tmp_path, router)
    assert record['status'] == 'SOURCE_METADATA_ONLY'
    assert any(a['outcome'] == 'IDENTITY_MISMATCH' for a in record['attempts'])
    assert not record['passages']


def test_abstract_only_landing_page_is_not_full_text(monkeypatch, tmp_path):
    html = (f'<html><head><meta name="citation_title" content="{TITLE}"><meta name="citation_doi" content="10.5555/robot.2021">'
            f'</head><body><h1>{TITLE}</h1><p>{BODY[0]}</p></body></html>').encode()
    router = base_router(landing='https://publisher.example/article').add('publisher.example/article', httpx.Response(200, content=html)) \
        .add('doi.org/10.5555', httpx.Response(404))
    record, _ = resolve(monkeypatch, tmp_path, router)
    assert record['status'] in ('SOURCE_METADATA_ONLY', 'ABSTRACT_ONLY')
    assert any(a['outcome'] == 'ABSTRACT_ONLY_PAGE' for a in record['attempts'])


def test_open_html_full_text_is_extracted_without_reference_list():
    paragraphs = ''.join(f'<p>{s} {s}</p>' for s in BODY * 30)
    html = (f'<html><head><meta name="citation_title" content="{TITLE}"></head><body><nav><p>Menu item one two three four five six seven eight</p></nav>'
            f'<h1>{TITLE}</h1><h2>Results</h2>{paragraphs}<h2>References</h2><p>1. Someone else. A cited paper title here. 2001.</p></body></html>').encode()
    doc = extract_html(html, 'publisher')
    assert not doc['abstract_only'] and doc['words'] > 1200
    assert all(p['section'] != 'References' for p in doc['paragraphs'])
    assert not any('Menu item' in p['text'] for p in doc['paragraphs'])
    assert validate_identity({'title': TITLE, 'authors': []}, doc)['valid']


def test_abstract_fallback_is_marked_when_full_text_unavailable(monkeypatch, tmp_path):
    abstract = ' '.join(BODY * 2)
    router = Router().add('api.crossref.org/works/10.5555', {'message': WORK}) \
        .add('api.openalex.org/works/https://doi.org', openalex_work(TITLE, AUTHORS, 2021, '10.5555/robot.2021', abstract=abstract)) \
        .add('doi.org/10.5555', httpx.Response(403))
    record, metadata = resolve(monkeypatch, tmp_path, router)
    assert record['status'] == 'ABSTRACT_ONLY' and record['status_without_abstract'] == 'SOURCE_METADATA_ONLY'
    assert record['abstract_passages'][0]['evidence_tier'] == 'abstract'
    assert record['abstract_passages'][0]['extraction_method'].startswith('metadata:')


def test_unidentified_reference_never_downloads(monkeypatch, tmp_path):
    router = Router()
    record, metadata = resolve(monkeypatch, tmp_path, router, raw='[9] Someone. An obscure memo nobody indexed. 1999.')
    assert record['identification']['status'] == 'SOURCE_NOT_FOUND'
    assert record['status'] == 'SOURCE_UNAVAILABLE' and record['failure_stage'] == 'identification'
    assert not any(a.get('stage') == 'full_text_retrieval' for a in record['attempts'])


def test_submitted_manuscript_is_never_its_own_evidence(monkeypatch, tmp_path):
    import hashlib
    pdf = article_pdf()
    router = base_router(pdf='https://repo.example/robot.pdf').add('repo.example/robot.pdf', httpx.Response(200, content=pdf)) \
        .add('doi.org/10.5555', httpx.Response(404))
    install_http(monkeypatch, router, tmp_path)

    async def run():
        async with SourceResolver(settings(), pace=False, cache_dir=None) as resolver:
            resolver.blocked_digests = frozenset({hashlib.sha256(pdf).hexdigest()})
            return await resolver.resolve(RAW)
    record, _ = asyncio.run(run())
    assert record['status'] != 'FULL_TEXT'
    assert any(a['outcome'] == 'SELF_REFERENCE' for a in record['attempts'])


def test_duplicate_documents_are_detected(monkeypatch, tmp_path):
    raw2 = RAW.replace('[1]', '[2]').replace('Journal of Automation', 'J. Automation')
    router = base_router(pdf='https://repo.example/robot.pdf').add('repo.example/robot.pdf', httpx.Response(200, content=article_pdf()))
    router.add('query.bibliographic', {'message': {'items': [WORK]}})
    install_http(monkeypatch, router, tmp_path)

    async def run():
        async with SourceResolver(settings(), pace=False, cache_dir=None) as resolver:
            first = await resolver.resolve(RAW)
            second = await resolver.resolve(raw2)
            return first, second
    (r1, _), (r2, _) = asyncio.run(run())
    assert r1['status'] == r2['status'] == 'FULL_TEXT'
    assert r2['full_text'].get('duplicate_of') == r1['source_id']


# ---- OCR fallback (case 9) ----------------------------------------------------------------------------------

def test_scanned_pdf_uses_ocr_fallback(tmp_path):
    path = tmp_path / 'scan.pdf'
    path.write_bytes(make_pdf([], blank=True))
    recognised = f'{TITLE}\n' + '\n'.join(BODY)
    parsed = DocumentParser(ocr_recogniser=('fake-ocr', lambda image: recognised)).parse(str(path))
    assert parsed['extraction']['method'] == 'ocr:fake-ocr'
    assert any('robot scientist' in p['text'] for p in parsed['paragraphs'])


def test_scanned_pdf_without_ocr_engine_fails_explicitly(tmp_path, monkeypatch):
    from app.services import ocr
    path = tmp_path / 'scan.pdf'
    path.write_bytes(make_pdf([], blank=True))
    ocr.reset_for_tests()
    monkeypatch.setattr(ocr, '_rapidocr', lambda: (_ for _ in ()).throw(ImportError('not installed')))
    monkeypatch.setattr(ocr, '_tesseract', lambda: (_ for _ in ()).throw(ImportError('not installed')))
    with pytest.raises(RuntimeError, match='OCR_UNAVAILABLE'):
        DocumentParser().parse(str(path))
    ocr.reset_for_tests()


def test_two_column_text_comes_out_in_reading_order(tmp_path):
    """pdfium follows the content stream; the extraction engine is recorded per document."""
    path = tmp_path / 'a.pdf'
    path.write_bytes(article_pdf(repeat=1))
    parsed = DocumentParser().parse(str(path))
    assert parsed['extraction']['method'] == 'pdf_text:pdfium'
