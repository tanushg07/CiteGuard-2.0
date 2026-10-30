"""Downloader behaviour against MOCKED HTTP (no live sites). Needs httpx + pydantic-settings; skipped otherwise."""
import asyncio

import pytest

httpx = pytest.importorskip('httpx')
pytest.importorskip('pydantic_settings')

from app.services import downloader  # noqa: E402
from backend.config import get_settings  # noqa: E402

PDF = b'%PDF-1.4\n' + b'x' * 200


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, 'CACHE', tmp_path / 'cache')
    async def allow(_url): return None
    monkeypatch.setattr(downloader, 'validate_url', allow)
    sleeps = []
    async def fake_sleep(seconds): sleeps.append(seconds)
    real = httpx.AsyncClient
    def install(handler):
        monkeypatch.setattr(downloader.httpx, 'AsyncClient',
                            lambda **kw: real(transport=httpx.MockTransport(handler), **{k: v for k, v in kw.items() if k != 'trust_env'}))
    return install, sleeps, fake_sleep


def run(coro): return asyncio.run(coro)


def test_403_is_recorded_not_retried_or_bypassed(harness):             # case 5
    install, sleeps, fake_sleep = harness
    calls = []
    install(lambda r: (calls.append(1), httpx.Response(403))[1])
    out = run(downloader.fetch_pdf('https://pub.example/paper.pdf', get_settings(), sleep=fake_sleep))
    assert out['status'] == 'SOURCE_UNAVAILABLE' and len(calls) == 1 and not sleeps
    assert out['attempts'][0]['outcome'] == 'ACCESS_DENIED' and out['attempts'][0]['http_status'] == 403


def test_404_means_not_found_not_inaccessible(harness):
    install, _, fake_sleep = harness
    install(lambda r: httpx.Response(404))
    out = run(downloader.fetch_pdf('https://pub.example/missing.pdf', get_settings(), sleep=fake_sleep))
    assert out['attempts'][0]['outcome'] == 'NOT_FOUND'


def test_rate_limit_honours_retry_after_then_succeeds(harness):        # case 9
    install, sleeps, fake_sleep = harness
    state = {'n': 0}
    def handler(request):
        state['n'] += 1
        return httpx.Response(429, headers={'retry-after': '3'}) if state['n'] == 1 else httpx.Response(200, content=PDF)
    install(handler)
    out = run(downloader.fetch_pdf('https://pub.example/ok.pdf', get_settings(), sleep=fake_sleep))
    assert out['status'] == 'FETCHED' and sleeps == [3.0]
    assert [a['outcome'] for a in out['attempts']] == ['RATE_LIMITED', 'FETCHED']


def test_persistent_503_is_bounded_and_reported(harness):
    install, sleeps, fake_sleep = harness
    s = get_settings()
    install(lambda r: httpx.Response(503))
    out = run(downloader.fetch_pdf('https://pub.example/down.pdf', s, sleep=fake_sleep))
    assert out['status'] == 'SOURCE_UNAVAILABLE'
    assert len(out['attempts']) == s.source_max_retries + 1
    assert len(sleeps) == s.source_max_retries


def test_timeout_is_recorded(harness):
    install, _, fake_sleep = harness
    def handler(request): raise httpx.ReadTimeout('slow', request=request)
    install(handler)
    out = run(downloader.fetch_pdf('https://pub.example/slow.pdf', get_settings(), sleep=fake_sleep))
    assert {a['outcome'] for a in out['attempts']} == {'TIMEOUT'}


def test_redirect_hop_and_html_page_are_not_counted_as_fulltext(harness):
    install, _, fake_sleep = harness
    def handler(request):
        if request.url.path == '/a':
            return httpx.Response(302, headers={'location': 'https://pub.example/landing'})
        return httpx.Response(200, content=b'<html>abstract only</html>', headers={'content-type': 'text/html'})
    install(handler)
    out = run(downloader.fetch_pdf('https://pub.example/a', get_settings(), sleep=fake_sleep))
    outcomes = [a['outcome'] for a in out['attempts']]
    assert 'FETCHED' not in outcomes and outcomes[0] == 'REDIRECT' and 'NOT_A_PDF' in outcomes
    # HTML is handed to HTML full-text extraction, never reported as a fetched full-text document
    assert out['status'] == 'HTML_ONLY' and out['format'] == 'html'


def test_second_fetch_is_served_from_cache_without_network(harness):    # case 15
    install, _, fake_sleep = harness
    calls = []
    install(lambda r: (calls.append(1), httpx.Response(200, content=PDF))[1])
    first = run(downloader.fetch_pdf('https://pub.example/c.pdf', get_settings(), sleep=fake_sleep))
    second = run(downloader.fetch_pdf('https://pub.example/c.pdf', get_settings(), sleep=fake_sleep))
    assert first['status'] == 'FETCHED' and second['cached'] is True and len(calls) == 1
    assert second['attempts'][0]['outcome'] == 'CACHED'
