"""Bounded public-document fetcher: redirect validation, content-addressed cache, bounded retries.

Every HTTP request is recorded as an Attempt (provider, url, timestamp, outcome, reason). Access controls
are respected: 401/403 is recorded and not retried or circumvented. Transient failures (timeout, 429, 5xx)
are retried a bounded number of times, honouring Retry-After (capped).

A fetch returns either a PDF ('format': 'pdf') or, when a location serves an HTML article page without a
publisher-declared PDF link, the HTML itself ('format': 'html') so the caller can try permitted HTML full-text
extraction. HTML is never treated as evidence here; identity and substance are validated by the caller.
"""
import asyncio
import hashlib
import ipaddress
import json
import socket
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from app.services.fetch_policy import Attempt, backoff_delay, classify_http, parse_retry_after, provider_for
from backend.config import ROOT, get_settings

CACHE = ROOT / 'data' / 'sources'
REDIRECT_LIMIT = 6
CACHE_TTL_SECONDS = 7 * 86400
# Hosts that only ever serve authentication walls; following them cannot yield permitted full text.
LOGIN_HOST_MARKERS = ('idp.', 'login.', 'auth.', 'sso.', 'shibboleth', 'openathens', 'secure.')


async def validate_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only public HTTP(S) source URLs are allowed.')
    if parsed.port not in (None, 80, 443):
        raise ValueError('Nonstandard source port blocked.')
    addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError('Private or reserved source address blocked.')


class _PDFLinks(HTMLParser):
    """Only publisher-declared citation_pdf_url links are followed, never arbitrary page links."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == 'meta' and attr.get('name', attr.get('property')) == 'citation_pdf_url' and attr.get('content'):
            self.links.append(attr['content'])


def _result(final_url, attempts, **extra):
    """Build a fetch result. `extra` may itself contain 'url' (e.g. a cache manifest); it wins, because it is the
    final URL actually fetched. (A positional parameter named 'url' made every successful fetch raise TypeError.)"""
    return {'url': final_url, **extra, 'attempts': [a.as_dict() for a in attempts]}


def _login_wall(url):
    host = (urlsplit(url).hostname or '').lower()
    return any(host.startswith(m) or m in host for m in LOGIN_HOST_MARKERS)


async def fetch_pdf(url, settings=None, sleep=asyncio.sleep):
    """Fetch one location. Returns a dict with 'status' FETCHED|SOURCE_UNAVAILABLE, 'attempts' (always), and on
    success 'path', 'sha256', 'format' ('pdf' or 'html').

    `sleep` is injectable so retry behaviour is testable without real waiting.
    """
    config = settings or get_settings()
    attempts: list[Attempt] = []
    budget = getattr(config, 'source_url_budget_seconds', 45.0)
    try:
        return await asyncio.wait_for(_fetch(url, config, attempts, sleep), timeout=budget)
    except TimeoutError:
        attempts.append(Attempt(provider_for(url), url, 'TIMEOUT', f'Per-location time budget ({budget:g}s) exceeded.'))
        return _result(url, attempts, status='SOURCE_UNAVAILABLE', reason='Retrieval time budget exceeded.')


def _cached(url, attempts):
    key = hashlib.sha256(url.encode()).hexdigest()
    manifest = CACHE / (key + '.json')
    if not manifest.exists():
        return None
    try:
        record = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    suffix = '.html' if record.get('format') == 'html' else '.pdf'
    path = CACHE / (record.get('sha256', '') + suffix)
    if not path.exists() or time.time() - record.get('fetched_at', 0) >= CACHE_TTL_SECONDS:
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        return None
    attempts.append(Attempt(provider_for(url), url, 'CACHED', 'Served from the local content-addressed cache.', cached=True))
    return _result(url, attempts, **{**record, 'format': record.get('format', 'pdf'), 'path': str(path), 'cached': True})


async def _store(original, final_url, content, fmt):
    digest = hashlib.sha256(content).hexdigest()
    path = CACHE / (digest + ('.html' if fmt == 'html' else '.pdf'))
    await asyncio.to_thread(path.write_bytes, content)
    # HTML is a candidate for full-text extraction only; it is never reported as a fetched full-text document.
    record = {'status': 'FETCHED' if fmt == 'pdf' else 'HTML_ONLY', 'url': final_url, 'requested_url': original,
              'sha256': digest, 'format': fmt,
              'bytes': len(content), 'fetched_at': time.time()}
    manifest = CACHE / (hashlib.sha256(original.encode()).hexdigest() + '.json')
    await asyncio.to_thread(manifest.write_text, json.dumps(record), encoding='utf-8')
    return record, path


async def _fetch(url, config, attempts, sleep):
    original = url
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = _cached(url, attempts)
    if cached:
        return cached
    headers = {'User-Agent': f'CiteGuard/2.1 (mailto:{config.api_contact_email})',
               'Accept': 'application/pdf,text/html;q=0.9,*/*;q=0.5'}
    followed_pdf_link = False
    async with httpx.AsyncClient(timeout=config.source_timeout_seconds, follow_redirects=False, trust_env=False) as client:
        for _ in range(REDIRECT_LIMIT):
            if _login_wall(url):
                attempts.append(Attempt(provider_for(url), url, 'ACCESS_DENIED',
                                        'Location redirected to an institutional login; access controls are respected.'))
                return _result(url, attempts, status='SOURCE_UNAVAILABLE', reason='Full text requires authentication.')
            outcome = await _get_with_retries(client, url, headers, config, attempts, sleep)
            if outcome['kind'] == 'redirect':
                url = outcome['url']
                continue
            if outcome['kind'] == 'html':
                parser = _PDFLinks()
                parser.feed(outcome['content'][:2_000_000].decode('utf-8', errors='replace'))
                if parser.links and not followed_pdf_link:
                    followed_pdf_link = True
                    html_record = await _store(original + '#landing', url, outcome['content'], 'html')
                    pdf_url = urljoin(url, parser.links[0])
                    attempts.append(Attempt(provider_for(url), url, 'PDF_LINK_DECLARED',
                                            f'Publisher page declares citation_pdf_url {pdf_url}.'))
                    pdf = await _follow_pdf_link(client, pdf_url, headers, config, attempts, sleep)
                    if pdf is not None:
                        record, path = await _store(original, pdf['url'], pdf['content'], 'pdf')
                        return _result(pdf['url'], attempts, **{**record, 'path': str(path), 'cached': False})
                    record, path = html_record
                    return _result(url, attempts, **{**record, 'path': str(path), 'cached': False,
                                                     'note': 'Declared PDF was not retrievable; landing HTML returned.'})
                attempts.append(Attempt(provider_for(url), url, 'NOT_A_PDF',
                                        'Location returned a web page with no publisher PDF link; '
                                        'only HTML full-text extraction can be attempted.'))
                record, path = await _store(original, url, outcome['content'], 'html')
                return _result(url, attempts, **{**record, 'path': str(path), 'cached': False})
            if outcome['kind'] == 'failed':
                return _result(url, attempts, status='SOURCE_UNAVAILABLE', reason=outcome['reason'])
            record, path = await _store(original, url, outcome['content'], 'pdf')
            return _result(url, attempts, **{**record, 'path': str(path), 'cached': False})
    attempts.append(Attempt(provider_for(url), url, 'HTTP_ERROR', 'Redirect limit exceeded.'))
    return _result(url, attempts, status='SOURCE_UNAVAILABLE', reason='Source redirect limit exceeded.')


async def _follow_pdf_link(client, url, headers, config, attempts, sleep):
    """Follow a declared PDF link through redirects. Returns {'url','content'} for a real PDF, else None."""
    for _ in range(REDIRECT_LIMIT):
        if _login_wall(url):
            attempts.append(Attempt(provider_for(url), url, 'ACCESS_DENIED',
                                    'Declared PDF redirected to an institutional login; access controls are respected.'))
            return None
        outcome = await _get_with_retries(client, url, headers, config, attempts, sleep)
        if outcome['kind'] == 'redirect':
            url = outcome['url']
            continue
        if outcome['kind'] == 'pdf':
            return {'url': url, 'content': outcome['content']}
        if outcome['kind'] == 'html':
            attempts.append(Attempt(provider_for(url), url, 'NOT_A_PDF', 'Declared PDF link returned a web page.'))
        return None
    return None


async def _get_with_retries(client, url, headers, config, attempts, sleep):
    """One logical GET. Retries transient outcomes; returns kind in {redirect, html, pdf, failed}."""
    provider = provider_for(url)
    for number in range(config.source_max_retries + 1):
        started = time.monotonic()
        retry_after = None

        def note(outcome, reason, http=None):
            attempts.append(Attempt(provider, url, outcome, reason, http, duration_seconds=round(time.monotonic() - started, 3),
                                    attempt_number=number + 1))
        try:
            await validate_url(url)
            async with client.stream('GET', url, headers=headers) as response:
                status = response.status_code
                if status in (301, 302, 303, 307, 308):
                    note('REDIRECT', f'Redirect HTTP {status}.', status)
                    return {'kind': 'redirect', 'url': urljoin(url, response.headers.get('location', ''))}
                outcome, retryable = classify_http(status)
                if outcome != 'FETCHED':
                    note(outcome, f'Server returned HTTP {status}.', status)
                    retry_after = parse_retry_after(response.headers.get('retry-after'), config.source_retry_after_cap_seconds)
                    if not retryable:
                        return {'kind': 'failed', 'reason': f'Full-text server returned HTTP {status}.', 'outcome': outcome}
                else:
                    if int(response.headers.get('content-length', '0') or 0) > config.source_max_bytes:
                        raise ValueError('Source exceeds size limit.')
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > config.source_max_bytes:
                            raise ValueError('Source exceeds size limit.')
                        chunks.append(chunk)
                    content = b''.join(chunks)
                    is_pdf = content.startswith(b'%PDF-')
                    note('FETCHED' if is_pdf else 'HTML_PAGE', f'HTTP 200, {size} bytes' + ('' if is_pdf else ', not a PDF') + '.', 200)
                    return {'kind': 'pdf' if is_pdf else 'html', 'content': content}
        except httpx.TimeoutException as exc:
            note('TIMEOUT', f'Request timed out ({type(exc).__name__}).')
        except httpx.HTTPError as exc:
            note('NETWORK_ERROR', f'{type(exc).__name__}: network failure.')
        except ValueError as exc:
            kind = 'TOO_LARGE' if 'size' in str(exc) else 'BLOCKED_ADDRESS'
            note(kind, str(exc))
            return {'kind': 'failed', 'reason': str(exc), 'outcome': kind}
        except OSError as exc:
            note('NETWORK_ERROR', f'{type(exc).__name__}: {exc}')
        if number < config.source_max_retries:
            await sleep(backoff_delay(number, config.source_backoff_seconds, retry_after, config.source_retry_after_cap_seconds))
    return {'kind': 'failed', 'reason': f'Retrieval failed after {config.source_max_retries + 1} attempts: {attempts[-1].outcome}.',
            'outcome': attempts[-1].outcome}


async def download_source_pdf(url: str, title: str = '') -> str | None:
    """Compatibility helper; cached PDFs belong to the cache, not to the caller."""
    result = await fetch_pdf(url)
    return result.get('path') if result.get('format', 'pdf') == 'pdf' else None
