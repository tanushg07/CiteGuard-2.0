"""Deterministic test doubles shared by the v2.1 regression tests (no network, no models)."""
import json

import httpx

from backend.config import Settings


def settings(**overrides):
    base = dict(crossref_enabled=True, openalex_enabled=True, arxiv_enabled=True, semantic_scholar_enabled=True,
                unpaywall_enabled=True, europepmc_enabled=True, api_contact_email='tests@citeguard.invalid',
                identification_cache_days=0, source_max_retries=0, source_backoff_seconds=0, groq_api_key='')
    base.update(overrides)
    return Settings(_env_file=None, **base)


def crossref_work(title, authors, year, doi, venue='Example Journal', volume=None, page=None, links=()):
    return {'title': [title], 'author': [{'given': a.split()[0], 'family': a.split()[-1]} for a in authors],
            'issued': {'date-parts': [[year]]}, 'DOI': doi, 'container-title': [venue], 'volume': volume, 'page': page,
            'link': [{'URL': u, 'content-type': 'application/pdf'} for u in links], 'type': 'journal-article'}


def openalex_work(title, authors, year, doi=None, venue='Example Journal', pdf=None, landing=None, abstract=None):
    loc = {'is_oa': bool(pdf or landing), 'pdf_url': pdf, 'landing_page_url': landing}
    inv = {w: [i] for i, w in enumerate(abstract.split())} if abstract else None
    return {'title': title, 'publication_year': year, 'doi': f'https://doi.org/{doi}' if doi else None,
            'authorships': [{'author': {'display_name': a}} for a in authors],
            'primary_location': {'source': {'display_name': venue}}, 'best_oa_location': loc if (pdf or landing) else None,
            'locations': [loc] if (pdf or landing) else [], 'abstract_inverted_index': inv, 'ids': {}, 'biblio': {}}


class Router:
    """Route mocked HTTP requests by host/path substring; records every request."""

    def __init__(self):
        self.routes = []
        self.requests = []

    def add(self, match, response):
        self.routes.append((match, response))
        return self

    def __call__(self, request):
        self.requests.append(request)
        url = str(request.url)
        for match, response in self.routes:
            if match in url:
                result = response(request) if callable(response) else response
                return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)
        # Unrouted provider queries behave like "nothing found".
        if 'api.crossref.org/works?' in url or url.endswith('api.crossref.org/works'):
            return httpx.Response(200, json={'message': {'items': []}})
        if 'api.openalex.org/works?' in url:
            return httpx.Response(200, json={'results': []})
        if 'search/match' in url:
            return httpx.Response(404, json={'error': 'Title match not found'})
        if 'export.arxiv.org' in url:
            return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
        if 'europepmc' in url:
            return httpx.Response(200, json={'resultList': {'result': []}})
        if 'unpaywall' in url:
            return httpx.Response(404, json={'error': True})
        return httpx.Response(404)

    def count(self, fragment):
        return sum(fragment in str(r.url) for r in self.requests)


def install_http(monkeypatch, router, tmp_path):
    """Route ALL httpx.AsyncClient traffic (identification + downloads) through `router`; isolate caches."""
    from app.services import downloader, fulltext
    real = httpx.AsyncClient

    class RoutedAsyncClient(real):  # a real subclass, so libraries that subclass httpx.AsyncClient keep working
        def __init__(self, *args, **kwargs):
            kwargs.pop('transport', None)
            kwargs.pop('trust_env', None)
            super().__init__(*args, transport=httpx.MockTransport(router), **kwargs)
    monkeypatch.setattr(httpx, 'AsyncClient', RoutedAsyncClient)

    async def allow(_url):
        return None
    monkeypatch.setattr(downloader, 'validate_url', allow)
    monkeypatch.setattr(downloader, 'CACHE', tmp_path / 'sources')
    monkeypatch.setattr(fulltext, 'PARSED_CACHE', tmp_path / 'parsed')


def make_pdf(lines, blank=False):
    """A minimal valid one-page PDF whose text layer contains `lines` (or no text at all when blank=True)."""
    if blank:
        stream = b''
    else:
        ops = ['BT /F1 9 Tf 40 780 Td 11 TL']
        for line in lines:
            safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            ops.append(f'({safe}) Tj T*')
        ops.append('ET')
        stream = '\n'.join(ops).encode('latin-1', errors='replace')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    data = b'%PDF-1.4\n'
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'
    xref = len(data)
    data += b'xref\n0 6\n0000000000 65535 f \n' + b''.join(f'{o:010d} 00000 n \n'.encode() for o in offsets)
    data += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode()
    return data


def article_lines(title, authors, body_sentences, repeat=1):
    lines = [title, ', '.join(authors), 'Abstract']
    for _ in range(repeat):
        lines.extend(body_sentences)
    return lines


def jdump(obj):
    return json.dumps(obj)
