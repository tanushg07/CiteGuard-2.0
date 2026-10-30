"""LIVE integration tests against real scholarly providers. Opt-in: CITEGUARD_LIVE_TESTS=1.

They depend on third-party availability and may legitimately fail when a provider is down or rate-limits;
unit tests with mocked providers (tests/test_identification.py, tests/test_fulltext_pipeline.py) are the
deterministic regression suite. These use well-known, long-lived public records only.
"""
import asyncio

import pytest

from app.services.identification import IDENTIFIED, ReferenceIdentifier
from app.services.source_resolver import SourceResolver

pytestmark = pytest.mark.live

WITH_DOI = ('[1] Ross D. King, Kenneth E. Whelan, Ffion M. Jones, Philip G. K. Reiser, Christopher H. Bryant, Stephen H. '
            'Muggleton, Douglas B. Kell, and Stephen G. Oliver. Functional genomic hypothesis generation and '
            'experimentation by a robot scientist. Nature, 427(6971):247–252, 2004. doi: 10.1038/nature02236.')
WITHOUT_DOI = ('[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, '
               'and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.')
ARXIV = ('[3] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. '
         'Large language models cannot self-correct reasoning yet. In ICLR 2024. arXiv:2310.01798.')


def identify(raw):
    async def run():
        async with ReferenceIdentifier(cache_dir=None) as ri:
            return await ri.identify(raw)
    return asyncio.run(run())


def test_live_doi_reference():
    result = identify(WITH_DOI)
    assert result.status in IDENTIFIED and result.best['doi'] == '10.1038/nature02236'


def test_live_reference_without_doi():
    result = identify(WITHOUT_DOI)
    assert result.status in IDENTIFIED, result.reason
    assert 'attention is all you need' in result.best['title'].lower()


def test_live_arxiv_full_text():
    async def run():
        async with SourceResolver(cache_dir=None) as resolver:
            return await resolver.resolve(ARXIV)
    record, _ = asyncio.run(run())
    assert record['status'] == 'FULL_TEXT', record['reason']
    assert record['full_text']['identity']['valid'] and record['passages']
