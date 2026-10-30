"""Retrieval outcome classification (cases 5, 6, 9, 15). Dependency-free."""
import unittest
from datetime import UTC, datetime, timedelta

from app.services.fetch_policy import (
    Attempt,
    backoff_delay,
    classify_http,
    classify_identifier,
    parse_retry_after,
    provider_for,
    summarize_attempts,
)


def att(outcome, provider='publisher_or_repository:x'):
    return Attempt(provider, 'https://x', outcome).as_dict()


class Http(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify_http(200), ('FETCHED', False))
        self.assertEqual(classify_http(403), ('ACCESS_DENIED', False))   # never retried or bypassed
        self.assertEqual(classify_http(401), ('ACCESS_DENIED', False))
        self.assertEqual(classify_http(404), ('NOT_FOUND', False))
        self.assertEqual(classify_http(429), ('RATE_LIMITED', True))
        self.assertEqual(classify_http(503), ('TRANSIENT_ERROR', True))

    def test_retry_after(self):
        self.assertEqual(parse_retry_after('5', 20), 5.0)
        self.assertEqual(parse_retry_after('999', 20), 20.0)             # capped
        self.assertIsNone(parse_retry_after('soon', 20))
        now = datetime(2026, 1, 1, tzinfo=UTC)
        when = (now + timedelta(seconds=7)).strftime('%a, %d %b %Y %H:%M:%S GMT')
        self.assertEqual(parse_retry_after(when, 20, now), 7.0)

    def test_backoff_is_bounded_and_server_hint_wins(self):
        self.assertEqual([backoff_delay(n, 1.0) for n in range(3)], [1.0, 2.0, 4.0])
        self.assertEqual(backoff_delay(10, 1.0, cap=20), 20)
        self.assertEqual(backoff_delay(0, 1.0, retry_after=3.0), 3.0)

    def test_provider_names(self):
        self.assertEqual(provider_for('https://arxiv.org/pdf/1706.03762'), 'arxiv')
        self.assertEqual(provider_for('https://doi.org/10.1/x'), 'doi_resolver')


class Identifiers(unittest.TestCase):
    def test_malformed_doi(self):                                       # case 6
        self.assertEqual(classify_identifier(doi='10.12/abc')[0], 'MALFORMED')
        self.assertEqual(classify_identifier(doi='not a doi')[0], 'MALFORMED')
        self.assertEqual(classify_identifier(doi='doi:10.1162/neco.1997.9.8.1735')[0], 'MALFORMED')  # prefix must be stripped upstream
    def test_valid_doi(self):
        self.assertEqual(classify_identifier(doi='10.1162/neco.1997.9.8.1735')[0], 'VALID')
        self.assertEqual(classify_identifier(doi='https://doi.org/10.1162/neco.1997.9.8.1735')[0], 'VALID')
    def test_arxiv_and_absent(self):
        self.assertEqual(classify_identifier(arxiv_id='1706.03762v5')[0], 'VALID')
        self.assertEqual(classify_identifier(arxiv_id='hello')[0], 'MALFORMED')
        self.assertEqual(classify_identifier()[0], 'ABSENT')


class Summaries(unittest.TestCase):
    def test_doi_resolves_but_fulltext_inaccessible(self):              # case 5
        status, reason = summarize_attempts([att('ACCESS_DENIED'), att('NOT_A_PDF')], source_identified=True)
        self.assertEqual(status, 'SOURCE_METADATA_ONLY')
        self.assertIn('Metadata is not evidence', reason)

    def test_timeout_and_rate_limit_are_retrieval_failures_not_missing_source(self):   # case 9
        for o in ('TIMEOUT', 'RATE_LIMITED', 'NETWORK_ERROR', 'TRANSIENT_ERROR'):
            status, reason = summarize_attempts([att(o)], source_identified=True)
            self.assertEqual(status, 'RETRIEVAL_FAILED', o)
            self.assertIn('Support was not assessed', reason)

    def test_definitive_negative_beats_transient(self):
        self.assertEqual(summarize_attempts([att('TIMEOUT'), att('ACCESS_DENIED')], True)[0], 'SOURCE_METADATA_ONLY')

    def test_unidentified_source(self):
        self.assertEqual(summarize_attempts([], source_identified=False)[0], 'SOURCE_UNAVAILABLE')

    def test_wrong_paper_is_a_definitive_negative_not_success(self):
        # a PDF was FETCHED but belonged to a different publication: it must not count as full text
        status, _ = summarize_attempts([att('FETCHED'), att('IDENTITY_MISMATCH')], True)
        self.assertEqual(status, 'SOURCE_METADATA_ONLY')

    def test_attempt_records_have_provenance(self):
        a = Attempt('arxiv', 'https://arxiv.org/pdf/1', 'TIMEOUT', 'slow').as_dict()
        for k in ('provider', 'url', 'outcome', 'reason', 'timestamp'):
            self.assertTrue(a[k])


if __name__ == '__main__':
    unittest.main()
