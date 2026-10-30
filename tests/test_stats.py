"""Aggregation consistency (case 16). Dependency-free."""
import unittest

from app.core.stats import PairRecord, StatsInconsistency, compute_stats, validate_pair


def pr(i, sent, verdict, ident=True, full=False, ev=0, sem=False, atom=None, review=None, tier=None, stage=None):
    resolved = verdict in ('SUPPORTED', 'PARTIALLY_SUPPORTED', 'CONTRADICTED', 'VERIFIED_SUPPORTED',
                           'VERIFIED_PARTIALLY_SUPPORTED', 'VERIFIED_CONTRADICTED', 'NOT_VERIFIABLE')
    review = (not resolved) if review is None else review   # unresolved pairs are always in the review queue
    return PairRecord(f'p{i}', sent, atom or f'{sent}.1', verdict, ident, full, ev, sem, review, tier, stage)


def sample():
    return [
        pr(1, 'S1', 'VERIFIED_SUPPORTED', full=True, ev=2, sem=True),
        pr(2, 'S1', 'INSUFFICIENT_EVIDENCE', full=True, ev=1, sem=True, atom='S1.2'),
        pr(3, 'S2', 'VERIFIED_CONTRADICTED', full=True, ev=1, sem=True, review=True),
        pr(4, 'S3', 'SOURCE_METADATA_ONLY', stage='full_text_retrieval'),
        pr(5, 'S4', 'SOURCE_UNAVAILABLE', ident=False, stage='identification'),
        pr(6, 'S5', 'NLI_UNAVAILABLE', full=True, ev=2, stage='nli'),
        pr(7, 'S6', 'RETRIEVAL_FAILED', stage='extraction'),
        pr(8, 'S7', 'NOT_VERIFIABLE', ident=False),
    ]


class Stats(unittest.TestCase):
    def test_hierarchy_and_partition(self):
        s = compute_stats(sample())
        self.assertEqual(s['claim_citation_pairs'], 8)
        self.assertEqual(s['distinct_cited_sentences'], 7)
        self.assertEqual(s['atomic_claims'], 8)  # S1 has two atomic claims
        self.assertEqual(sum(s['verdict_counts'].values()), 8)
        chain = [s['claim_citation_pairs'], s['pairs_source_identified'], s['pairs_full_text_retrieved'],
                 s['pairs_with_evidence_passages'], s['pairs_semantic_verification_completed']]
        self.assertEqual(chain, sorted(chain, reverse=True))
        self.assertEqual((s['pairs_full_text_retrieved'], s['pairs_with_evidence_passages'],
                          s['pairs_semantic_verification_completed']), (4, 4, 3))

    def test_rates_have_explicit_denominators(self):
        s = compute_stats(sample())
        self.assertEqual(s['support_rate']['numerator'], 1)
        self.assertEqual(s['support_rate']['denominator'], 3)       # semantic-completed pairs, NOT all 8
        self.assertEqual(s['support_rate']['percent'], 33.3)
        self.assertEqual(s['evidence_coverage']['denominator'], 7)  # excludes NOT_VERIFIABLE
        self.assertEqual(s['unresolved'], 8 - 2)
        self.assertEqual(s['model_failures'], 1)
        self.assertEqual(s['retrieval_failures'], 2)

    def test_support_rate_undefined_not_zero_or_hundred_when_nothing_verified(self):
        s = compute_stats([pr(1, 'S1', 'SOURCE_UNAVAILABLE', ident=False)])
        self.assertIsNone(s['support_rate']['percent'])

    def test_sentence_rollup_derived_from_pairs(self):
        r = compute_stats(sample())['sentence_rollup']
        self.assertEqual(r['PARTIALLY_SUPPORTED'], 1)   # S1: one supported, one not
        self.assertEqual(r['HAS_CONTRADICTION'], 1)
        self.assertEqual(sum(r.values()), 7)

    def test_double_counting_detected(self):
        with self.assertRaises(StatsInconsistency):
            compute_stats([pr(1, 'S1', 'VERIFIED_SUPPORTED', full=True, ev=1, sem=True),
                           pr(1, 'S1', 'VERIFIED_SUPPORTED', full=True, ev=1, sem=True)])

    def test_impossible_states_rejected(self):
        bad = [pr(1, 'S', 'VERIFIED_SUPPORTED'),                                # supported with nothing
               pr(2, 'S', 'VERIFIED_SUPPORTED', full=True, ev=1, sem=False),    # supported without semantic check
               pr(3, 'S', 'NLI_UNAVAILABLE', full=True, ev=1, sem=True),        # unavailable but "verified"
               pr(4, 'S', 'SOURCE_METADATA_ONLY', full=True, ev=1),             # metadata-only with full text
               pr(5, 'S', 'INSUFFICIENT_EVIDENCE', full=False, ev=1),           # passage without full text/abstract
               pr(6, 'S', 'INSUFFICIENT_EVIDENCE', full=True, ev=0, sem=True),  # semantic check with no passage
               pr(7, 'S', 'VERIFIED_SUPPORTED', ident=False, full=True, ev=1, sem=True),  # full text, no identity
               pr(8, 'S', 'VERIFIED_CONTRADICTED', ev=1, sem=True, tier='abstract'),      # contradiction from abstract
               pr(9, 'S', 'SOURCE_UNAVAILABLE', ident=False, review=False)]   # unresolved hidden from review queue
        for p in bad:
            with self.assertRaises(StatsInconsistency, msg=p.pair_id):
                validate_pair(p)

    def test_failure_stages_and_review_queue(self):                      # case 20
        s = compute_stats(sample())
        self.assertEqual(s['failure_stage_breakdown']['identification'], 1)
        self.assertEqual(s['failure_stage_breakdown']['full_text_retrieval'], 1)
        self.assertEqual(s['failure_stage_breakdown']['extraction'], 1)
        self.assertEqual(s['failure_stage_breakdown']['nli'], 1)
        # every unresolved assertion (6 - NOT_VERIFIABLE = 5) plus the contradiction is in the review queue
        self.assertEqual(s['review_required'], 6)

    def test_abstract_evidence_is_counted_separately(self):
        rows = [pr(1, 'S1', 'VERIFIED_SUPPORTED', ev=1, sem=True, tier='abstract', review=True),
                pr(2, 'S2', 'VERIFIED_SUPPORTED', full=True, ev=1, sem=True)]
        s = compute_stats(rows)
        self.assertEqual((s['pairs_with_abstract_evidence_only'], s['pairs_with_full_text_passages']), (1, 1))
        self.assertEqual(s['supported_from_abstract_only'], 1)
        self.assertEqual(s['support_rate']['denominator'], 1)   # abstract-tier outcomes never enter the full-text rate

    def test_multiple_references_supporting_one_claim_roll_up(self):   # case 19
        rows = [pr(1, 'S1', 'VERIFIED_SUPPORTED', full=True, ev=1, sem=True, atom='S1.1'),
                pr(2, 'S1', 'INSUFFICIENT_EVIDENCE', full=True, ev=1, sem=True, atom='S1.1')]
        self.assertEqual(compute_stats(rows)['proposition_rollup'], {'SUPPORTED_BY_AT_LEAST_ONE_CITATION': 1})

    def test_empty(self):
        s = compute_stats([])
        self.assertEqual(s['claim_citation_pairs'], 0)
        self.assertIsNone(s['support_rate']['percent'])


if __name__ == '__main__':
    unittest.main()
