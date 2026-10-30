"""Verdict policy: the enforcement rules from the repair spec. Dependency-free."""
import unittest

from app.core.verdicts import (
    RESOLVED,
    DecisionInput,
    Verdict,
    as_verdict,
    assert_decision_invariants,
    combine_components,
    decide,
)


def full(**kw):
    base = dict(source_status='FULL_TEXT', passage_count=2, nli_status='SUCCESS')
    base.update(kw)
    return DecisionInput(**base)


class Policy(unittest.TestCase):
    def check(self, inp):
        d = decide(inp)
        assert_decision_invariants(d, inp)
        return d

    def test_supported_with_direct_passage(self):                      # case 13
        d = self.check(full(nli_label='ENTAILMENT'))
        self.assertEqual(d.verdict, Verdict.VERIFIED_SUPPORTED)
        self.assertTrue(d.semantic_verification_completed)
        self.assertFalse(d.review_required)
        self.assertEqual(d.evidence_tier, 'full_text')

    def test_topical_overlap_is_not_support(self):                     # case 16
        d = self.check(full(nli_label='NEUTRAL'))
        self.assertEqual(d.verdict, Verdict.INSUFFICIENT_EVIDENCE)
        self.assertIn('does not show the citation is wrong', d.reason)
        self.assertTrue(d.review_required)                             # unresolved => review queue

    def test_partial_support_of_compound_claim(self):                  # case 14
        d = self.check(full(nli_label='NEUTRAL', component_verdicts=('VERIFIED_SUPPORTED', 'INSUFFICIENT_EVIDENCE')))
        self.assertEqual(d.verdict, Verdict.VERIFIED_PARTIALLY_SUPPORTED)
        self.assertTrue(d.review_required)

    def test_partial_is_never_upgraded(self):
        self.assertEqual(combine_components(['VERIFIED_SUPPORTED', 'INSUFFICIENT_EVIDENCE']), Verdict.VERIFIED_PARTIALLY_SUPPORTED)
        self.assertEqual(combine_components(['SUPPORTED', 'SUPPORTED']), Verdict.VERIFIED_SUPPORTED)  # legacy names
        self.assertIsNone(combine_components(['VERIFIED_SUPPORTED', 'NLI_UNAVAILABLE']))

    def test_contradiction_requires_grounded_nli_contradiction(self):  # case 15
        self.assertEqual(self.check(full(nli_label='CONTRADICTION')).verdict, Verdict.VERIFIED_CONTRADICTED)
        d = self.check(full(nli_label='CONTRADICTION', contradiction_grounded=False))
        self.assertEqual(d.verdict, Verdict.NEEDS_HUMAN_REVIEW)
        # a failed numeric check alone is NOT a contradiction
        d = self.check(full(nli_label='NEUTRAL', numerical_status='FAILED'))
        self.assertNotEqual(d.verdict, Verdict.VERIFIED_CONTRADICTED)
        self.assertTrue(d.review_required)

    def test_metadata_only_never_supports(self):                       # case 5
        d = self.check(DecisionInput(source_status='SOURCE_METADATA_ONLY', passage_count=0))
        self.assertEqual(d.verdict, Verdict.SOURCE_METADATA_ONLY)
        self.assertIn('not indicate the citation is incorrect', d.reason)
        self.assertTrue(d.review_required)

    def test_even_with_passages_and_entailment_metadata_only_stays_unresolved(self):
        d = self.check(DecisionInput(source_status='SOURCE_METADATA_ONLY', passage_count=3, nli_status='SUCCESS', nli_label='ENTAILMENT'))
        self.assertNotIn(d.verdict, RESOLVED)

    def test_abstract_only_entailment_is_marked_and_reviewed(self):
        d = self.check(DecisionInput(source_status='ABSTRACT_ONLY', fallback_status='SOURCE_METADATA_ONLY', passage_count=1,
                                     nli_status='SUCCESS', nli_label='ENTAILMENT'))
        self.assertEqual(d.verdict, Verdict.VERIFIED_SUPPORTED)
        self.assertEqual(d.evidence_tier, 'abstract')
        self.assertIn('abstract_only_evidence', d.limitations)
        self.assertTrue(d.review_required)

    def test_abstract_only_neutral_falls_back_to_full_text_state(self):
        d = self.check(DecisionInput(source_status='ABSTRACT_ONLY', fallback_status='SOURCE_METADATA_ONLY', passage_count=1,
                                     nli_status='SUCCESS', nli_label='NEUTRAL'))
        self.assertEqual(d.verdict, Verdict.SOURCE_METADATA_ONLY)

    def test_abstract_contradiction_never_contradicts(self):
        d = self.check(DecisionInput(source_status='ABSTRACT_ONLY', fallback_status='SOURCE_METADATA_ONLY', passage_count=1,
                                     nli_status='SUCCESS', nli_label='CONTRADICTION'))
        self.assertEqual(d.verdict, Verdict.NEEDS_HUMAN_REVIEW)

    def test_source_unavailable_is_not_incorrect_citation(self):
        d = self.check(DecisionInput(source_status='SOURCE_UNAVAILABLE'))
        self.assertEqual(d.verdict, Verdict.SOURCE_UNAVAILABLE)
        self.assertNotIn(d.verdict, RESOLVED)

    def test_extraction_failure_is_a_retrieval_failure_with_its_own_code(self):  # case 8
        d = self.check(DecisionInput(source_status='EXTRACTION_FAILED'))
        self.assertEqual((d.verdict, d.reason_code), (Verdict.RETRIEVAL_FAILED, 'EXTRACTION_FAILED'))

    def test_unavailable_nli_never_verifies(self):                     # case 17
        for status in (None, 'MODEL_UNAVAILABLE', 'INFERENCE_FAILED', 'LABEL_MAPPING_UNAVAILABLE'):
            d = self.check(full(nli_status=status, nli_label='ENTAILMENT'))  # even a stale label must be ignored
            self.assertEqual(d.verdict, Verdict.NLI_UNAVAILABLE, status)
            self.assertFalse(d.semantic_verification_completed)
            self.assertTrue(d.review_required)

    def test_missing_passage(self):
        d = self.check(full(passage_count=0, nli_status=None))
        self.assertEqual(d.verdict, Verdict.INSUFFICIENT_EVIDENCE)
        self.assertFalse(d.semantic_verification_completed)

    def test_retrieval_error_is_distinct_from_no_passage(self):
        d = self.check(full(passage_count=0, nli_status=None, retrieval_error=True))
        self.assertEqual(d.verdict, Verdict.RETRIEVAL_FAILED)

    def test_transient_source_failure_status(self):
        d = self.check(DecisionInput(source_status='RETRIEVAL_FAILED', source_reason='timeout'))
        self.assertEqual(d.verdict, Verdict.RETRIEVAL_FAILED)

    def test_conflicting_passages_need_human_review(self):
        d = self.check(full(nli_label='NEUTRAL', nli_conflict=True))
        self.assertEqual(d.verdict, Verdict.NEEDS_HUMAN_REVIEW)
        self.assertTrue(d.review_required)
        self.assertIn('nli_passage_conflict', d.disagreements)

    def test_nli_vs_numeric_disagreement_is_preserved(self):
        d = self.check(full(nli_label='ENTAILMENT', numerical_status='FAILED'))
        self.assertEqual(d.verdict, Verdict.NEEDS_HUMAN_REVIEW)
        self.assertIn('nli_vs_numerical', d.disagreements)

    def test_truncated_input_cannot_be_full_support(self):
        d = self.check(full(nli_label='ENTAILMENT', nli_truncated=True))
        self.assertNotEqual(d.verdict, Verdict.VERIFIED_SUPPORTED)

    def test_ambiguous_scope_support_is_flagged_for_review(self):     # case 12
        for assoc in ('ambiguous_scope', 'multi_proposition', 'scope_ambiguous'):
            d = self.check(full(nli_label='ENTAILMENT', association=assoc))
            self.assertEqual(d.verdict, Verdict.VERIFIED_SUPPORTED)
            self.assertTrue(d.review_required)
            self.assertIn('citation_scope_ambiguous', d.review_reasons)

    def test_identity_needing_review_flags_support(self):
        d = self.check(full(nli_label='ENTAILMENT', identification_status='IDENTIFIED_NEEDS_REVIEW'))
        self.assertIn('source_identity_needs_review', d.review_reasons)

    def test_not_verifiable_claim(self):
        self.assertEqual(self.check(DecisionInput(claim_type='not_verifiable')).verdict, Verdict.NOT_VERIFIABLE)

    def test_legacy_names_are_readable(self):
        self.assertIs(as_verdict('SUPPORTED'), Verdict.VERIFIED_SUPPORTED)
        self.assertIs(as_verdict('EXTRACTION_FAILED'), Verdict.RETRIEVAL_FAILED)

    def test_exhaustive_no_resolved_verdict_without_prerequisites(self):
        """Property check over the input grid: RESOLVED implies source text + passage + successful NLI, and every
        unresolved assertion is in the review queue."""
        for src in ('FULL_TEXT', 'ABSTRACT_ONLY', 'SOURCE_METADATA_ONLY', 'SOURCE_UNAVAILABLE', 'EXTRACTION_FAILED', 'RETRIEVAL_FAILED'):
            for n in (0, 2):
                for nli in (None, 'SUCCESS', 'MODEL_UNAVAILABLE'):
                    for label in (None, 'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'):
                        for conflict in (False, True):
                            for grounded in (False, True):
                                for num in (None, 'PASSED', 'FAILED', 'NUMERICAL_CHECK_UNSUPPORTED'):
                                    inp = DecisionInput(source_status=src, passage_count=n, nli_status=nli, nli_label=label,
                                                        nli_conflict=conflict, numerical_status=num,
                                                        contradiction_grounded=grounded, fallback_status='SOURCE_METADATA_ONLY')
                                    d = decide(inp)
                                    assert_decision_invariants(d, inp)
                                    if d.verdict in RESOLVED:
                                        self.assertIn(src, ('FULL_TEXT', 'ABSTRACT_ONLY'))
                                        self.assertEqual(nli, 'SUCCESS')
                                        self.assertGreater(n, 0)
                                    if d.verdict is Verdict.VERIFIED_CONTRADICTED:
                                        self.assertEqual((src, grounded), ('FULL_TEXT', True))


if __name__ == '__main__':
    unittest.main()
