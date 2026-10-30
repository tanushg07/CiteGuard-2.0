"""Atomic decomposition and proposition-level citation association (cases 10-12). Dependency-free."""
import unittest

from app.core.claims import associate, predicate_components, split_clauses
from app.services.extractor import Extractor

ROBOT = ('Robot scientists have generated hypotheses and tested them in the laboratory [1, 2], while autonomous '
         'laboratories have used experimental feedback to select and carry out further experiments [3, 4].')


def pairs(sentence):
    return Extractor().extract([{'text': sentence, 'page_number': 1}], enrich=False, total_pages=1)


class Decomposition(unittest.TestCase):
    def test_compound_sentence_with_four_references_is_two_propositions(self):   # case 11 (spec example)
        out = pairs(ROBOT)
        props = {p.atomic_claim_id: p.text for p in out}
        self.assertEqual(props, {
            'CLM-0001.1': 'Robot scientists have generated hypotheses and tested them in the laboratory.',
            'CLM-0001.2': 'Autonomous laboratories have used experimental feedback to select and carry out further experiments.'})
        mapping = {(p.atomic_claim_id, p.citation_marker): p.association_type for p in out}
        self.assertEqual(mapping, {('CLM-0001.1', '[1]'): 'direct', ('CLM-0001.1', '[2]'): 'direct',
                                   ('CLM-0001.2', '[3]'): 'direct', ('CLM-0001.2', '[4]'): 'direct'})
        self.assertTrue(all(p.sentence_text == ROBOT for p in out))           # original sentence preserved
        self.assertTrue(all(p.clause_text in ROBOT for p in out))             # verbatim slice preserved
        self.assertEqual(out[2].clause_relation, 'contrast_or_simultaneity')
        self.assertIn('own subject', out[2].decomposition_rationale)

    def test_shared_subject_predicates_stay_one_proposition(self):
        s = 'Systems based on large language models (LLMs) can also search the literature, propose ideas, plan experiments and analyse data [5–7].'
        out = pairs(s)
        self.assertEqual(len({p.atomic_claim_id for p in out}), 1)
        self.assertEqual({p.association_type for p in out}, {'sentence_level'})   # case 10: several works, one claim
        self.assertEqual(out[0].predicate_components[0], 'Systems based on large language models (LLMs) can search the literature.')
        self.assertEqual(len(out[0].predicate_components), 4)

    def test_coordinated_subjects_are_not_split(self):
        s = ('Independent verification, precommitted predictions and acceptance criteria, and experimental controls '
             'provide safeguards against unjustified conclusions [13–15].')
        self.assertEqual(len(split_clauses(s)), 1)

    def test_infinitive_coordination_is_not_a_component_list(self):
        self.assertEqual(predicate_components('Autonomous laboratories have used experimental feedback to select and carry out further experiments.'), [])

    def test_causal_and_conditional_relations_are_preserved(self):
        for s in ('The model converged faster because the learning rate was reduced after each epoch of training [2].',
                  'If the dose exceeds the threshold for several days, the patients develop severe symptoms [3].',
                  'The sensor drifted over time, and thus the calibration had to be repeated every hour [4].'):
            self.assertEqual(len(split_clauses(s)), 1, s)

    def test_punctuation_alone_never_splits(self):
        s = 'The benchmark includes translation, summarisation, question answering, and dialogue tasks from many sources [1].'
        self.assertEqual(len(split_clauses(s)), 1)

    def test_citation_scope_ambiguity_is_flagged(self):                     # case 12
        s = ('Their optically readable spins can be manipulated through precisely controlled microwave-pulse sequences, '
             'while variations in spin properties create a nontrivial problem of predicting ensemble responses [16–18].')
        out = pairs(s)
        first = [p for p in out if p.atomic_claim_id.endswith('.1')]
        second = [p for p in out if p.atomic_claim_id.endswith('.2')]
        self.assertEqual({p.association_type for p in first}, {'ambiguous_scope'})
        self.assertEqual({p.association_type for p in second}, {'multi_proposition'})
        self.assertTrue(all(p.ambiguity_status == 'ambiguous' for p in out))
        self.assertEqual(len(out), 6)   # never silently dropped: each proposition x each cited work is checked

    def test_associations_record_rationale(self):
        clauses = split_clauses(ROBOT)
        markers = [('[1]', ROBOT.index('[1')), ('[3]', ROBOT.index('[3'))]
        for a in associate(ROBOT, clauses, markers):
            self.assertTrue(a.rationale)

    def test_long_unsplit_sentence_is_flagged_uncertain(self):
        s = ' '.join(['The method'] + ['carefully'] * 70 + ['improves accuracy [1].'])
        self.assertTrue(split_clauses(s)[0].uncertain)


if __name__ == '__main__':
    unittest.main()
