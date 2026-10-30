"""Atomic claims, citation scope, page mapping (cases 10, 11, 14). Dependency-free."""
import unittest

from app.core.claims import CITATION, associate, split_clauses, validate_page


def run(sentence):
    clauses = split_clauses(sentence)
    markers = {}
    for m in CITATION.finditer(sentence):
        markers.setdefault(m.group(), m.start())
    return clauses, associate(sentence, clauses, list(markers.items()))


class Atomic(unittest.TestCase):
    def test_one_citation_two_claims_is_ambiguous_not_assumed(self):    # case 11
        s = 'BLEU improved on the English-German task over prior systems [8]; the decoder attends to encoder outputs via cross-attention.'
        clauses, assoc = run(s)
        self.assertEqual(len(clauses), 2)
        kinds = {(a.clause_index, a.association) for a in assoc}
        self.assertEqual(kinds, {(0, 'direct'), (1, 'ambiguous_scope')})
        self.assertTrue(all('[8]' == a.marker for a in assoc))

    def test_multiple_citations_each_bound_to_its_clause(self):         # case 10
        s = 'The Transformer uses multi-head self-attention in the encoder [1], and the decoder attends to encoder outputs through cross-attention [2].'
        clauses, assoc = run(s)
        self.assertEqual([(a.clause_index, a.marker, a.association) for a in assoc],
                         [(0, '[1]', 'direct'), (1, '[2]', 'direct')])

    def test_multiple_citations_one_proposition_stay_together(self):
        clauses, assoc = run('Multiple studies show this effect on translation quality across datasets [1] and [2].')
        self.assertEqual(len(clauses), 1)
        self.assertEqual({a.marker for a in assoc}, {'[1]', '[2]'})

    def test_clause_text_is_verbatim_substring(self):
        s = 'The Transformer uses multi-head self-attention in the encoder [1], and the decoder attends to encoder outputs through cross-attention [2].'
        for c in split_clauses(s):
            self.assertIn(c.text, s)

    def test_qualifiers_prevent_splitting(self):
        s = 'Although recurrent networks are slow to train, they remain widely used in sequence modeling tasks [4], and attention is now common.'
        self.assertEqual(len(split_clauses(s)), 1)
        self.assertEqual(len(split_clauses('Models A and B scored 90 and 85 on the respective test sets reported in the paper [1], and the baseline scored lower overall.'.replace('respective', 'respectively'))), 1)

    def test_anaphora_prevents_orphaned_pronoun(self):
        s = 'Recurrent models factor computation along symbol positions [3]; this precludes parallelization within training examples.'
        self.assertEqual(len(split_clauses(s)), 1)

    def test_shared_subject_not_split(self):
        s = 'The authors trained the model on large corpora [2], and improving results on several benchmark tasks consistently.'
        self.assertEqual(len(split_clauses(s)), 1)

    def test_short_fragments_not_split(self):
        self.assertEqual(len(split_clauses('It works [1], and it is fast.')), 1)


class Pages(unittest.TestCase):                                         # case 14
    def test_missing_page_is_not_page_one(self):
        self.assertEqual(validate_page(None, 15), (None, 'missing'))
    def test_out_of_range(self):
        self.assertEqual(validate_page(0, 15), (None, 'out_of_range'))
        self.assertEqual(validate_page(16, 15), (None, 'out_of_range'))
    def test_invalid(self):
        self.assertEqual(validate_page('x', 15), (None, 'invalid'))
        self.assertEqual(validate_page(True, 15), (None, 'missing'))
    def test_valid_and_unknown_total(self):
        self.assertEqual(validate_page(3, 15), (3, 'ok'))
        self.assertEqual(validate_page(40, None), (40, 'ok'))


if __name__ == '__main__':
    unittest.main()
