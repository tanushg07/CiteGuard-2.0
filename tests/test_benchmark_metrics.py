"""Metric arithmetic on hand-built predictions. Dependency-free. Tests the METRIC CODE, not model accuracy."""
import unittest

from app.evaluation.metrics import compute


def row(gold, pred, sem=True): return {'gold': gold, 'predicted': pred, 'semantic_completed': sem}


class Metrics(unittest.TestCase):
    def test_known_values(self):
        rows = [row('SUPPORTED', 'SUPPORTED'), row('SUPPORTED', 'INSUFFICIENT_EVIDENCE'),
                row('INSUFFICIENT', 'SUPPORTED'), row('CONTRADICTED', 'CONTRADICTED'),
                row('UNRESOLVABLE', 'SOURCE_UNAVAILABLE', sem=False)]
        m = compute(rows)
        self.assertEqual(m['support_precision'], 0.5)           # 1 of 2 predicted-supported
        self.assertEqual(m['support_recall'], 0.5)              # 1 of 2 gold-supported
        self.assertEqual(m['false_positive_support_rate'], round(1 / 3, 4))   # 1 of 3 non-supported gold
        self.assertEqual(m['contradiction_precision'], 1.0)
        self.assertEqual(m['unresolved_rate'], 0.4)
        self.assertEqual(m['evidence_coverage'], 0.8)

    def test_abstaining_system_has_no_precision_and_zero_fp(self):
        rows = [row('SUPPORTED', 'NLI_UNAVAILABLE', False), row('INSUFFICIENT', 'NLI_UNAVAILABLE', False)]
        m = compute(rows)
        self.assertIsNone(m['support_precision'])               # undefined, not 0 or 1
        self.assertEqual((m['support_recall'], m['false_positive_support_rate'], m['unresolved_rate']), (0.0, 0.0, 1.0))

    def test_empty(self):
        self.assertEqual(compute([])['n'], 0)


if __name__ == '__main__':
    unittest.main()
