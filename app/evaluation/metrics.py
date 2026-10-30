"""Benchmark metrics. Pure Python.

Gold labels: SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | INSUFFICIENT (source examined, claim not established)
| UNRESOLVABLE (no usable source text exists for the case).
A prediction is a final_verdict string from the pipeline plus whether semantic verification completed.
Legacy verdict names (SUPPORTED, CONTRADICTED, PARTIALLY_SUPPORTED) are accepted as predictions.
"""
from __future__ import annotations

NORMALISE = {'SUPPORTED': 'VERIFIED_SUPPORTED', 'PARTIALLY_SUPPORTED': 'VERIFIED_PARTIALLY_SUPPORTED',
             'CONTRADICTED': 'VERIFIED_CONTRADICTED'}
SUPPORT_VERDICTS = {'VERIFIED_SUPPORTED'}
RESOLVED = {'VERIFIED_SUPPORTED', 'VERIFIED_PARTIALLY_SUPPORTED', 'VERIFIED_CONTRADICTED'}
GOLD_TO_VERDICT = {'SUPPORTED': 'VERIFIED_SUPPORTED', 'PARTIALLY_SUPPORTED': 'VERIFIED_PARTIALLY_SUPPORTED',
                   'CONTRADICTED': 'VERIFIED_CONTRADICTED'}


def _ratio(n, d):
    return round(n / d, 4) if d else None


def _pred(c):
    return NORMALISE.get(c['predicted'], c['predicted'])


def compute(cases: list[dict]) -> dict:
    """cases: [{'gold': str, 'predicted': str, 'semantic_completed': bool}]"""
    n = len(cases)
    gold_sup = [c for c in cases if c['gold'] == 'SUPPORTED']
    pred_sup = [c for c in cases if _pred(c) in SUPPORT_VERDICTS]
    tp = sum(c['gold'] == 'SUPPORTED' for c in pred_sup)
    non_sup = [c for c in cases if c['gold'] != 'SUPPORTED']
    fp = sum(_pred(c) in SUPPORT_VERDICTS for c in non_sup)
    pred_con = [c for c in cases if _pred(c) == 'VERIFIED_CONTRADICTED']
    tp_con = sum(c['gold'] == 'CONTRADICTED' for c in pred_con)
    gold_con = [c for c in cases if c['gold'] == 'CONTRADICTED']
    unresolved = [c for c in cases if _pred(c) not in RESOLVED]
    return {
        'n': n,
        'support_precision': _ratio(tp, len(pred_sup)),
        'support_recall': _ratio(tp, len(gold_sup)),
        'contradiction_precision': _ratio(tp_con, len(pred_con)),
        'contradiction_recall': _ratio(tp_con, len(gold_con)),
        'false_positive_support_rate': _ratio(fp, len(non_sup)),   # support asserted where gold says not supported
        'unresolved_rate': _ratio(len(unresolved), n),
        'evidence_coverage': _ratio(sum(bool(c['semantic_completed']) for c in cases), n),
        'exact_verdict_agreement': _ratio(sum(_agree(c) for c in cases), n),
        'counts': {'gold_supported': len(gold_sup), 'predicted_supported': len(pred_sup),
                   'false_positive_support': fp, 'unresolved': len(unresolved)},
    }


def _agree(c) -> bool:
    gold, pred = c['gold'], _pred(c)
    if gold in GOLD_TO_VERDICT:
        return GOLD_TO_VERDICT[gold] == pred
    return pred not in RESOLVED   # INSUFFICIENT / UNRESOLVABLE: any non-resolved verdict is acceptable
