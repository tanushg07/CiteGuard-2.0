"""Canonical statistics, computed only from per-pair and per-reference records.

Units are explicit. Every metric is one of:
  pairs      - claim-citation pairs (one atomic proposition x one cited reference)
  claims     - atomic propositions;  sentences - distinct cited manuscript sentences
  references - bibliography entries; attempts - individual provider / HTTP attempts

Pair counting model (hierarchical, each row a subset of the one above):
    pairs >= source_identified >= full_text >= with_full_text_passages >= full_text_semantic_completed
Abstract-tier evidence (provider abstract of an identified source when full text is unavailable) is counted in
separate rows and is never merged into full-text coverage.

Verdict buckets ARE mutually exclusive: every pair has exactly one verdict, so sum(verdict_counts) == pairs.
Failure-stage buckets are also mutually exclusive (one failure stage per unresolved pair, or none).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from app.core.verdicts import RESOLVED, SOURCE_SIDE_FAILURES, Verdict, as_verdict


class StatsInconsistency(AssertionError):
    """Raised when pair records could not have come from a valid pipeline run."""


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    sentence_id: str
    atomic_claim_id: str
    verdict: str
    source_identified: bool
    full_text: bool
    evidence_count: int
    semantic_completed: bool
    review_required: bool = False
    evidence_tier: str | None = None     # full_text | abstract | None
    failure_stage: str | None = None
    reference_id: str | None = None

    @property
    def tier(self):
        return self.evidence_tier or ('full_text' if self.full_text and self.evidence_count else None)


def validate_pair(p: PairRecord) -> None:
    v = as_verdict(p.verdict)
    if p.full_text and not p.source_identified:
        raise StatsInconsistency(f'{p.pair_id}: full text without an identified source')
    if p.evidence_count and not p.full_text and p.tier != 'abstract':
        raise StatsInconsistency(f'{p.pair_id}: evidence passages without retrieved full text or an abstract')
    if p.tier == 'abstract' and not p.source_identified:
        raise StatsInconsistency(f'{p.pair_id}: abstract evidence without an identified source')
    if p.semantic_completed and not p.evidence_count:
        raise StatsInconsistency(f'{p.pair_id}: semantic verification completed with no passage')
    if v in RESOLVED and not (p.evidence_count and p.semantic_completed and (p.full_text or p.tier == 'abstract')):
        raise StatsInconsistency(f'{p.pair_id}: {v} without source text, passage and semantic verification')
    if v is Verdict.VERIFIED_CONTRADICTED and p.tier != 'full_text':
        raise StatsInconsistency(f'{p.pair_id}: contradiction without full-text evidence')
    if v is Verdict.NLI_UNAVAILABLE and p.semantic_completed:
        raise StatsInconsistency(f'{p.pair_id}: NLI_UNAVAILABLE but marked semantically verified')
    if v in (Verdict.SOURCE_METADATA_ONLY, Verdict.SOURCE_UNAVAILABLE) and p.full_text:
        raise StatsInconsistency(f'{p.pair_id}: {v} but full text is marked retrieved')
    if v not in RESOLVED and v is not Verdict.NOT_VERIFIABLE and not p.review_required:
        raise StatsInconsistency(f'{p.pair_id}: unresolved {v} is hidden from the review queue')


def _rate(n: int, d: int):
    return round(100 * n / d, 1) if d else None


def sentence_rollup(group: list[PairRecord]) -> str:
    vs = {as_verdict(p.verdict) for p in group}
    if Verdict.VERIFIED_CONTRADICTED in vs:
        return 'HAS_CONTRADICTION'
    if all(as_verdict(p.verdict) is Verdict.VERIFIED_SUPPORTED for p in group):
        return 'ALL_SUPPORTED'
    if vs & {Verdict.VERIFIED_SUPPORTED, Verdict.VERIFIED_PARTIALLY_SUPPORTED}:
        return 'PARTIALLY_SUPPORTED'
    return 'UNRESOLVED'


def proposition_rollup(group: list[PairRecord]) -> str:
    """One atomic proposition cited by several works: supported if ANY cited work supports it."""
    vs = {as_verdict(p.verdict) for p in group}
    if Verdict.VERIFIED_SUPPORTED in vs:
        return 'SUPPORTED_BY_AT_LEAST_ONE_CITATION'
    if Verdict.VERIFIED_CONTRADICTED in vs:
        return 'HAS_CONTRADICTION'
    if Verdict.VERIFIED_PARTIALLY_SUPPORTED in vs:
        return 'PARTIALLY_SUPPORTED'
    return 'UNRESOLVED'


METRIC_DEFINITIONS = {
    'distinct_cited_sentences': 'sentences: manuscript sentences containing at least one recognised citation marker',
    'atomic_claims': 'claims: atomic propositions after decomposition',
    'claim_citation_pairs': 'pairs: atomic proposition x cited reference',
    'pairs_source_identified': 'pairs whose reference was identified (high confidence or needs review)',
    'pairs_full_text_retrieved': 'pairs whose identified source full text was retrieved and validated',
    'pairs_with_evidence_passages': 'pairs with >=1 examined passage of any tier (full text or abstract)',
    'pairs_with_full_text_passages': 'pairs with >=1 examined full-text passage',
    'pairs_with_abstract_evidence_only': 'pairs whose only examined passage is a provider abstract',
    'pairs_semantic_verification_completed': 'pairs where NLI inference completed on >=1 passage (any tier)',
    'pairs_full_text_semantic_completed': 'pairs where NLI inference completed on full-text passages',
    'review_required': 'pairs in the human-review queue (all unresolved pairs plus flagged verified pairs)',
    'support_rate': 'pairs: VERIFIED_SUPPORTED (full text) / full-text pairs with a conclusive semantic outcome',
    'evidence_coverage': 'pairs: pairs with >=1 examined passage / pairs that are assertions',
}


def compute_stats(pairs: list[PairRecord], validate: bool = True) -> dict:
    if validate:
        for p in pairs:
            validate_pair(p)
        ids = [p.pair_id for p in pairs]
        # one pair per (atomic claim, citation); a repeated pair id means double counting
        if len(set(ids)) != len(ids):
            dupes = [k for k, n in Counter(ids).items() if n > 1][:3]
            raise StatsInconsistency(f'duplicate pair ids (double counting): {dupes}')

    n = len(pairs)
    counts = Counter(as_verdict(p.verdict).value for p in pairs)
    identified = sum(p.source_identified for p in pairs)
    full_text = sum(p.full_text for p in pairs)
    ft_evidence = sum(p.evidence_count > 0 and p.tier == 'full_text' for p in pairs)
    abstract_evidence = sum(p.evidence_count > 0 and p.tier == 'abstract' for p in pairs)
    any_evidence = sum(p.evidence_count > 0 for p in pairs)
    semantic = sum(p.semantic_completed for p in pairs)
    ft_semantic = sum(p.semantic_completed and p.tier == 'full_text' for p in pairs)
    if validate and not (n >= identified >= full_text >= ft_evidence >= ft_semantic):
        raise StatsInconsistency('coverage chain is not monotone: '
                                 f'{n}/{identified}/{full_text}/{ft_evidence}/{ft_semantic}')
    if validate and sum(counts.values()) != n:
        raise StatsInconsistency('verdict buckets do not partition the pairs')

    v = Verdict
    supported = counts[v.VERIFIED_SUPPORTED]
    partial = counts[v.VERIFIED_PARTIALLY_SUPPORTED]
    contradicted = counts[v.VERIFIED_CONTRADICTED]
    resolved = supported + partial + contradicted
    not_verifiable = counts[v.NOT_VERIFIABLE]
    verifiable = n - not_verifiable
    unresolved = n - resolved
    retrieval_failures = sum(counts[x] for x in SOURCE_SIDE_FAILURES)
    conclusive_ft = [p for p in pairs if p.tier == 'full_text' and p.semantic_completed
                     and as_verdict(p.verdict) in RESOLVED | {v.INSUFFICIENT_EVIDENCE}]
    supported_ft = sum(as_verdict(p.verdict) is v.VERIFIED_SUPPORTED for p in conclusive_ft)
    sentences, propositions = defaultdict(list), defaultdict(list)
    for p in pairs:
        sentences[p.sentence_id].append(p)
        propositions[p.atomic_claim_id].append(p)
    stage = Counter(p.failure_stage for p in pairs if p.failure_stage)

    return {
        'distinct_cited_sentences': len(sentences),
        'atomic_claims': len(propositions),
        'claim_citation_pairs': n,
        'pairs_source_identified': identified,
        'pairs_full_text_retrieved': full_text,
        'pairs_with_evidence_passages': any_evidence,
        'pairs_with_full_text_passages': ft_evidence,
        'pairs_with_abstract_evidence_only': abstract_evidence,
        'pairs_semantic_verification_completed': semantic,
        'pairs_full_text_semantic_completed': ft_semantic,
        'supported': supported, 'partially_supported': partial, 'contradicted': contradicted,
        'supported_from_abstract_only': sum(as_verdict(p.verdict) is v.VERIFIED_SUPPORTED and p.tier == 'abstract' for p in pairs),
        'needs_human_review_verdicts': counts[v.NEEDS_HUMAN_REVIEW],
        'insufficient_evidence': counts[v.INSUFFICIENT_EVIDENCE],
        'unresolved': unresolved,
        'not_verifiable': not_verifiable,
        'retrieval_failures': retrieval_failures,
        'retrieval_failure_breakdown': {x.value: counts[x] for x in sorted(SOURCE_SIDE_FAILURES) if counts[x]},
        'source_metadata_only': counts[v.SOURCE_METADATA_ONLY],
        'model_failures': counts[v.NLI_UNAVAILABLE],
        # mutually exclusive: the stage at which an unresolved pair stopped
        'failure_stage_breakdown': {
            'reference_matching': stage['reference_matching'], 'identification': stage['identification'],
            'full_text_retrieval': stage['full_text_retrieval'], 'extraction': stage['extraction'],
            'evidence_retrieval': stage['evidence_retrieval'], 'nli': stage['nli']},
        'semantic_outcomes': {'entailed': supported + partial, 'contradicted': contradicted,
                              'not_entailed': counts[v.INSUFFICIENT_EVIDENCE],
                              'needs_human_review': counts[v.NEEDS_HUMAN_REVIEW]},
        'review_required': sum(p.review_required for p in pairs),
        'verdict_counts': dict(counts),
        'sentence_rollup': dict(Counter(sentence_rollup(g) for g in sentences.values())),
        'proposition_rollup': dict(Counter(proposition_rollup(g) for g in propositions.values())),
        # Denominators are stated, never implicit.
        'support_rate': {
            'numerator': supported_ft, 'denominator': len(conclusive_ft),
            'denominator_definition': 'full-text pairs whose NLI check completed with a conclusive outcome '
                                      '(verified supported/partial/contradicted, or not entailed)',
            'percent': _rate(supported_ft, len(conclusive_ft))},
        'evidence_coverage': {
            'numerator': any_evidence, 'denominator': verifiable,
            'denominator_definition': 'pairs that are assertions (excludes NOT_VERIFIABLE)',
            'percent': _rate(any_evidence, verifiable)},
        'full_text_evidence_coverage': {'numerator': ft_evidence, 'denominator': verifiable,
                                        'percent': _rate(ft_evidence, verifiable)},
        'unresolved_rate': {'numerator': unresolved - not_verifiable, 'denominator': verifiable,
                            'percent': _rate(unresolved - not_verifiable, verifiable)},
        'definitions': METRIC_DEFINITIONS,
        'confidence_note': 'Model scores are uncalibrated; they are not probabilities that a citation is correct.',
    }


def reference_stats(records: list[dict], total_references: int) -> dict:
    """Per-reference (unique source) metrics from resolver records. Unit: references (bibliography entries)."""
    ident = Counter((r.get('identification') or {}).get('status', 'NOT_ATTEMPTED') for r in records)
    full = Counter((r.get('full_text') or {}).get('status', 'NOT_ATTEMPTED') for r in records)
    identified = ident['IDENTIFIED_HIGH_CONFIDENCE'] + ident['IDENTIFIED_NEEDS_REVIEW']
    retrieved = sum(r.get('status') == 'FULL_TEXT' for r in records)
    usable = sum(r.get('status') == 'FULL_TEXT' and len(r.get('passages') or []) > 0 for r in records)
    attempts = Counter(a.get('outcome') for r in records for a in r.get('attempts', []))
    return {
        'total_references': total_references,
        'references_resolved': len(records),
        'references_not_cited_or_not_matched': max(0, total_references - len(records)),
        'identified': identified,
        'identified_high_confidence': ident['IDENTIFIED_HIGH_CONFIDENCE'],
        'identified_needs_review': ident['IDENTIFIED_NEEDS_REVIEW'],
        'unidentified': len(records) - identified,
        'identification_breakdown': dict(ident),
        'full_text_retrieved': retrieved,
        'usable_extracted_text': usable,
        'abstract_only': sum(r.get('status') == 'ABSTRACT_ONLY' for r in records),
        'full_text_breakdown': dict(full),
        'extraction_failures': full['EXTRACTION_FAILED'],
        'duplicate_documents': sum(bool((r.get('full_text') or {}).get('duplicate_of')) for r in records),
        'attempts_total': sum(attempts.values()),
        'attempt_outcomes': dict(attempts),
        'definitions': {'unit': 'references (unique bibliography entries cited in the manuscript); attempts are individual requests',
                        'identified': 'IDENTIFIED_HIGH_CONFIDENCE + IDENTIFIED_NEEDS_REVIEW',
                        'usable_extracted_text': 'full text retrieved, identity-validated and extracted to >=1 passage'},
    }
