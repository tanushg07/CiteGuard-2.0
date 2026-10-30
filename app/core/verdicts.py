"""Single source of truth for claim-citation verdicts.

Nothing else in the pipeline may assign a verdict. Every rule below exists so that *absence* of evidence or of a
working model can never be rendered as support, and so that a missing source is never rendered as a wrong citation.

This module is deliberately free of third-party imports so that the policy can be unit-tested without torch,
transformers or pydantic installed.

Verdict vocabulary (v2.1)
  VERIFIED_SUPPORTED            a retrieved source passage entails the proposition; no numerical check conflicts
  VERIFIED_PARTIALLY_SUPPORTED  independently checked components: some entailed, the rest unresolved
  VERIFIED_CONTRADICTED         a RELEVANT retrieved passage contradicts the proposition (always human-checked)
  INSUFFICIENT_EVIDENCE         source text was examined; it does not establish the proposition
  SOURCE_METADATA_ONLY          source identified; full text not retrievable (abstract, if any, was insufficient)
  SOURCE_UNAVAILABLE            source could not be identified
  RETRIEVAL_FAILED              identification/full-text/extraction/passage retrieval did not complete
  NLI_UNAVAILABLE               passages exist but the semantic check did not run
  NOT_VERIFIABLE                navigation, acknowledgement, flattened table...
  NEEDS_HUMAN_REVIEW            model signals conflict or cannot be trusted for this input; a person must decide

Every pair that is not VERIFIED_* or NOT_VERIFIABLE is placed in the review queue (review_required=True) with
explicit review_reasons - unresolved cases are never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    VERIFIED_SUPPORTED = 'VERIFIED_SUPPORTED'
    VERIFIED_PARTIALLY_SUPPORTED = 'VERIFIED_PARTIALLY_SUPPORTED'
    VERIFIED_CONTRADICTED = 'VERIFIED_CONTRADICTED'
    INSUFFICIENT_EVIDENCE = 'INSUFFICIENT_EVIDENCE'
    SOURCE_METADATA_ONLY = 'SOURCE_METADATA_ONLY'
    SOURCE_UNAVAILABLE = 'SOURCE_UNAVAILABLE'
    RETRIEVAL_FAILED = 'RETRIEVAL_FAILED'
    NLI_UNAVAILABLE = 'NLI_UNAVAILABLE'
    NOT_VERIFIABLE = 'NOT_VERIFIABLE'
    NEEDS_HUMAN_REVIEW = 'NEEDS_HUMAN_REVIEW'


# Pre-v2.1 names, accepted when reading stored reports or legacy callers.
LEGACY_NAMES = {'SUPPORTED': Verdict.VERIFIED_SUPPORTED, 'PARTIALLY_SUPPORTED': Verdict.VERIFIED_PARTIALLY_SUPPORTED,
                'CONTRADICTED': Verdict.VERIFIED_CONTRADICTED, 'EXTRACTION_FAILED': Verdict.RETRIEVAL_FAILED}


def as_verdict(value: str) -> Verdict:
    return LEGACY_NAMES.get(value) or Verdict(value)


RESOLVED = frozenset({Verdict.VERIFIED_SUPPORTED, Verdict.VERIFIED_PARTIALLY_SUPPORTED, Verdict.VERIFIED_CONTRADICTED})
SOURCE_SIDE_FAILURES = frozenset({Verdict.SOURCE_UNAVAILABLE, Verdict.RETRIEVAL_FAILED})
SOURCE_STATUSES = {'SOURCE_METADATA_ONLY': Verdict.SOURCE_METADATA_ONLY,
                   'SOURCE_UNAVAILABLE': Verdict.SOURCE_UNAVAILABLE,
                   'EXTRACTION_FAILED': Verdict.RETRIEVAL_FAILED,
                   'RETRIEVAL_FAILED': Verdict.RETRIEVAL_FAILED}

UNCALIBRATED = 'uncalibrated_model_score'  # confidence_kind: never a probability the citation is correct
AMBIGUOUS_ASSOCIATIONS = frozenset({'ambiguous_scope', 'multi_proposition', 'uncertain', 'scope_ambiguous'})


@dataclass(frozen=True)
class DecisionInput:
    claim_type: str = 'assertion_candidate'
    source_status: str = 'SOURCE_UNAVAILABLE'   # FULL_TEXT | ABSTRACT_ONLY | one of SOURCE_STATUSES
    source_reason: str = ''
    fallback_status: str = ''                   # for ABSTRACT_ONLY: the full-text status behind the abstract
    passage_count: int = 0                      # usable retrieved passages for this pair
    retrieval_error: bool = False
    nli_status: str | None = None               # 'SUCCESS' or a failure code
    nli_label: str | None = None                # ENTAILMENT / CONTRADICTION / NEUTRAL (aggregate)
    nli_truncated: bool = False
    nli_conflict: bool = False                  # different passages gave entail AND contradict
    contradiction_grounded: bool = True         # the contradicting passage is relevant to the proposition
    numerical_status: str | None = None
    association: str = 'direct'
    identification_status: str | None = None    # IDENTIFIED_NEEDS_REVIEW adds a review reason
    decomposition_uncertain: bool = False
    component_verdicts: tuple[str, ...] = ()    # verdicts of independently checked components, if any


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason_code: str
    reason: str
    review_required: bool = False
    semantic_verification_completed: bool = False
    limitations: tuple[str, ...] = ()
    disagreements: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    evidence_tier: str | None = None


def combine_components(verdicts) -> Verdict | None:
    """Roll component verdicts of one proposition up. None = no rollup possible."""
    vs = [as_verdict(v) for v in verdicts]
    if len(vs) < 2:
        return None
    if all(v is Verdict.VERIFIED_SUPPORTED for v in vs):
        return Verdict.VERIFIED_SUPPORTED
    if Verdict.VERIFIED_CONTRADICTED in vs and Verdict.VERIFIED_SUPPORTED in vs:
        return None  # mixed: needs human review, must not be hidden inside "partial"
    if Verdict.VERIFIED_CONTRADICTED in vs:
        return Verdict.VERIFIED_CONTRADICTED
    if Verdict.VERIFIED_SUPPORTED in vs and all(v in (Verdict.VERIFIED_SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE) for v in vs):
        return Verdict.VERIFIED_PARTIALLY_SUPPORTED
    return None


def _context_reviews(i: DecisionInput) -> tuple[str, ...]:
    out = []
    if i.association in AMBIGUOUS_ASSOCIATIONS:
        out.append('citation_scope_ambiguous')
    if i.identification_status == 'IDENTIFIED_NEEDS_REVIEW':
        out.append('source_identity_needs_review')
    if i.decomposition_uncertain:
        out.append('decomposition_uncertain')
    return tuple(out)


def decide(i: DecisionInput) -> Decision:
    ctx = _context_reviews(i)

    # 1. Not an assertion we can safely test.
    if i.claim_type == 'not_verifiable':
        return Decision(Verdict.NOT_VERIFIABLE, 'NOT_ASSERTION',
                        'This citation-bearing text is navigational, an acknowledgement, or a flattened table '
                        'without a safely isolated assertion; inspect the original PDF.')

    # 2. No examinable source text => no claim-support verdict of any kind. Never "incorrect citation".
    abstract = i.source_status == 'ABSTRACT_ONLY' and i.passage_count > 0
    if i.source_status != 'FULL_TEXT' and not abstract:
        status = i.fallback_status or i.source_status
        verdict = SOURCE_STATUSES.get(status, Verdict.SOURCE_UNAVAILABLE)
        code = 'EXTRACTION_FAILED' if status == 'EXTRACTION_FAILED' else 'NO_SOURCE_TEXT'
        reason = i.source_reason or {
            Verdict.SOURCE_METADATA_ONLY: 'Source identified, but full text could not be retrieved. Metadata is not evidence.',
        }.get(verdict, 'No readable source text is available.')
        return Decision(verdict, code, reason + ' No source passage was examined; this does not indicate the citation is incorrect.',
                        review_required=True, review_reasons=('no_source_text',) + ctx)
    tier = 'abstract' if abstract else 'full_text'

    # 3. Full text but no passage to examine.
    if i.passage_count <= 0:
        if i.retrieval_error:
            return Decision(Verdict.RETRIEVAL_FAILED, 'PASSAGE_RETRIEVAL_ERROR',
                            'Passage retrieval raised an error for this pair; no source passage was examined.',
                            review_required=True, review_reasons=('evidence_retrieval_failed',) + ctx, evidence_tier=tier)
        return Decision(Verdict.INSUFFICIENT_EVIDENCE, 'NO_RELEVANT_PASSAGE',
                        'Source full text was retrieved but no passage could be assessed against this proposition.',
                        review_required=True, review_reasons=('no_relevant_passage',) + ctx, evidence_tier=tier)

    # 4. Passages exist; the semantic check must have actually succeeded.
    if i.nli_status != 'SUCCESS':
        return Decision(Verdict.NLI_UNAVAILABLE, 'NLI_NOT_RUN',
                        f'Candidate passages were found, but semantic verification did not run '
                        f'(status: {i.nli_status or "none"}). Lexical overlap and reranker relevance cannot establish support.',
                        review_required=True, limitations=('semantic_verification_unavailable',),
                        review_reasons=('nli_unavailable',) + ctx, evidence_tier=tier)
    done = True  # from here on a model result exists
    abstract_limit = ('abstract_only_evidence',) if abstract else ()

    def fallback(code, text):
        """Abstract-only evidence that does not settle the claim: report the full-text state, not a verdict."""
        verdict = SOURCE_STATUSES.get(i.fallback_status, Verdict.SOURCE_METADATA_ONLY)
        return Decision(verdict, code, text + ' Full text was not available. ' + (i.source_reason or ''),
                        review_required=True, semantic_verification_completed=done, limitations=abstract_limit,
                        review_reasons=('abstract_insufficient',) + ctx, evidence_tier=tier)

    # 5. Disagreement between passages is preserved, never resolved in favour of either side.
    if i.nli_conflict:
        return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'CONFLICTING_PASSAGES',
                        'Different source passages give both entailment and contradiction signals; a person must decide.',
                        review_required=True, semantic_verification_completed=done,
                        disagreements=('nli_passage_conflict',), review_reasons=('conflicting_passages',) + ctx,
                        evidence_tier=tier)
    if i.nli_truncated:
        return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'INPUT_TRUNCATED',
                        'The proposition or passage exceeded the model context window; the model saw only part of it.',
                        review_required=True, semantic_verification_completed=done, limitations=('input_truncated',),
                        review_reasons=('nli_input_truncated',) + ctx, evidence_tier=tier)

    num_failed = i.numerical_status == 'FAILED'
    num_open = i.numerical_status in ('NUMERICAL_CHECK_UNSUPPORTED', 'VALUES_MISSING')

    # 6. Contradiction only when the model found contradiction in a RELEVANT retrieved full-text passage.
    if i.nli_label == 'CONTRADICTION':
        if abstract:
            return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'ABSTRACT_CONTRADICTION',
                            'The NLI model flags a contradiction with the source abstract; an abstract cannot ground a '
                            'contradiction verdict without the full text.', review_required=True,
                            semantic_verification_completed=done, limitations=abstract_limit,
                            review_reasons=('abstract_contradiction',) + ctx, evidence_tier=tier)
        if not i.contradiction_grounded:
            return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'UNGROUNDED_CONTRADICTION',
                            'The NLI model flags a contradiction, but only in a passage the reranker scores as weakly '
                            'relevant to this proposition; the signal is not grounded enough for a contradiction verdict.',
                            review_required=True, semantic_verification_completed=done,
                            review_reasons=('weakly_relevant_contradiction',) + ctx, evidence_tier=tier)
        return Decision(Verdict.VERIFIED_CONTRADICTED, 'NLI_CONTRADICTION',
                        'The NLI model found contradiction between the proposition and a relevant retrieved source passage.',
                        review_required=True,  # contradiction claims are high-impact; always human-checked
                        semantic_verification_completed=done, review_reasons=('contradiction_check',) + ctx,
                        evidence_tier=tier)

    if i.nli_label == 'ENTAILMENT':
        if num_failed:
            return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'NUMERICAL_DISAGREEMENT',
                            'NLI entails the proposition but an explicit numerical check disagrees; a person must decide.',
                            review_required=True, semantic_verification_completed=done,
                            disagreements=('nli_vs_numerical',), review_reasons=('numerical_disagreement',) + ctx,
                            evidence_tier=tier)
        if num_open:
            return Decision(Verdict.INSUFFICIENT_EVIDENCE, 'NUMERICAL_UNRESOLVED',
                            'NLI entails the proposition but its numerical values could not be found in the passage.',
                            review_required=True, semantic_verification_completed=done,
                            limitations=('numerical_check_unresolved',), review_reasons=('numerical_unresolved',) + ctx,
                            evidence_tier=tier)
        if abstract:
            return Decision(Verdict.VERIFIED_SUPPORTED, 'ABSTRACT_ENTAILMENT',
                            'The abstract of the identified source entails the proposition; full text was not available, '
                            'so qualifiers stated only in the body could not be checked.',
                            review_required=True, semantic_verification_completed=done, limitations=abstract_limit,
                            review_reasons=('abstract_only_evidence',) + ctx, evidence_tier=tier)
        return Decision(Verdict.VERIFIED_SUPPORTED, 'NLI_ENTAILMENT',
                        'The NLI model finds the proposition entailed by a retrieved source passage and no numerical check conflicts.',
                        review_required=bool(ctx), semantic_verification_completed=done, review_reasons=ctx,
                        evidence_tier=tier)

    # 7. NLI neutral for the whole proposition: independently checked components may still settle it.
    parts = combine_components(i.component_verdicts) if i.component_verdicts else None
    if parts is Verdict.VERIFIED_CONTRADICTED and (abstract or not i.contradiction_grounded):
        parts = None
    if parts in RESOLVED:
        text = {Verdict.VERIFIED_SUPPORTED: 'Every independently checked component is entailed by a retrieved passage.',
                Verdict.VERIFIED_PARTIALLY_SUPPORTED: 'At least one independently checked component is entailed; another remains unresolved.',
                Verdict.VERIFIED_CONTRADICTED: 'An independently checked component is contradicted by a retrieved passage.'}[parts]
        return Decision(parts, 'COMPONENT_ROLLUP', text, review_required=True, semantic_verification_completed=done,
                        limitations=('component_level_check',) + abstract_limit,
                        review_reasons=('component_level_verdict',) + ctx, evidence_tier=tier)
    if i.component_verdicts and parts is None and Verdict.VERIFIED_CONTRADICTED in map(as_verdict, i.component_verdicts) \
            and Verdict.VERIFIED_SUPPORTED in map(as_verdict, i.component_verdicts):
        return Decision(Verdict.NEEDS_HUMAN_REVIEW, 'MIXED_COMPONENTS',
                        'Some components are supported and others contradicted; a person must decide.',
                        review_required=True, semantic_verification_completed=done,
                        disagreements=('component_support_vs_contradiction',), review_reasons=('mixed_components',) + ctx,
                        evidence_tier=tier)
    if abstract:
        return fallback('ABSTRACT_NOT_ENTAILED', 'The source abstract was examined and does not establish the proposition.')
    return Decision(Verdict.INSUFFICIENT_EVIDENCE, 'NOT_ENTAILED',
                    'The retrieved passages were assessed and none entails the proposition. This does not show the citation is wrong.',
                    review_required=True, semantic_verification_completed=done,
                    disagreements=('numerical_discrepancy_flagged',) if num_failed else (),
                    review_reasons=('not_entailed',) + ctx, evidence_tier=tier)


def assert_decision_invariants(d: Decision, i: DecisionInput) -> None:
    """Cheap runtime guard; raises AssertionError if a rule is violated."""
    if d.verdict in RESOLVED:
        assert i.source_status in ('FULL_TEXT', 'ABSTRACT_ONLY'), 'resolved verdict without source text'
        assert i.passage_count > 0, 'resolved verdict without a passage'
        assert i.nli_status == 'SUCCESS', 'resolved verdict without a successful semantic check'
        assert d.semantic_verification_completed, 'resolved verdict not flagged as semantically verified'
    if d.verdict is Verdict.VERIFIED_CONTRADICTED:
        assert d.evidence_tier == 'full_text', 'contradiction verdict without full-text evidence'
    if d.verdict is Verdict.NLI_UNAVAILABLE:
        assert not d.semantic_verification_completed
    if d.verdict not in RESOLVED and d.verdict is not Verdict.NOT_VERIFIABLE:
        assert d.review_required, 'unresolved pair hidden from the review queue'


NEXT_ACTIONS = {
    Verdict.VERIFIED_SUPPORTED: 'Spot-check the highlighted passage; model support is not a substitute for reading the source.',
    Verdict.VERIFIED_PARTIALLY_SUPPORTED: 'Read the source: only some components of this proposition are supported. Revise the claim or add a citation for the rest.',
    Verdict.VERIFIED_CONTRADICTED: 'Read the source passage and the claim; correct the claim or the citation if the contradiction is real.',
    Verdict.INSUFFICIENT_EVIDENCE: 'The retrieved passages do not establish the claim. Read the source to check whether support exists elsewhere in it.',
    Verdict.SOURCE_METADATA_ONLY: 'Supply the source PDF (upload it as a source) or obtain it through your library, then re-run.',
    Verdict.SOURCE_UNAVAILABLE: 'Check the bibliography entry for errors, or upload the source PDF, then re-run.',
    Verdict.RETRIEVAL_FAILED: 'Retrieval did not complete; re-run later or upload the source PDF.',
    Verdict.NLI_UNAVAILABLE: 'Prepare the models (python scripts/prepare_models.py), confirm /api/models/health reports READY, and re-run.',
    Verdict.NOT_VERIFIABLE: 'Inspect the original PDF; this text is not a safely isolated assertion.',
    Verdict.NEEDS_HUMAN_REVIEW: 'Read the listed passages and decide manually; the automated signals conflict or are unreliable here.',
}


def next_action(verdict: Verdict, review_required: bool = False) -> str:
    base = NEXT_ACTIONS[verdict]
    return base + (' Flagged for manual review.' if review_required and verdict not in (Verdict.NOT_VERIFIABLE, Verdict.NEEDS_HUMAN_REVIEW) else '')
