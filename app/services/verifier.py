"""NLI verification of one proposition against its retrieved source passages.

Every claim-passage pair records: the claim (hypothesis), the passage (premise), the model id, the raw label
scores, the mapped ENTAILMENT / NEUTRAL / CONTRADICTION scores, processing time and any error. Raw scores are
uncalibrated model outputs, never probabilities that a citation is correct.

Operational states (per call and per run):
  MODEL_LOADED -> INFERENCE_STARTED -> INFERENCE_COMPLETED | INFERENCE_FAILED
A model that is READY but never ran inference (e.g. no passages were retrieved) stays MODEL_LOADED, so a report
can no longer imply that verification happened when it did not.

This module produces model outputs only; verdicts are decided centrally by app.core.verdicts.decide().
"""
import logging
import re
import time
from typing import Any

from app.core.model_health import ModelUnavailable, normalize_nli_scores
from app.core.models import NLI_ID, nli_label_map, nli_pipeline
from app.core.schemas import (
    ClaimCitationPair,
    NumericalComparison,
    RetrievedEvidence,
    VerificationLabel,
    VerificationResult,
)


def pipeline(*args, **kwargs):
    return nli_pipeline()


logger = logging.getLogger(__name__)

NLI_MAX_WORDS = 300          # word-level guard when no tokenizer is available
NLI_MAX_TOKENS = 512
# Relevance gates (heuristic, documented in docs/VERIFICATION_POLICY.md). The cross-encoder never decides whether a
# citation is correct; it only decides whether a passage is ABOUT the proposition, so that NLI output on an
# off-topic passage is not used. ms-marco-MiniLM logits are > 0 for relevant passages and strongly negative for
# unrelated ones; lexical scores are 0..1 and only used when the cross-encoder is unavailable.
SUPPORT_MIN_RERANK = -1.0
SUPPORT_MIN_LEXICAL = 0.10
CONTRADICTION_MIN_RERANK = 0.0
CONTRADICTION_MIN_LEXICAL = 0.25
# A genuine contradiction is symmetric: if the passage contradicts the claim, the claim contradicts the passage.
# MNLI models often emit one-directional "contradiction" when the hypothesis merely adds details.
SYMMETRY_MIN_CONTRADICTION = 0.5


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def clean_hypothesis(text: str) -> str:
    clean = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", text)
    clean = re.sub(r"\((?:19|20)\d{2}[a-z]?\)", "", clean)
    clean = re.sub(r"^(?:Section\s+\d+[:\.]?|\d+\.\d*\s+)", "", clean.strip(), flags=re.IGNORECASE).strip()
    return ' '.join(clean.split())


class Verifier:
    def __init__(self, use_nli: bool = True, entailment_threshold: float = 0.7, contradiction_threshold: float = 0.7):
        self.use_nli = use_nli
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        from backend.modules.nli_verifier import ReasoningGenerator
        self.reasoner = ReasoningGenerator()
        self.usage = {'model': NLI_ID, 'state': 'NOT_LOADED', 'inference_calls': 0, 'batched_calls': 0,
                      'passages_scored': 0, 'failures': 0, 'seconds': 0.0, 'last_error': None}
        self._cache: dict[tuple[str, str], dict] = {}

        self.nli_model = None
        self.nli_unavailable: dict | None = None
        if self.use_nli:
            try:
                # Loaded once per process by the model registry, which has already run a real inference probe.
                self.nli_model = pipeline("text-classification")
                self.usage['state'] = 'MODEL_LOADED'
            except ModelUnavailable as exc:
                self.nli_unavailable = exc.status.as_dict()
                logger.warning("NLI unavailable (%s at %s): %s", exc.status.cause, exc.status.stage, exc.status.error)
                self.nli_model, self.use_nli = None, False
            except Exception as exc:  # noqa: BLE001 - never let init errors escape as a silent success
                self.nli_unavailable = {'status': 'UNAVAILABLE', 'cause': 'INIT_ERROR', 'error': f'{type(exc).__name__}: {exc}'[:300]}
                logger.warning("NLI unavailable: %s", exc)
                self.nli_model, self.use_nli = None, False
        else:
            self.nli_unavailable = {'status': 'DISABLED', 'cause': 'DISABLED', 'error': 'NLI was disabled by configuration (baseline mode).'}
            self.usage['state'] = 'DISABLED'

    # ---- numerical checks (unchanged policy) ---------------------------------------------------------
    def _extract_numerical_entities(self, text: str) -> list[str]:
        clean = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", text)
        clean = re.sub(r"\([^)]*(?:19|20)\d{2}[a-z]?\)", "", clean)
        pattern = r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[- ]?\s*(?:percentage points?|percent|%|mg|kg|km|cm|mm|meters?|hours?|weeks?|days?|years?|months?|seconds?|million|billion|g\b|m\b))?"
        return list(dict.fromkeys(m.group().strip() for m in re.finditer(pattern, clean, re.I)))

    def _quantity(self, entity):
        match = re.match(r"([+-]?[\d,]+(?:\.\d+)?)(.*)", entity.lower())
        value = float(match[1].replace(',', ''))
        unit = match[2].strip(' -').replace('percent', '%')
        unit = {'meters': 'm', 'meter': 'm', 'weeks': 'week', 'days': 'day',
                'years': 'year', 'hours': 'hour', 'seconds': 'second'}.get(unit, unit)
        scales = {'kg': ('g', 1000), 'mg': ('g', .001), 'km': ('m', 1000), 'cm': ('m', .01), 'mm': ('m', .001)}
        base, scale = scales.get(unit, (unit, 1))
        return round(value * scale, 9), base

    def _check_numerical_match(self, claim_text, evidence_text):
        return self._check_numerical_alignment(claim_text, evidence_text).is_match

    def _check_numerical_alignment(self, claim_text, evidence_text):
        claim_stats = self._extract_numerical_entities(claim_text)
        evidence_stats = self._extract_numerical_entities(evidence_text)
        from app.services.numerical import check_change
        structured = check_change(claim_text, evidence_text, claim_stats, evidence_stats)
        if structured is not None:
            return structured
        values = {self._quantity(e) for e in evidence_stats}
        unmatched = [c for c in claim_stats if self._quantity(c) not in values]
        match = not unmatched if claim_stats else None
        return NumericalComparison(status=('VALUES_PRESENT' if match else 'VALUES_MISSING') if claim_stats else 'NOT_APPLICABLE', has_numerical_data=bool(claim_stats),
            claim_entities=claim_stats, evidence_entities=evidence_stats, is_match=match,
            details=('Values present in evidence; context still requires NLI.' if match else
                     'Missing or different values: ' + ', '.join(unmatched)) if claim_stats else None)

    # ---- NLI inference ----------------------------------------------------------------------------
    def _is_hf_pipeline(self) -> bool:
        return type(self.nli_model).__module__.startswith('transformers')

    def _truncated(self, premise: str, hypothesis: str) -> bool:
        tokenizer = getattr(self.nli_model, 'tokenizer', None) if self._is_hf_pipeline() else None
        if tokenizer is not None:
            try:
                return len(tokenizer(premise, hypothesis)['input_ids']) > NLI_MAX_TOKENS
            except Exception:  # noqa: BLE001 - fall back to the word heuristic
                pass
        return len(hypothesis.split()) > 80 or len(premise.split()) > NLI_MAX_WORDS

    def _raw_predictions(self, inputs: list[dict]) -> list[list[dict] | None]:
        """One batched call for a real HF pipeline; per-item calls otherwise (test doubles take one dict)."""
        self.usage['state'] = 'INFERENCE_STARTED'
        if self._is_hf_pipeline() and len(inputs) > 1:
            self.usage['batched_calls'] += 1
            raw = self.nli_model(inputs, batch_size=min(16, len(inputs)))
            return [r if isinstance(r, list) else [r] for r in raw]
        out = []
        for item in inputs:
            try:
                raw = self.nli_model(item)
                out.append(raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw)
            except Exception as exc:  # noqa: BLE001 - recorded per passage
                logger.warning('NLI inference error: %s', exc)
                self.usage['last_error'] = f'{type(exc).__name__}: {exc}'[:300]
                out.append(None)
        return out

    def score_passages(self, hypothesis: str, premises: list[str]) -> list[dict]:
        """Run NLI for (premise_i, hypothesis). Returns one record per premise; never raises."""
        hypothesis = clean_hypothesis(hypothesis)
        if not self.use_nli or not self.nli_model:
            info = self.nli_unavailable or {}
            return [{'status': 'MODEL_UNAVAILABLE', 'operational_state': self.usage['state'], 'model': NLI_ID,
                     'reason': info.get('error') or 'NLI model not loaded.', 'cause': info.get('cause')} for _ in premises]
        records: list[dict | None] = [None] * len(premises)
        todo = []
        for i, premise in enumerate(premises):
            key = (premise, hypothesis)
            if key in self._cache:
                records[i] = {**self._cache[key], 'cached': True}
            else:
                todo.append(i)
        if todo:
            inputs = [{'text': _truncate_to_words(premises[i], NLI_MAX_WORDS), 'text_pair': _truncate_to_words(hypothesis, 80)}
                      for i in todo]
            started = time.monotonic()
            try:
                raws = self._raw_predictions(inputs)
                error = None
            except Exception as exc:  # noqa: BLE001
                raws, error = [None] * len(todo), f'{type(exc).__name__}: {exc}'[:300]
                self.usage['last_error'] = error
            took = time.monotonic() - started
            self.usage['inference_calls'] += 1
            self.usage['seconds'] = round(self.usage['seconds'] + took, 3)
            config = getattr(getattr(self.nli_model, 'model', None), 'config', None)
            id2label = getattr(config, 'id2label', None)
            id2label = id2label if isinstance(id2label, dict) else {}
            for i, raw in zip(todo, raws):
                rec = {'model': NLI_ID, 'seconds': round(took / max(1, len(todo)), 4), 'premise_words': len(premises[i].split())}
                if not raw:
                    self.usage['failures'] += 1
                    rec.update(status='INFERENCE_FAILED', operational_state='INFERENCE_FAILED',
                               reason='The NLI model raised an error or returned no scores for this passage.',
                               error=error or self.usage['last_error'])
                else:
                    try:
                        scores, mapping = normalize_nli_scores(raw, id2label, nli_label_map())
                    except (KeyError, ValueError, TypeError):
                        scores, mapping = None, 'unmapped'
                    if scores is None:
                        self.usage['failures'] += 1
                        rec.update(status='LABEL_MAPPING_UNAVAILABLE', operational_state='INFERENCE_FAILED',
                                   reason='Model output labels could not be mapped to entailment/contradiction/neutral.',
                                   raw=[{'label': str(p.get('label')), 'score': float(p.get('score', 0))} for p in raw])
                    else:
                        self.usage['passages_scored'] += 1
                        entail, contra = scores['ENTAILMENT'], scores['CONTRADICTION']
                        decision = ('CONTRADICTION' if contra >= self.contradiction_threshold else
                                    'ENTAILMENT' if entail >= self.entailment_threshold else 'NEUTRAL')
                        best = max(scores, key=scores.get)
                        rec.update(status='SUCCESS', operational_state='INFERENCE_COMPLETED', label=best,
                                   score=scores[best], decision_label=decision, scores=scores,
                                   entailment=entail, neutral=scores['NEUTRAL'], contradiction=contra,
                                   raw=[{'label': str(p['label']), 'score': float(p['score'])} for p in raw],
                                   label_mapping=mapping, truncated=self._truncated(premises[i], hypothesis))
                        self._cache[(premises[i], hypothesis)] = dict(rec)
                records[i] = rec
        ok = sum(1 for r in records if r and r.get('status') == 'SUCCESS')
        self.usage['state'] = 'INFERENCE_COMPLETED' if ok else ('INFERENCE_FAILED' if records else self.usage['state'])
        return records  # type: ignore[return-value]

    # ---- relevance and grounding ----------------------------------------------------------------------
    @staticmethod
    def _relevant(evidence: RetrievedEvidence, rerank_min: float, lexical_min: float) -> bool:
        if evidence.rerank_score is not None:
            return evidence.rerank_score >= rerank_min
        # Without a cross-encoder the passage's lexical retrieval score is the only relevance signal.
        return max(evidence.retrieval_score or 0, evidence.relevance_score or 0) >= lexical_min

    def _grounded(self, evidence: RetrievedEvidence) -> bool:
        return self._relevant(evidence, CONTRADICTION_MIN_RERANK, CONTRADICTION_MIN_LEXICAL)

    def _symmetric(self, hypothesis: str, passages: list[RetrievedEvidence]) -> dict[str, dict]:
        """Reverse-direction NLI for contradiction candidates: premise = claim, hypothesis = each SENTENCE of the
        passage. Sentence level, because a passage usually says more than the claim, and "the claim does not mention
        X" is neutral, not contradictory. Returns the most contradictory sentence record per passage."""
        out = {}
        claim = clean_hypothesis(hypothesis)
        for i, e in enumerate(passages):
            from app.services.extractor import Extractor
            sentences = [s for s in Extractor()._split_sentences(e.evidence_text) if len(s.split()) >= 4] or [e.evidence_text]
            records = [self.score_passages(s, [claim])[0] for s in sentences[:8]]
            ok = [r for r in records if r.get('status') == 'SUCCESS']
            out[e.evidence_id or str(i)] = max(ok, key=lambda r: r['contradiction']) if ok else (records[0] if records else {})
        return out

    def _classify(self, claim_text: str, evidence: list[RetrievedEvidence], records: list[dict]) -> dict:
        """Gate per-passage NLI decisions by relevance and contradiction symmetry. Returns a summary dict."""
        ok = [(e, r) for e, r in zip(evidence, records) if r.get('status') == 'SUCCESS']
        entail, contra, ignored = [], [], []
        for e, r in ok:
            if r['decision_label'] == 'ENTAILMENT':
                if self._relevant(e, SUPPORT_MIN_RERANK, SUPPORT_MIN_LEXICAL):
                    entail.append((e, r))
                else:
                    r['gate'] = 'entailment ignored: passage weakly relevant to the proposition'
                    ignored.append((e, r))
            elif r['decision_label'] == 'CONTRADICTION':
                if self._grounded(e):
                    contra.append((e, r))
                else:
                    r['gate'] = 'contradiction ignored: passage weakly relevant to the proposition'
                    ignored.append((e, r))
        symmetric = self._symmetric(claim_text, [e for e, _ in contra])
        grounded, asymmetric = [], []
        for i, (e, r) in enumerate(contra):
            rev = symmetric.get(e.evidence_id or str(i)) or {}
            r['reverse_contradiction'] = rev.get('contradiction')
            if rev.get('status') == 'SUCCESS' and rev['contradiction'] >= SYMMETRY_MIN_CONTRADICTION:
                grounded.append((e, r))
            else:
                r['gate'] = ('contradiction not confirmed in the reverse direction '
                             f'(reverse contradiction score {rev.get("contradiction")})')
                asymmetric.append((e, r))
        return {'ok': ok, 'entail': entail, 'contra': grounded, 'asymmetric': asymmetric, 'ignored': ignored}

    # ---- one proposition ----------------------------------------------------------------------------

    def _carry(self, claim):
        return dict(manuscript_id=claim.manuscript_id, sentence_id=claim.sentence_id, sentence_text=claim.sentence_text,
                    atomic_claim_id=claim.atomic_claim_id, association_rationale=claim.association_rationale,
                    resolved_reference_id=claim.resolved_reference_id, page_mapping=claim.page_mapping,
                    clause_text=claim.clause_text, decomposition_method=claim.decomposition_method,
                    clause_relation=claim.clause_relation, decomposition_rationale=claim.decomposition_rationale,
                    decomposition_uncertain=claim.decomposition_uncertain, predicate_components=claim.predicate_components,
                    association_type=claim.association_type, ambiguity_status=claim.ambiguity_status)

    def _verify_one(self, claim: ClaimCitationPair, evidence_list: list[RetrievedEvidence]) -> VerificationResult:
        return self.verify(claim, evidence_list, False)

    def verify(self, claim, evidence_list, explain=True):
        carry = self._carry(claim)
        if not evidence_list:
            result = VerificationResult(
                id=claim.id, text=claim.text, context=claim.context, citation_marker=claim.citation_marker,
                source_document="No Source Document", evidence="", evidence_list=[],
                status=VerificationLabel.INSUFFICIENT, confidence=None,  # no passage => no score, not a zero score
                numerical_check=None, numerical_comparison=None, page_number=claim.page_number,
                nli={'status': 'NO_EVIDENCE', 'operational_state': 'NOT_RUN', 'label': None, 'score': None,
                     'decision_label': None, 'reason': 'No source passage was available, so no inference was run.'}, **carry)
            return self.reasoner.explain(result, self.use_nli) if explain else result
        records = self.score_passages(claim.text, [e.evidence_text for e in evidence_list])
        evidence_list = [e.model_copy(update={'nli': r}) for e, r in zip(evidence_list, records)]
        ok = [(e, r) for e, r in zip(evidence_list, records) if r.get('status') == 'SUCCESS']
        hypothesis = clean_hypothesis(claim.text)
        verbatim_norm = re.sub(r"[^a-z0-9 ]", "", hypothesis.lower()).strip()

        if not ok:
            first = records[0]
            chosen = evidence_list[0]
            nli = {'status': first.get('status', 'INFERENCE_FAILED'), 'operational_state': first.get('operational_state'),
                   'label': None, 'score': None, 'decision_label': None, 'model': NLI_ID,
                   'reason': first.get('reason'), 'cause': first.get('cause'), 'passages_scored': 0,
                   'passages_examined': len(evidence_list)}
            label, confidence = VerificationLabel.INSUFFICIENT, None
        else:
            gated = self._classify(claim.text, evidence_list, records)
            entail, contra, asymmetric = gated['entail'], gated['contra'], gated['asymmetric']
            conflict = bool(entail and contra)
            grounded = True
            if entail:
                chosen, rec = max(entail, key=lambda x: x[1]['entailment'])
                decision = 'NEUTRAL' if conflict else 'ENTAILMENT'
            elif contra:
                chosen, rec = max(contra, key=lambda x: x[1]['contradiction'])
                decision = 'CONTRADICTION'
            elif asymmetric:
                # Relevant passage, one-directional contradiction: not grounded enough for a verdict.
                chosen, rec = max(asymmetric, key=lambda x: x[1]['contradiction'])
                decision, grounded = 'CONTRADICTION', False
            else:
                # Neutral: report the most entailing passage that passed the relevance gate, so the displayed
                # scores never show a gated-out (off-topic) entailment as if it were the decisive evidence.
                gated_ids = {id(r) for _, r in gated['ignored']}
                pool = [(e, r) for e, r in gated['ok'] if id(r) not in gated_ids] or gated['ok']
                chosen, rec = max(pool, key=lambda x: x[1]['entailment'])
                decision = 'NEUTRAL'
            nli = {'status': 'SUCCESS', 'operational_state': 'INFERENCE_COMPLETED', 'model': NLI_ID,
                   'label': rec['label'], 'score': rec['score'], 'decision_label': decision, 'scores': rec['scores'],
                   'raw': rec.get('raw'), 'label_mapping': rec.get('label_mapping'),
                   'chosen_evidence_id': chosen.evidence_id, 'passages_scored': len(gated['ok']),
                   'passages_examined': len(evidence_list), 'truncated': rec.get('truncated', False),
                   'thresholds': {'entailment': self.entailment_threshold, 'contradiction': self.contradiction_threshold,
                                  'support_min_rerank': SUPPORT_MIN_RERANK, 'contradiction_min_rerank': CONTRADICTION_MIN_RERANK,
                                  'symmetry_min_contradiction': SYMMETRY_MIN_CONTRADICTION},
                   'contradiction_grounded': grounded,
                   'gated_out': [{'evidence_id': e.evidence_id, 'label': r['decision_label'], 'gate': r.get('gate')}
                                 for e, r in gated['ignored'] + asymmetric],
                   'seconds': round(sum(r.get('seconds', 0) for _, r in gated['ok']), 4)}
            ok = gated['ok']
            if conflict:
                nli.update(conflict=True, conflict_detail='Different retrieved passages entail and contradict the claim.')
            confidence = None if conflict else {'CONTRADICTION': rec['contradiction'], 'ENTAILMENT': rec['entailment']}.get(decision, rec['score'])
            label = {'CONTRADICTION': VerificationLabel.CONTRADICTED,
                     'ENTAILMENT': VerificationLabel.SUPPORTED}.get(decision, VerificationLabel.INSUFFICIENT)
        normalized_evidence = re.sub(r"[^a-z0-9 ]", "", chosen.evidence_text.lower())
        nli['lexical_verbatim_match'] = bool(verbatim_norm) and verbatim_norm in normalized_evidence  # informational only
        num_comp = self._check_numerical_alignment(claim.text, chosen.evidence_text)
        if label == VerificationLabel.SUPPORTED and num_comp.is_match is False:
            label = VerificationLabel.NUMERICAL_MISMATCH
        conf_val = None if confidence is None else round(confidence * 100, 1)
        result = VerificationResult(
            id=claim.id, text=claim.text, context=claim.context, citation_marker=claim.citation_marker,
            source_document=chosen.source_document, evidence=chosen.evidence_text, evidence_list=evidence_list,
            status=label, confidence=conf_val, numerical_check=num_comp.details, numerical_comparison=num_comp,
            page_number=claim.page_number, nli=nli, **carry)
        return self.reasoner.explain(result, self.use_nli and self.nli_model is not None) if explain else result

    def component_checks(self, components: list[str], evidence_list: list[RetrievedEvidence]) -> list[dict[str, Any]]:
        """Check derived sub-propositions against the same passages (for partial support). One record each."""
        out = []
        for text in components:
            records = self.score_passages(text, [e.evidence_text for e in evidence_list])
            gated = self._classify(text, evidence_list, [dict(r) for r in records])
            ok, entail, contra = gated['ok'], gated['entail'], gated['contra']
            if not ok:
                out.append({'text': text, 'verdict': 'NLI_UNAVAILABLE', 'nli': records[0] if records else None})
                continue
            if entail and contra:
                verdict, (ev, rec) = 'NEEDS_HUMAN_REVIEW', entail[0]
            elif entail:
                verdict, (ev, rec) = 'VERIFIED_SUPPORTED', max(entail, key=lambda x: x[1]['entailment'])
            elif contra:
                verdict, (ev, rec) = 'VERIFIED_CONTRADICTED', max(contra, key=lambda x: x[1]['contradiction'])
            else:
                verdict, (ev, rec) = 'INSUFFICIENT_EVIDENCE', max(ok, key=lambda x: x[1]['entailment'])
            out.append({'text': text, 'verdict': verdict, 'evidence_id': ev.evidence_id,
                        'scores': rec.get('scores'), 'decision_label': rec.get('decision_label'), 'derived': True})
        return out
