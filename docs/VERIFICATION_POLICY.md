# Verification policy (v2.1)

All verdicts come from `app/core/verdicts.py::decide()`. Statistics come from `app/core/stats.py`.
Nothing else assigns a status.

## Verdicts
| Verdict | Meaning |
|---|---|
| `VERIFIED_SUPPORTED` | A relevant retrieved passage entails the proposition; no numerical check conflicts |
| `VERIFIED_PARTIALLY_SUPPORTED` | Whole proposition not entailed; at least one derived component entailed, the rest unresolved |
| `VERIFIED_CONTRADICTED` | A relevant **full-text** passage contradicts the proposition in both NLI directions (always reviewed) |
| `INSUFFICIENT_EVIDENCE` | Source text examined; it does not establish the proposition (not "citation wrong") |
| `SOURCE_METADATA_ONLY` | Source identified; full text not retrievable (or only an abstract that did not settle it) |
| `SOURCE_UNAVAILABLE` | The reference could not be identified |
| `RETRIEVAL_FAILED` | Identification, full-text download, PDF extraction (`reason_code=EXTRACTION_FAILED`) or passage retrieval did not complete |
| `NLI_UNAVAILABLE` | Passages exist but the semantic check did not run |
| `NOT_VERIFIABLE` | Navigation, acknowledgement or flattened table |
| `NEEDS_HUMAN_REVIEW` | Conflicting passages, one-directional contradiction, NLI-vs-numeric disagreement, truncated input, mixed components |

Pre-v2.1 names (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONTRADICTED`, `EXTRACTION_FAILED`) are accepted when reading old reports
(`verdicts.as_verdict`). Each pair also records `failure_stage` (reference_matching, identification, full_text_retrieval,
extraction, evidence_retrieval, nli) so failures are attributed to the stage that failed.

## Rules enforced (each has a test)
1. No passage → no verified verdict. 2. Metadata is never evidence; a provider **abstract** of a HIGH-confidence identified
source may be examined only when no full text exists. It is labelled `evidence_tier=abstract`: entailment gives
`VERIFIED_SUPPORTED` with limitation `abstract_only_evidence` and review; a neutral result falls back to the full-text state;
an abstract can never ground a contradiction. 3. Lexical overlap never entails. 4. Failed/unavailable NLI → `NLI_UNAVAILABLE`.
5. Unavailable source ≠ wrong citation. 6. Contradiction must be **grounded**: the passage must be relevant (cross-encoder
logit ≥ 0, or lexical ≥ 0.25 without a cross-encoder) and the contradiction must hold in the reverse direction (claim as
premise, passage sentence as hypothesis, contradiction ≥ 0.5). Otherwise `NEEDS_HUMAN_REVIEW`.
7. Entailment only counts from a relevant passage (cross-encoder logit ≥ −1.0, or lexical ≥ 0.10); entailments from weakly
relevant passages are recorded in `nli.gated_out` and ignored. 8. Partial support is never upgraded; mixed support +
contradiction is `NEEDS_HUMAN_REVIEW`. 9. Scores are `uncalibrated_model_score`, never a probability of correctness.
10. With no passage, the explanation says no passage was examined. 11. **Every unresolved pair is in the review queue**
(`review_required=True` with `review_reasons`); verified pairs are also queued when citation scope is ambiguous, the source
identity needs review, decomposition is uncertain, evidence is abstract-only or the verdict is a contradiction.

The relevance thresholds are heuristics chosen on the development paper, not calibrated values.

## Source identification
`app/services/identification.py`. Exact identifiers first (arXiv → arXiv API + Semantic Scholar; DOI → Crossref + OpenAlex),
then Crossref bibliographic search + OpenAlex title search, Semantic Scholar title match, arXiv title search. A DOI is not
required. Every candidate is scored on title similarity, first-author/author overlap, year, venue, volume, pages and
identifiers. Title similarity alone never accepts a match. Statuses: `IDENTIFIED_HIGH_CONFIDENCE`, `IDENTIFIED_NEEDS_REVIEW`,
`AMBIGUOUS_SOURCE` (different works score alike; none chosen), `SOURCE_NOT_FOUND` (all providers searched successfully),
`PROVIDER_ERROR`, `RATE_LIMITED` (the source may exist). Every provider request is an attempt record.

## Full text
`app/services/fulltext.py`: arXiv PDF → Europe PMC open-access JATS → Unpaywall/OpenAlex/Semantic Scholar open-access
PDF or landing pages → Crossref links → DOI landing page (publisher-declared `citation_pdf_url`, or open HTML full text).
Logins (401/403, institutional IdP redirects) are recorded as `ACCESS_RESTRICTED` and never worked around. A document is
used only after identity validation against its front matter (identifier, or title + author surnames) and only if it is
substantive (abstract-only pages are rejected). PDFs are parsed with pdfium (column order), falling back to pdfplumber and
then OCR for image-only PDFs.

## Claims and citation scope
`app/core/claims.py`. A sentence is split only at a coordinating joint where the right-hand side has its own subject and
finite verb and the left-hand side has a verb of its own (coordinated subjects are not split). Never split after a leading
subordinate clause, before an anaphoric opener (`this`, `they`, `thus`), across `respectively`, or at causal/temporal/
conditional connectives. Coordinated predicates with a shared subject stay one proposition; derived subject+predicate
components are checked only to detect partial support. Associations: `direct`, `sentence_level` (one proposition, several
cited works), `multi_proposition` (sentence-final marker that may cover several propositions; reviewed), `ambiguous_scope`
(proposition without its own marker; reviewed).

## Statistics
Pair coverage is hierarchical: pairs ⊇ source-identified ⊇ full-text ⊇ with full-text passages ⊇ full-text NLI completed;
abstract-tier evidence is counted separately. Verdict buckets and failure stages each partition the pairs. Support rate =
VERIFIED_SUPPORTED (full text) / full-text pairs with a conclusive NLI outcome; when the denominator is 0 the UI says the rate
is not computed because evidence is missing. Evidence coverage = pairs with ≥1 examined passage / assertion pairs.
Reference-level metrics (identified, full text, usable text, attempts) are reported separately with the unit "references".
`compute_stats` raises `StatsInconsistency` on impossible states, double counting or an unresolved pair outside the review
queue; the orchestrator then flags the report instead of silently showing the numbers.
