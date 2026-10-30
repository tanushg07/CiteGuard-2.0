from enum import StrEnum

from pydantic import BaseModel, Field


class VerificationLabel(StrEnum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT = "Unrelated"
    NUMERICAL_MISMATCH = "Numerical Mismatch"

class ReferenceMetadata(BaseModel):
    raw_reference: str = ""
    provider: str = "bibliography"
    status: str = "not_found"
    title: str | None = None
    doi: str | None = None
    abstract: str | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    pdf_url: str | None = None
    urls: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    # IDENTIFIED_HIGH_CONFIDENCE | IDENTIFIED_NEEDS_REVIEW | AMBIGUOUS_SOURCE | SOURCE_NOT_FOUND | PROVIDER_ERROR |
    # RATE_LIMITED | NO_BIBLIOGRAPHY_ENTRY. `status` is the legacy coarse field ('enriched' when identified).
    identification_status: str | None = None
    identification: dict = Field(default_factory=dict)


class ClaimCitationPair(BaseModel):
    claim_group_id: str = ''
    citation_id: str = ''
    sentence_index: int = 0
    paragraph: int = 0
    association_confidence: str = 'sentence_local'
    claim_type: str = 'assertion_candidate'
    reference_metadata: ReferenceMetadata | None = None
    # Structured provenance (one pair = one atomic claim x one citation marker)
    manuscript_id: str = ''
    sentence_id: str = ''
    sentence_text: str = ''
    atomic_claim_id: str = ''
    association_rationale: str = ''
    resolved_reference_id: str | None = None
    page_mapping: str = 'ok'  # ok | missing | invalid | out_of_range - never silently defaulted to page 1
    # Atomic decomposition (v2.1). `text` is the proposition used as the NLI hypothesis.
    clause_text: str = ''                 # verbatim slice of the manuscript sentence (markers retained)
    decomposition_method: str = 'none'    # none | semicolon | coordinating_conjunction | citation_boundary
    clause_relation: str = 'none'         # relation to the previous proposition (conjunction, contrast, ...)
    decomposition_rationale: str = ''
    decomposition_uncertain: bool = False
    predicate_components: list[str] = Field(default_factory=list)  # derived sub-hypotheses (partial support only)
    # Citation association (v2.1): direct | sentence_level | multi_proposition | ambiguous_scope
    association_type: str = 'direct'
    ambiguity_status: str = 'unambiguous'  # unambiguous | shared | ambiguous
    id: str = Field(..., description="Unique identifier for the claim-citation pair")
    text: str = Field(..., description="The atomic claim text being assessed")
    context: str = Field(default="", description="The surrounding paragraph context")
    citation_marker: str = Field(..., description="The citation marker (e.g. [12], (Vaswani et al., 2017))")
    page_number: int | None = Field(default=None, description="Manuscript page number; None when the mapping is unknown")
    section: str | None = Field(default=None, description="Section heading if detected")

class NumericalComparison(BaseModel):
    status: str = 'NOT_APPLICABLE'
    calculation: str | None = None
    has_numerical_data: bool = Field(default=False)
    claim_entities: list[str] = Field(default_factory=list)
    evidence_entities: list[str] = Field(default_factory=list)
    is_match: bool | None = Field(default=None)
    details: str | None = Field(default=None)

class RetrievedEvidence(BaseModel):
    evidence_id: str = ''
    source_id: str = ''
    source_url: str | None = None
    section: str | None = None
    paragraph: int | None = None
    retrieval_score: float = 0
    rerank_score: float | None = None
    ranking_method: str = 'lexical'  # lexical = candidate proposal only, never evidence of support
    lexical_rank: int | None = None   # 1-based rank among lexical candidates
    rerank_rank: int | None = None    # 1-based rank after cross-encoder reranking (None when not reranked)
    candidate_count: int | None = None
    context: str | None = None        # surrounding source paragraph (the passage is a window inside it)
    evidence_tier: str = 'full_text'  # full_text | abstract (provider-supplied abstract of an identified source)
    extraction_method: str | None = None  # pdf_text | ocr:<engine> | html_article | jats_xml | metadata:<provider>
    nli: dict | None = None           # per-passage NLI outputs (raw scores, model id, timing)
    page_number: int | None = Field(default=None, description="Page in the SOURCE document; None when unknown")
    claim_id: str = Field(..., description="The ID of the claim this evidence is for")
    source_document: str = Field(..., description="The name or identifier of the source document")
    evidence_text: str = Field(..., description="The retrieved chunk of text from the source")
    relevance_score: float = Field(..., description="The retrieval/reranker relevance score")
    source_citation: str | None = Field(default=None, description="Bibliographic reference citation")

class VerificationResult(BaseModel):
    final_verdict: str = 'INSUFFICIENT_EVIDENCE'
    claim_group_id: str = ''
    citation_id: str = ''
    sentence_index: int = 0
    section: str | None = None
    association_confidence: str = 'sentence_local'
    source_identification: dict = Field(default_factory=dict)
    retrieval: dict = Field(default_factory=dict)
    nli: dict = Field(default_factory=dict)
    decision_reason: str = ''
    reasoning: str = ''
    reasoning_provider: str = 'template'
    reason_code: str = ''
    review_required: bool = False
    limitations: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    semantic_verification_completed: bool = False
    evidence_state: str = 'NONE'  # NONE | PASSAGES - explanations may only cite passages when PASSAGES
    retrieval_attempts: list[dict] = Field(default_factory=list)
    next_action: str = ''
    confidence_kind: str = 'uncalibrated_model_score'
    manuscript_id: str = ''
    sentence_id: str = ''
    sentence_text: str = ''
    atomic_claim_id: str = ''
    association_rationale: str = ''
    resolved_reference_id: str | None = None
    page_mapping: str = 'ok'
    reference_metadata: ReferenceMetadata | None = None
    clause_text: str = ''
    decomposition_method: str = 'none'
    clause_relation: str = 'none'
    decomposition_rationale: str = ''
    decomposition_uncertain: bool = False
    predicate_components: list[str] = Field(default_factory=list)
    association_type: str = 'direct'
    ambiguity_status: str = 'unambiguous'
    failure_stage: str | None = None   # reference_matching | identification | full_text_retrieval | extraction |
                                       # evidence_retrieval | nli | None (no failure)
    review_reasons: list[str] = Field(default_factory=list)
    evidence_tier: str | None = None   # full_text | abstract | None (no passage examined)
    id: str = Field(..., description="Claim ID")
    text: str = Field(..., description="Claim text")
    context: str = Field(default="", description="Paragraph context in target document")
    citation_marker: str = Field(default="", description="Citation marker in text")
    source_document: str = Field(default="Source Document", description="Source document or reference")
    evidence: str = Field(default="", description="Best retrieved evidence passage")
    evidence_list: list[RetrievedEvidence] = Field(default_factory=list, description="All retrieved candidates")
    status: VerificationLabel = Field(..., description="Final verification status")
    confidence: float | None = Field(default=None, description="Uncalibrated model score as a percentage; None when no model produced one. Not the probability that the citation is correct.")
    numerical_check: str | None = Field(default=None, description="Numerical comparison summary string")
    numerical_comparison: NumericalComparison | None = Field(default=None, description="Structured numerical check")
    page_number: int | None = None

    # Backwards compatibility properties
    @property
    def claim_id(self) -> str:
        return self.id

    @property
    def claim_text(self) -> str:
        return self.text

    @property
    def nli_label(self) -> VerificationLabel:
        return self.status

    @property
    def nli_confidence(self) -> float | None:
        if self.confidence is None:
            return None
        return round(self.confidence / 100.0, 2) if self.confidence > 1.0 else self.confidence

    @property
    def numerical_match(self) -> bool | None:
        if self.numerical_comparison is not None:
            return self.numerical_comparison.is_match
        return None

class DocumentSummary(BaseModel):
    total_pages: int = 0
    total_references: int = 0
    unique_claims: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    document_name: str
    total_claims: int
    supported_count: int
    contradicted_count: int
    unrelated_count: int
    numerical_mismatch_count: int
    supported_percentage: float | None  # DEPRECATED: = stats.support_rate.percent (eligible denominator, not all pairs)
    average_confidence: float | None    # mean of real model scores only; None if no model scored any pair
    processing_time_seconds: float
    stats: dict = Field(default_factory=dict)

class AnalysisResponse(BaseModel):
    document: dict = Field(default_factory=dict)
    sources: list[dict] = Field(default_factory=list)
    job_id: str
    summary: DocumentSummary
    claims: list[VerificationResult]
    status: str = "completed"
    warnings: list[str] = Field(default_factory=list)
    engines: dict[str, str] = Field(default_factory=dict)
    model_health: dict = Field(default_factory=dict)
    # What the models actually did in THIS run (loaded vs. inference started/completed/failed, counts, seconds).
    model_usage: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    references: list[dict] = Field(default_factory=list)   # one row per bibliography entry (reference-level audit)

class PipelineStepUpdate(BaseModel):
    step_id: int
    name: str
    status: str  # "pending" | "loading" | "complete" | "error"
    detail: str | None = None
    claims_found: int | None = None
    progress_percentage: int = 0
