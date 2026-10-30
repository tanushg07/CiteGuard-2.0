export type Status = 'Supported' | 'Contradicted' | 'Unrelated' | 'Numerical Mismatch';

export interface PassageNLI {
  status?: string; operational_state?: string; model?: string; decision_label?: string; label?: string;
  scores?: Record<string, number>; entailment?: number; neutral?: number; contradiction?: number;
  reverse_contradiction?: number | null; gate?: string; seconds?: number; reason?: string; truncated?: boolean;
}

export interface RetrievedEvidenceItem {
  evidence_id?: string; source_id?: string; source_url?: string; page_number?: number;
  section?: string; paragraph?: number; retrieval_score?: number; rerank_score?: number | null;
  ranking_method?: string; lexical_rank?: number | null; rerank_rank?: number | null; candidate_count?: number | null;
  context?: string | null; evidence_tier?: 'full_text' | 'abstract'; extraction_method?: string | null;
  nli?: PassageNLI | null;
  claim_id: string;
  source_document: string;
  evidence_text: string;
  relevance_score: number;
  source_citation?: string;
}

export interface NumericalComparison {
  status?: string; calculation?: string | null;
  has_numerical_data: boolean;
  claim_entities: string[];
  evidence_entities: string[];
  is_match: boolean | null;
  details: string | null;
}

export interface RetrievalAttempt {
  provider: string; url: string; outcome: string; reason?: string; http_status?: number | null;
  timestamp?: string; attempt_number?: number; duration_seconds?: number | null; stage?: string;
}

export interface Rate { numerator: number; denominator: number; denominator_definition?: string; percent: number | null }

export interface ReferenceStats {
  total_references: number; references_resolved: number; references_not_cited_or_not_matched: number;
  identified: number; identified_high_confidence: number; identified_needs_review: number; unidentified: number;
  identification_breakdown: Record<string, number>; full_text_retrieved: number; usable_extracted_text: number;
  abstract_only: number; full_text_breakdown: Record<string, number>; extraction_failures: number;
  duplicate_documents: number; attempts_total: number; attempt_outcomes: Record<string, number>;
  definitions?: Record<string, string>;
}

export interface SummaryStats {
  distinct_cited_sentences: number; atomic_claims: number; claim_citation_pairs: number;
  pairs_source_identified: number; pairs_full_text_retrieved: number; pairs_with_evidence_passages: number;
  pairs_with_full_text_passages?: number; pairs_with_abstract_evidence_only?: number;
  pairs_semantic_verification_completed: number; pairs_full_text_semantic_completed?: number;
  supported: number; partially_supported: number; contradicted: number; unresolved: number; not_verifiable: number;
  supported_from_abstract_only?: number; needs_human_review_verdicts?: number; insufficient_evidence?: number;
  retrieval_failures: number; model_failures: number; review_required: number;
  failure_stage_breakdown?: Record<string, number>;
  semantic_outcomes?: Record<string, number>;
  sentence_rollup?: Record<string, number>; proposition_rollup?: Record<string, number>;
  support_rate: Rate; evidence_coverage: Rate; full_text_evidence_coverage?: Rate; unresolved_rate: Rate;
  definitions?: Record<string, string>;
  references?: ReferenceStats;
  confidence_note?: string; consistency_error?: string;
}

export interface ReferenceRow {
  source_id: string; raw_reference: string; status: string; failure_stage?: string | null; reason?: string;
  identification_status?: string; identification_reason?: string;
  match_score?: Record<string, unknown> | null; alternatives?: {title?: string; doi?: string; year?: number; score?: number}[];
  title?: string | null; doi?: string | null; arxiv_id?: string | null; year?: number | null; venue?: string | null;
  authors?: string[]; provider?: string | null;
  full_text_status?: string | null; full_text_provider?: string | null; full_text_url?: string | null;
  full_text_format?: string | null; full_text_version?: string | null;
  identity_validation?: {valid?: boolean; method?: string; reason?: string; checks?: Record<string, unknown>} | null;
  extraction?: {status?: string; method?: string; words?: number; paragraphs?: number; cause?: string} | null;
  full_text_failure_reason?: string | null; duplicate_of?: string | null;
  passages: number; abstract_available: boolean; attempts: number; timings?: Record<string, number>;
}

export interface Claim {
  final_verdict?: string; claim_group_id?: string; citation_id?: string; section?: string;
  association_confidence?: string; decision_reason?: string;
  source_identification?: {status: string; reason: string; url?: string; failure_stage?: string | null;
    identification?: {status?: string; reason?: string; score?: Record<string, unknown>};
    full_text?: {status?: string; provider?: string; url?: string; format?: string; version?: string | null; failure_reason?: string | null;
      identity?: {valid?: boolean; method?: string; reason?: string}; extraction?: {method?: string; words?: number}};
    [key: string]: unknown};
  nli?: {status: string; operational_state?: string; model?: string; label?: string | null; score?: number | null;
    decision_label?: string | null; reason?: string; cause?: string; scores?: Record<string, number>;
    passages_scored?: number; passages_examined?: number; chosen_evidence_id?: string; contradiction_grounded?: boolean;
    gated_out?: {evidence_id?: string; label?: string; gate?: string}[];
    components?: {text: string; verdict: string; scores?: Record<string, number>; evidence_id?: string}[]};
  clause_text?: string; decomposition_method?: string; clause_relation?: string; decomposition_rationale?: string;
  decomposition_uncertain?: boolean; predicate_components?: string[];
  association_type?: string; ambiguity_status?: string; failure_stage?: string | null; review_reasons?: string[];
  evidence_tier?: string | null;
  reason_code?: string; review_required?: boolean; limitations?: string[]; disagreements?: string[];
  semantic_verification_completed?: boolean; evidence_state?: 'NONE' | 'PASSAGES'; next_action?: string;
  retrieval_attempts?: RetrievalAttempt[]; confidence_kind?: string;
  sentence_id?: string; sentence_text?: string; atomic_claim_id?: string; association_rationale?: string;
  resolved_reference_id?: string | null; page_mapping?: string; manuscript_id?: string;
  reasoning?: string;
  reasoning_provider?: 'template' | 'groq';
  reference_metadata?: {
    raw_reference: string;
    provider: string;
    status: string;
    identification_status?: string | null;
    arxiv_id?: string | null;
    volume?: string | null; issue?: string | null; pages?: string | null;
    title?: string | null;
    doi?: string | null;
    abstract?: string | null;
    venue?: string | null;
    authors: string[];
    year?: number | null;
  } | null;
  id: string;
  text: string;
  context: string;
  citation_marker: string;
  source_document: string;
  evidence: string;
  evidence_list?: RetrievedEvidenceItem[];
  status: Status;
  confidence: number | null;
  numerical_check: string | null;
  numerical_comparison?: NumericalComparison;
  page_number: number | null;
}

export interface DocumentSummary {
  total_pages?: number; total_references?: number; unique_claims?: number;
  verdict_counts?: Record<string, number>;
  document_name: string;
  total_claims: number;
  supported_count: number;
  contradicted_count: number;
  unrelated_count: number;
  numerical_mismatch_count: number;
  supported_percentage: number | null;
  average_confidence: number | null;
  processing_time_seconds: number;
  stats?: SummaryStats;
}

export interface AnalysisResponse {
  document?: { metadata?: {title?: string; authors?: string[]; abstract?: string};
    sentences?: unknown[]; sections?: {title: string; page: number}[] };
  warnings?: string[];
  engines?: Record<string, string>;
  model_health?: Record<string, {status: string; cause?: string | null; stage?: string | null; error?: string | null; remedy?: string | null; device?: string | null}>;
  model_usage?: Record<string, {state?: string; model?: string | null; health_status?: string; inference_calls?: number;
    passages_scored?: number; failures?: number; seconds?: number; rerank_calls?: number; rerank_pairs?: number; rerank_failures?: number}>;
  references?: ReferenceRow[];
  timings?: Record<string, number>;
  job_id: string;
  summary: DocumentSummary;
  claims: Claim[];
  status: string;
}

export interface PipelineStep {
  id: number;
  name: string;
  status: 'pending' | 'loading' | 'complete' | 'error';
  detail?: string;
  claims_found?: number;
}
