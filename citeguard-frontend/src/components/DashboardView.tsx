import { useMemo, useState } from 'react';
import type { AnalysisResponse } from '../types';
import ClaimCard, { OverlapText } from './ClaimCard';
import DocumentSummary from './DocumentSummary';
import { api } from '../services/api';

const label = (v?: string | null) => (v || '').toLowerCase().replaceAll('_', ' ');
const fmt = (n?: number | null, d = 2) => (n == null ? '—' : n.toFixed(d));

export default function DashboardView({data, onReset}: {data: AnalysisResponse; onReset: () => void}) {
  const [selected, setSelected] = useState(data.claims[0]?.id || '');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('ALL');
  const claims = useMemo(() => data.claims.filter(c => (filter === 'ALL' || (filter === 'REVIEW' ? c.review_required : c.final_verdict === filter)) &&
    `${c.text} ${c.citation_marker} ${c.reference_metadata?.title || ''}`.toLowerCase().includes(query.toLowerCase())), [data.claims, query, filter]);
  const claim = claims.find(c => c.id === selected) || claims[0];
  const source = claim?.source_identification;
  const ident = source?.identification;
  const ft = source?.full_text;
  const reviewCount = data.claims.filter(c => c.review_required).length;
  return <div className="max-w-screen-2xl mx-auto">
    <div className="flex justify-between px-6 py-3 border-b text-xs border-slate-200">
      <button onClick={onReset} className="cursor-pointer">Upload another paper</button>
      <div className="flex gap-4"><button onClick={() => api.exportResults(data.job_id,'json')}>Export full JSON report</button><button onClick={() => api.exportResults(data.job_id,'csv')}>Export CSV</button></div>
    </div>
    <DocumentSummary data={data} />
    {data.warnings?.map((warning,i) => <p key={i} role="status" className="mx-6 mt-3 p-3 border border-amber-200 bg-amber-50 text-xs text-amber-950">{warning}</p>)}
    <div className="grid lg:grid-cols-[320px_1fr] gap-6 p-6">
      <aside className="border border-slate-200 bg-white self-start">
        <div className="p-3 space-y-3 border-b border-slate-200">
          <input aria-label="Search claims and sources" placeholder="Search claims or sources" value={query} onChange={e => setQuery(e.target.value)} className="w-full border border-slate-300 p-2 text-sm" />
          <select aria-label="Filter verdict" value={filter} onChange={e => setFilter(e.target.value)} className="w-full border border-slate-300 p-2 text-xs">
            <option value="ALL">All pairs ({data.claims.length})</option>
            <option value="REVIEW">Human-review queue ({reviewCount})</option>
            {Object.entries(data.summary.verdict_counts || {}).map(([v, n]) => <option key={v} value={v}>{label(v)} ({n})</option>)}
          </select>
        </div>
        <div className="max-h-[75vh] overflow-y-auto divide-y divide-slate-200">
          {claims.map(c => <ClaimCard key={c.id} claim={c} selected={claim?.id === c.id} onSelect={() => setSelected(c.id)} />)}
          {!claims.length && <p className="p-4 text-sm text-slate-500">{data.claims.length ? 'No claims match these filters.' : 'No supported citation patterns were detected.'}</p>}
        </div>
      </aside>
      {claim && <article className="min-w-0 space-y-5">
        <section className="border border-slate-200 bg-white p-5 space-y-3">
          <div className="flex flex-wrap justify-between gap-3 text-xs text-slate-500"><span>{claim.claim_group_id} · {claim.citation_marker} · Manuscript page {claim.page_number ?? 'unknown'}{claim.page_mapping && claim.page_mapping !== 'ok' ? ` (${label(claim.page_mapping)})` : ''} · {claim.section}</span><strong className="text-slate-900">{label(claim.final_verdict)}</strong></div>
          <p className="text-xs text-slate-600"><span className="font-semibold">1. Original manuscript sentence:</span> {claim.sentence_text || claim.text}</p>
          <p className="text-xs font-semibold text-slate-500">2. Atomic claim assessed ({claim.atomic_claim_id || claim.id})</p>
          <blockquote className="font-serif text-lg leading-relaxed"><OverlapText text={claim.text} other={claim.evidence} /></blockquote>
          <p className="text-xs text-slate-500">Decomposition: {claim.decomposition_rationale || 'single proposition'}{claim.decomposition_uncertain ? ' (uncertain — flagged for review)' : ''}</p>
          {!!claim.predicate_components?.length && <p className="text-xs text-slate-500">Derived components (checked only for partial support): {claim.predicate_components.join(' | ')}</p>}
          <p className="text-xs text-slate-500">Citation association: <span className="font-mono">{label(claim.association_type || claim.association_confidence)}</span> ({label(claim.ambiguity_status)}){claim.association_rationale ? ` — ${claim.association_rationale}` : ''}. Shared terms are highlighted; overlap does not establish support.</p>
          {claim.review_required && <p role="status" className="p-2 border border-amber-300 bg-amber-50 text-xs text-amber-950">In the human-review queue{claim.review_reasons?.length ? `: ${claim.review_reasons.map(label).join('; ')}` : ''}{claim.disagreements?.length ? ` · disagreement: ${claim.disagreements.map(label).join('; ')}` : ''}.</p>}
          <p className="border-t border-slate-200 pt-3 text-sm"><span className="font-semibold">10. Verdict:</span> {claim.decision_reason}</p>
          {claim.failure_stage && <p className="text-xs text-slate-700"><span className="font-semibold">Stopped at stage:</span> {label(claim.failure_stage)}</p>}
          {!!claim.limitations?.length && <p className="text-xs text-slate-700"><span className="font-semibold">12. Limitations:</span> {claim.limitations.map(label).join('; ')}</p>}
          {claim.next_action && <p className="text-xs text-slate-700"><span className="font-semibold">13. Recommended action:</span> {claim.next_action}</p>}
          <details className="text-xs text-slate-500"><summary>Original paragraph</summary><p className="mt-2 leading-relaxed">{claim.context}</p></details>
        </section>
        <section className="border border-slate-200 bg-white p-5 space-y-3">
          <h2 className="font-semibold text-sm">3–4. Reference → identified source</h2>
          <p className="text-sm">{claim.reference_metadata?.raw_reference || 'No unambiguous reference-list entry could be associated with this marker.'}</p>
          <p className="text-xs"><span className="font-mono">{ident?.status || claim.reference_metadata?.identification_status || 'not attempted'}</span> — {ident?.reason || source?.reason}</p>
          {claim.reference_metadata?.title && <>
            <p className="font-medium text-sm">{claim.reference_metadata.title}</p>
            <p className="text-xs text-slate-600">{claim.reference_metadata.authors?.slice(0, 8).join(', ')}{(claim.reference_metadata.authors?.length || 0) > 8 ? ', et al.' : ''} {claim.reference_metadata.year} · {claim.reference_metadata.venue}{claim.reference_metadata.volume ? ` ${claim.reference_metadata.volume}` : ''}{claim.reference_metadata.pages ? `: ${claim.reference_metadata.pages}` : ''} · via {claim.reference_metadata.provider}</p>
          </>}
          <p className="text-xs"><span className="font-semibold">Full text:</span> <span className="font-mono">{ft?.status || 'not attempted'}</span>{ft?.provider ? ` from ${ft.provider}` : ''}{ft?.format ? ` (${ft.format}${ft.version ? `, ${ft.version}` : ''})` : ''}{ft?.extraction?.method ? ` · extracted by ${ft.extraction.method}` : ''}{ft?.identity?.reason ? ` · identity: ${ft.identity.reason}` : ''}{ft?.failure_reason ? ` · ${ft.failure_reason}` : ''}</p>
          <div className="flex flex-wrap gap-4 text-xs text-teal-800 underline">
            {claim.reference_metadata?.doi && <a href={`https://doi.org/${encodeURIComponent(claim.reference_metadata.doi)}`} target="_blank" rel="noreferrer">DOI: {claim.reference_metadata.doi}</a>}
            {source?.url && /^https?:\/\//.test(source.url) && <a href={source.url} target="_blank" rel="noreferrer">Open retrieved source</a>}
          </div>
          <details className="text-xs" open={!claim.evidence_list?.length}><summary className="cursor-pointer">Retrieval attempts ({claim.retrieval_attempts?.length || 0})</summary>
            {claim.retrieval_attempts?.length ? <table className="w-full mt-2 text-left"><thead><tr><th>Stage</th><th>Provider</th><th>Outcome</th><th>Detail</th></tr></thead><tbody>{claim.retrieval_attempts.map((a,i) => <tr key={i} className="align-top border-t border-slate-200"><td className="pr-2">{label(a.stage)}</td><td className="pr-2">{a.provider}</td><td className="pr-2 font-mono">{a.outcome}{a.http_status ? ` ${a.http_status}` : ''}</td><td className="break-all">{a.reason}{a.url ? ` — ${a.url}` : ''}</td></tr>)}</tbody></table> : <p className="mt-2 text-slate-600">No retrieval attempts were recorded.</p>}
          </details>
          <details className="text-xs"><summary className="cursor-pointer">Identification and retrieval audit (raw)</summary><pre className="whitespace-pre-wrap break-all mt-2 bg-slate-50 p-3">{JSON.stringify(source, null, 2)}</pre></details>
        </section>
        <section className="border border-slate-200 bg-white p-5 space-y-4">
          <h2 className="font-semibold text-sm">5–9. Source evidence {claim.evidence_tier === 'abstract' ? '(provider abstract only — full text unavailable)' : ''}</h2>
          {!claim.evidence_list?.length && <div role="status" className="border border-dashed border-slate-400 bg-slate-50 p-3 text-sm text-slate-700"><p className="font-semibold">Evidence unavailable — no source passage was examined for this pair.</p><p className="text-xs mt-1">{claim.decision_reason} Bibliographic metadata and generated text are not evidence, and this is not a finding that the citation is wrong.</p></div>}
          {claim.evidence_list?.map((e,i) => {
            const chosen = e.evidence_id && e.evidence_id === claim.nli?.chosen_evidence_id;
            return <div key={`${e.source_id}-${e.evidence_id}-${i}`} className={`border-l-2 pl-4 space-y-2 ${chosen ? 'border-teal-700' : 'border-slate-300'}`}>
              <p className="text-xs font-medium">{chosen ? 'Decisive passage · ' : ''}{e.source_document} · {e.evidence_tier === 'abstract' ? 'Abstract (provider metadata)' : `Source page ${e.page_number ?? 'unknown'} (not the manuscript page)`} · {e.section || 'Section undetected'} · Paragraph {e.paragraph ?? 'unknown'}</p>
              <blockquote className="font-serif text-sm leading-relaxed"><OverlapText text={e.evidence_text} other={claim.text} /></blockquote>
              {e.context && <details className="text-xs text-slate-500"><summary>Surrounding source paragraph</summary><p className="mt-1 leading-relaxed">{e.context}</p></details>}
              <p className="text-xs text-slate-500">{e.evidence_id} · Lexical score {fmt(e.retrieval_score, 3)} (rank {e.lexical_rank ?? '—'}) · Cross-encoder {e.rerank_score == null ? 'unavailable' : `${fmt(e.rerank_score)} (rank ${e.rerank_rank} of ${e.candidate_count})`}. Ranking scores measure relevance, not support.{e.extraction_method ? ` Text extracted by ${e.extraction_method}.` : ''}</p>
              {e.nli && <p className="text-xs text-slate-700">NLI ({e.nli.model}): <span className="font-mono">{e.nli.status === 'SUCCESS' ? e.nli.decision_label : e.nli.status}</span>{e.nli.scores ? ` — entailment ${fmt(e.nli.entailment)}, neutral ${fmt(e.nli.neutral)}, contradiction ${fmt(e.nli.contradiction)}` : ''}{e.nli.reverse_contradiction != null ? ` · reverse contradiction ${fmt(e.nli.reverse_contradiction)}` : ''}{e.nli.gate ? ` · ${e.nli.gate}` : ''} (raw, uncalibrated)</p>}
            </div>;
          })}
        </section>
        <section className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="border border-slate-200 bg-white p-5 space-y-2"><h2 className="font-semibold">9. NLI result</h2>
            <p>{claim.nli?.status}{claim.nli?.operational_state ? ` (${label(claim.nli.operational_state)})` : ''}: {claim.nli?.decision_label || claim.nli?.label || 'not run'}</p>
            {claim.nli?.scores && <p className="text-xs font-mono">{Object.entries(claim.nli.scores).map(([k, v]) => `${k.toLowerCase()} ${v.toFixed(3)}`).join(' · ')}</p>}
            {claim.nli?.passages_examined != null && <p className="text-xs text-slate-600">{claim.nli.passages_scored ?? 0} of {claim.nli.passages_examined} passages scored by {claim.nli.model}.</p>}
            {claim.nli?.status !== 'SUCCESS' && claim.nli?.reason && <p className="text-xs text-amber-900">{claim.nli.reason}</p>}
            {!!claim.nli?.components?.length && <ul className="text-xs list-disc pl-4">{claim.nli.components.map((c, i) => <li key={i}>{c.text} → <span className="font-mono">{label(c.verdict)}</span></li>)}</ul>}
            <p className="text-xs text-slate-500">Model scores are not probabilities that the citation is correct.</p></div>
          <div className="border border-slate-200 bg-white p-5 space-y-2"><h2 className="font-semibold">Numerical verification</h2><p>{claim.numerical_comparison?.status || 'NOT_APPLICABLE'}</p><p className="text-xs text-slate-600">{claim.numerical_comparison?.details}</p>{claim.numerical_comparison?.calculation && <p className="font-mono text-xs">{claim.numerical_comparison.calculation}</p>}</div>
        </section>
        {claim.reasoning && <section className="border-t border-slate-200 pt-4 text-sm text-slate-600"><h2 className="text-xs font-semibold mb-2">11. Explanation · {claim.reasoning_provider === 'groq' ? 'Groq-generated rewording of recorded results, not evidence' : 'Template built only from recorded results'}</h2><p>{claim.reasoning}</p></section>}
      </article>}
    </div>
  </div>;
}
