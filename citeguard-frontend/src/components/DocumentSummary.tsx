import type { AnalysisResponse } from '../types';

const label = (v: string) => v.toLowerCase().replaceAll('_', ' ');

function Row({ name, value, unit, note }: { name: string; value: number | string | null | undefined; unit: string; note?: string }) {
  return <tr className="border-t border-slate-100 align-top">
    <td className="py-1 pr-3">{name}{note && <span className="block text-slate-500">{note}</span>}</td>
    <td className="py-1 pr-3 text-right font-semibold tabular-nums">{value ?? '—'}</td>
    <td className="py-1 text-slate-500">{unit}</td>
  </tr>;
}

function Breakdown({ title, data }: { title: string; data?: Record<string, number> }) {
  const entries = Object.entries(data || {}).filter(([, n]) => n);
  if (!entries.length) return null;
  return <p className="text-slate-600"><span className="font-semibold">{title}:</span> {entries.map(([k, n]) => `${label(k)} ${n}`).join(' · ')}</p>;
}

export default function DocumentSummary({ data }: { data: AnalysisResponse }) {
  const md = data.document?.metadata;
  const st = data.summary.stats;
  const refs = st?.references;
  const rate = st?.support_rate;
  return <section className="border-b border-slate-200 bg-white px-6 py-5" aria-label="Paper overview">
    <h1 className="text-xl font-semibold">{md?.title || data.summary.document_name}</h1>
    {md?.authors?.length ? <p className="text-sm text-slate-600 mt-1">{md.authors.join(', ')}</p> : <p className="text-xs text-slate-500 mt-1">Authors not reliably extracted; consult the original paper.</p>}
    <p className="text-xs text-slate-500 mt-2">{data.summary.total_pages} pages · {data.summary.total_references} references · {data.summary.unique_claims} distinct cited sentences · {data.summary.total_claims} claim–citation pairs · {data.summary.processing_time_seconds}s</p>
    <div className="flex flex-wrap gap-x-5 gap-y-2 mt-4 text-xs" aria-label="Verdict totals (pairs, mutually exclusive)">
      {Object.entries(data.summary.verdict_counts || {}).map(([v, count]) => <span key={v}><strong>{count}</strong> {label(v)}</span>)}
    </div>
    {st && <div className="mt-4 grid lg:grid-cols-3 gap-4 text-xs" aria-label="Evidence coverage and rates">
      {refs && <div className="border border-slate-200 p-3">
        <h2 className="font-semibold mb-1">Sources (unit: unique references)</h2>
        <table className="w-full"><tbody>
          <Row name="Total references" value={refs.total_references} unit="references" note={refs.references_not_cited_or_not_matched ? `${refs.references_not_cited_or_not_matched} not cited or not matched to a marker` : undefined} />
          <Row name="Identified" value={refs.identified} unit="references" note={`${refs.identified_high_confidence} high confidence · ${refs.identified_needs_review} need review`} />
          <Row name="Unidentified" value={refs.unidentified} unit="references" />
          <Row name="Full-text sources retrieved" value={refs.full_text_retrieved} unit="references" />
          <Row name="Sources with usable extracted text" value={refs.usable_extracted_text} unit="references" />
          <Row name="Abstract only (provider metadata)" value={refs.abstract_only} unit="references" />
          <Row name="Provider and download attempts" value={refs.attempts_total} unit="attempts" />
        </tbody></table>
        <div className="mt-2 space-y-1">
          <Breakdown title="Identification" data={refs.identification_breakdown} />
          <Breakdown title="Full text" data={refs.full_text_breakdown} />
        </div>
      </div>}
      <div className="border border-slate-200 p-3">
        <h2 className="font-semibold mb-1">Coverage (each row is a subset of the one above)</h2>
        <table className="w-full"><tbody>
          <Row name="Distinct cited sentences" value={st.distinct_cited_sentences} unit="sentences" />
          <Row name="Atomic claims" value={st.atomic_claims} unit="claims" />
          <Row name="Claim–citation pairs" value={st.claim_citation_pairs} unit="pairs" />
          <Row name="Source identified" value={st.pairs_source_identified} unit="pairs" />
          <Row name="Full text retrieved" value={st.pairs_full_text_retrieved} unit="pairs" />
          <Row name="With full-text evidence passages" value={st.pairs_with_full_text_passages ?? st.pairs_with_evidence_passages} unit="pairs" />
          <Row name="Full-text NLI verification completed" value={st.pairs_full_text_semantic_completed ?? st.pairs_semantic_verification_completed} unit="pairs" />
        </tbody></table>
        <p className="mt-2 text-slate-600">Separately: {st.pairs_with_abstract_evidence_only ?? 0} pairs had only a provider abstract to examine.</p>
      </div>
      <div className="border border-slate-200 p-3">
        <h2 className="font-semibold mb-1">Outcomes (unit: pairs)</h2>
        <table className="w-full"><tbody>
          <Row name="Verified supported" value={st.supported} unit="pairs" note={st.supported_from_abstract_only ? `${st.supported_from_abstract_only} of them from an abstract only` : undefined} />
          <Row name="Verified partially supported" value={st.partially_supported} unit="pairs" />
          <Row name="Verified contradicted" value={st.contradicted} unit="pairs" />
          <Row name="Unresolved" value={st.unresolved} unit="pairs" note="every pair without a verified verdict" />
          <Row name="Human-review queue" value={st.review_required} unit="pairs" note="all unresolved pairs plus flagged verified pairs" />
        </tbody></table>
        <div className="mt-2 space-y-1">
          <Breakdown title="Stopped at stage" data={st.failure_stage_breakdown} />
          <Breakdown title="Semantic outcomes" data={st.semantic_outcomes} />
          <p>Evidence coverage: {st.evidence_coverage.percent ?? 'n/a'}% ({st.evidence_coverage.numerator}/{st.evidence_coverage.denominator} assertion pairs with ≥1 examined passage)</p>
          {rate && (rate.denominator
            ? <p>Support rate: {rate.percent}% ({rate.numerator}/{rate.denominator}) — denominator: {rate.denominator_definition}</p>
            : <p>Support rate: not computed — no full-text pair reached a conclusive NLI outcome. This reflects missing evidence, not citation quality.</p>)}
          <p className="text-slate-500">{st.confidence_note}</p>
        </div>
      </div>
      {st.consistency_error && <p role="alert" className="lg:col-span-3 p-2 border border-red-300 bg-red-50 text-red-900">Internal consistency check failed: {st.consistency_error}. Totals are unvalidated.</p>}
    </div>}
    {data.model_health && <p className="mt-3 text-xs text-slate-600" aria-label="Model readiness">Models: {Object.entries(data.model_health).map(([name, m]) => {
      const use = data.model_usage?.[name];
      const ran = use ? ` · this run: ${label(use.state || 'unknown')}${use.passages_scored != null ? `, ${use.passages_scored} passages scored` : ''}${use.rerank_pairs != null ? `, ${use.rerank_pairs} pairs reranked` : ''}${use.failures ? `, ${use.failures} failures` : ''}` : '';
      return `${name} ${m.status}${m.cause ? ` (${m.cause})` : ''}${ran}`;
    }).join(' | ')}</p>}
    {!!data.references?.length && <details className="mt-4 text-xs text-slate-700"><summary className="cursor-pointer font-semibold">Reference audit ({data.references.length} cited references)</summary>
      <div className="overflow-x-auto"><table className="w-full mt-2 text-left">
        <thead><tr className="text-slate-500"><th className="pr-2">Reference</th><th className="pr-2">Identification</th><th className="pr-2">Identified work</th><th className="pr-2">Full text</th><th>Detail</th></tr></thead>
        <tbody>{data.references.map(r => <tr key={r.source_id} className="border-t border-slate-200 align-top">
          <td className="pr-2 py-1 max-w-xs">{r.raw_reference?.slice(0, 140)}</td>
          <td className="pr-2 py-1 font-mono">{r.identification_status}</td>
          <td className="pr-2 py-1">{r.title}{r.doi ? ` · doi:${r.doi}` : ''}{r.arxiv_id ? ` · arXiv:${r.arxiv_id}` : ''}</td>
          <td className="pr-2 py-1 font-mono">{r.full_text_status}{r.full_text_provider ? ` (${r.full_text_provider}${r.full_text_version ? `, ${r.full_text_version}` : ''})` : ''}{r.passages ? ` · ${r.passages} paragraphs` : ''}</td>
          <td className="py-1">{r.status === 'FULL_TEXT' ? r.identity_validation?.reason : (r.full_text_failure_reason || r.identification_reason || r.reason)}</td>
        </tr>)}</tbody></table></div>
    </details>}
    <details className="mt-4 text-xs text-slate-600"><summary className="cursor-pointer">Whole-paper structure and extraction</summary>
      <p className="mt-2">{data.document?.sentences?.length || 0} sentences extracted. Reference and heading fields use document-layout heuristics.</p>
      <div className="flex flex-wrap gap-3 mt-2">{data.document?.sections?.map((section, i) => <span key={i}>p. {section.page} · {section.title}</span>)}</div>
      {md?.abstract && <p className="mt-3 leading-relaxed">{md.abstract}</p>}
    </details>
  </section>;
}
