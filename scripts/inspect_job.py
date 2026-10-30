"""Print a diagnostic summary of a report: data/jobs/<job_id>.json (API state) or an e2e report JSON.

Usage: python scripts/inspect_job.py <report.json> [--claims] [--refs] [--evidence N]
"""
import argparse
import json
from collections import Counter


def load(path):
    data = json.loads(open(path, encoding='utf-8').read())
    return data['data'] if 'data' in data and 'claims' in data['data'] else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--claims', action='store_true')
    ap.add_argument('--refs', action='store_true')
    ap.add_argument('--evidence', type=int, default=0, help='print the top passage for the first N pairs with evidence')
    a = ap.parse_args()
    r = load(a.path)
    s = r['summary']
    st = s.get('stats', {})
    print(f"document {s['document_name']}: {s.get('total_pages')} pages, {s.get('total_references')} references, "
          f"{s['processing_time_seconds']}s")
    print('timings', json.dumps(r.get('timings'), indent=None))
    print('model usage', json.dumps(r.get('model_usage'), indent=None)[:700])
    for k in ('distinct_cited_sentences', 'atomic_claims', 'claim_citation_pairs', 'pairs_source_identified',
              'pairs_full_text_retrieved', 'pairs_with_evidence_passages', 'pairs_with_full_text_passages',
              'pairs_semantic_verification_completed', 'supported', 'partially_supported', 'contradicted',
              'needs_human_review_verdicts', 'unresolved', 'review_required'):
        print(f'  {k}: {st.get(k)}')
    print('  verdicts', st.get('verdict_counts'))
    print('  failure stages', st.get('failure_stage_breakdown'))
    print('  support rate', st.get('support_rate', {}).get('numerator'), '/', st.get('support_rate', {}).get('denominator'))
    refs = st.get('references', {})
    print('  references', {k: refs.get(k) for k in ('total_references', 'identified', 'identified_high_confidence',
                                                    'identified_needs_review', 'unidentified', 'full_text_retrieved',
                                                    'usable_extracted_text', 'abstract_only')})
    print('  identification', refs.get('identification_breakdown'))
    print('  full text', refs.get('full_text_breakdown'))
    for w in r.get('warnings', []):
        print('  WARNING', w[:200])
    if a.refs:
        for ref in sorted(r.get('references', []), key=lambda x: int((x.get('raw_reference') or '[999]')[1:].split(']')[0] or 999)):
            print(f"{(ref['raw_reference'] or '')[:5]:6} {ref['identification_status']:28} {ref['status']:22} "
                  f"{(ref.get('full_text_status') or ''):24} {(ref.get('full_text_provider') or '')[:30]:30} "
                  f"{(ref.get('title') or '')[:50]}")
            if ref['status'] != 'FULL_TEXT':
                print('        ', (ref.get('full_text_failure_reason') or ref.get('reason') or '')[:220])
    if a.claims:
        for c in r['claims']:
            nli = c.get('nli') or {}
            print(f"{c['atomic_claim_id']:11} {c['citation_marker']:5} {c.get('association_type', ''):17} "
                  f"{c['final_verdict']:28} {c.get('reason_code', ''):22} tier={c.get('evidence_tier')} "
                  f"nli={nli.get('decision_label')} {json.dumps(nli.get('scores')) if nli.get('scores') else ''}")
            print('      ', c['text'][:150])
    shown = 0
    for c in r['claims']:
        if shown >= a.evidence or not c.get('evidence_list'):
            continue
        shown += 1
        chosen = next((e for e in c['evidence_list'] if e.get('evidence_id') == (c.get('nli') or {}).get('chosen_evidence_id')),
                      c['evidence_list'][0])
        print(f"\n=== {c['atomic_claim_id']} {c['citation_marker']} -> {c['final_verdict']} ({c.get('reason_code')})")
        print('CLAIM   :', c['text'])
        print('SOURCE  :', chosen.get('source_document'), '| page', chosen.get('page_number'), '|', chosen.get('section'),
              '| tier', chosen.get('evidence_tier'), '| method', chosen.get('extraction_method'))
        print('PASSAGE :', chosen.get('evidence_text')[:700])
        print('RANKING : lexical', round(chosen.get('retrieval_score') or 0, 3), 'rank', chosen.get('lexical_rank'),
              '| cross-encoder', chosen.get('rerank_score'), 'rank', chosen.get('rerank_rank'), 'of', chosen.get('candidate_count'))
        print('NLI     :', json.dumps((chosen.get('nli') or {}).get('scores')))
    print('\nverdict counter', Counter(c['final_verdict'] for c in r['claims']))


if __name__ == '__main__':
    main()
