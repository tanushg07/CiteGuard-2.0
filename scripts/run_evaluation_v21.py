"""Measure CiteGuard v2.1 against evaluation/v21_labelled_set.json and write evaluation/v21_results.json.

    python scripts/run_evaluation_v21.py [--report data/e2e-after-cold.json] [--skip-network]

Parts
  A  source identification precision / coverage (LIVE providers, identification cache bypassed) incl. DOI-stripped
  B  claim decomposition quality and citation association accuracy (offline)
  C  support precision / recall / false-positive rate on the synthetic 15-case benchmark (offline, real models)
  D  the same metrics on real-paper claim/source pairs (sources via the resolver; real models)
  E  full-text retrieval rate and evidence coverage from an end-to-end report
Labels are agent-assigned (see label_provenance in the dataset). The sets are tiny: do not quote these numbers as
accuracy claims.
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.claims import predicate_components  # noqa: E402
from app.evaluation.metrics import compute  # noqa: E402
from app.services.extractor import Extractor  # noqa: E402
from app.services.identification import IDENTIFIED, ReferenceIdentifier, norm_title, title_similarity  # noqa: E402

DATA = json.loads((ROOT / 'evaluation' / 'v21_labelled_set.json').read_text(encoding='utf-8'))


def ratio(n, d):
    return round(n / d, 4) if d else None


def strip_doi(raw):
    return re.sub(r'\s*doi:\s*10\.\S+', '', raw).strip()


def correct(item, best):
    if not best:
        return False
    if item.get('gold_doi'):
        return (best.get('doi') or '').lower() == item['gold_doi'].lower()
    if item.get('gold_arxiv'):
        return re.sub(r'v\d+$', '', best.get('arxiv_id') or '') == item['gold_arxiv']
    return title_similarity([item['gold_title']], best.get('title') or '') >= .9 or \
        norm_title(item['gold_title']) in norm_title(best.get('title') or '')


async def part_a():
    rows = []
    async with ReferenceIdentifier(cache_dir=False) as ri:
        jobs = []
        for item in DATA['references']:
            jobs.append(('as_printed', item, item['raw']))
            if item.get('gold_doi'):
                jobs.append(('doi_stripped', item, strip_doi(item['raw'])))
        started = time.monotonic()
        results = await asyncio.gather(*(ri.identify(raw) for _, _, raw in jobs))
        seconds = round(time.monotonic() - started, 2)
    for (variant, item, raw), res in zip(jobs, results):
        rows.append({'key': item['key'], 'variant': variant, 'status': res.status, 'correct': correct(item, res.best),
                     'predicted_doi': (res.best or {}).get('doi'), 'predicted_title': (res.best or {}).get('title'),
                     'reason': res.reason[:200]})
    out = {'seconds_wall_clock': seconds, 'rows': rows}
    for variant in ('as_printed', 'doi_stripped'):
        sub = [r for r in rows if r['variant'] == variant]
        ident = [r for r in sub if r['status'] in IDENTIFIED]
        high = [r for r in sub if r['status'] == 'IDENTIFIED_HIGH_CONFIDENCE']
        out[variant] = {'n': len(sub), 'identified': len(ident), 'correct': sum(r['correct'] for r in ident),
                        'identification_precision': ratio(sum(r['correct'] for r in ident), len(ident)),
                        'high_confidence_precision': ratio(sum(r['correct'] for r in high), len(high)),
                        'coverage': ratio(sum(r['correct'] for r in ident), len(sub)),
                        'status_counts': {s: sum(r['status'] == s for r in sub) for s in sorted({r['status'] for r in sub})}}
    return out


def part_b():
    norm = lambda t: ' '.join(t.lower().replace('–', '-').split()).rstrip('.')  # noqa: E731
    report = {}
    for subset in ('development', 'held_out'):
        items = [d for d in DATA['decomposition'] if d['set'] == subset]
        count_ok = gold_props = pred_props = matched = assoc_total = assoc_ok = 0
        details = []
        for item in items:
            pairs = Extractor().extract([{'text': item['sentence'], 'page_number': 1}], enrich=False, total_pages=1)
            predicted = {}
            for p in pairs:
                predicted.setdefault(p.atomic_claim_id, {'text': p.text, 'citations': {}})['citations'][p.citation_marker] = p.association_type
            preds = list(predicted.values())
            count_ok += len(preds) == len(item['propositions'])
            gold_props += len(item['propositions'])
            pred_props += len(preds)
            for g in item['propositions']:
                hit = next((p for p in preds if norm(p['text']) == norm(g['text'])), None)
                matched += hit is not None
                for marker, gold_type in g['citations'].items():
                    assoc_total += 1
                    got = (hit or {}).get('citations', {}).get(marker)
                    ok = got in ('ambiguous_scope', 'multi_proposition') if gold_type == 'ambiguous' else got == gold_type
                    assoc_ok += bool(ok)
            details.append({'id': item['id'], 'gold_propositions': len(item['propositions']), 'predicted_propositions': len(preds),
                            'predicted': preds})
        report[subset] = {'sentences': len(items), 'proposition_count_exact': ratio(count_ok, len(items)),
                          'proposition_precision': ratio(matched, pred_props), 'proposition_recall': ratio(matched, gold_props),
                          'association_accuracy': ratio(assoc_ok, assoc_total), 'association_pairs': assoc_total,
                          'details': details}
    return report


async def part_c():
    from app.services.orchestrator import Orchestrator
    bench = json.loads((ROOT / 'evaluation' / 'citation_benchmark.json').read_text(encoding='utf-8'))
    orch = Orchestrator(use_models=True)
    rows = []
    for case in bench['cases']:
        sources = [{'name': 'bench-source', 'text': case['source_text'], 'markers': ['[1]']}] if case['source_text'] else []
        result = await orch.run_pipeline(target_text=case['claim'], source_texts=sources, document_title=case['id'])
        claim = result.claims[0]
        rows.append({'id': case['id'], 'gold': case['gold'], 'predicted': claim.final_verdict,
                     'semantic_completed': claim.semantic_verification_completed, 'reason_code': claim.reason_code})
    return {'metrics': compute(rows), 'rows': rows, 'model_health': {k: v.get('status') for k, v in result.model_health.items()}}


async def part_d():
    from app.core.schemas import ClaimCitationPair
    from app.services.orchestrator import finalize
    from app.services.retriever import Retriever
    from app.services.source_resolver import SourceResolver
    from app.services.verifier import Verifier
    refs = {r['key']: r['raw'] for r in DATA['references']}
    pairs = [p for p in DATA['support_pairs'] if p['gold'] != 'uncertain']
    needed = sorted({p['reference'] for p in pairs}, key=lambda k: int(k[1:-1]))
    async with SourceResolver() as resolver:   # default caches: re-uses downloads from the end-to-end run
        resolved = dict(zip(needed, await asyncio.gather(*(resolver.resolve(refs[k]) for k in needed))))
    retriever_proto = Retriever(True, 250, 3)
    verifier = Verifier(True)
    rows = []
    for p in pairs:
        record, metadata = resolved[p['reference']]
        passages = record.get('passages') or record.get('abstract_passages') or []
        claim = ClaimCitationPair(id=p['id'], text=p['claim'], citation_marker=p['reference'],
                                  predicate_components=predicate_components(p['claim']))
        retriever = Retriever.__new__(Retriever)
        retriever.__dict__.update(retriever_proto.__dict__)
        retriever.add_sources(passages)
        evidence = retriever.retrieve(claim, 3, 30) if retriever.chunks else []
        result = verifier.verify(claim, evidence, False)
        finalize(result, claim, record)
        if result.reason_code in ('NOT_ENTAILED', 'ABSTRACT_NOT_ENTAILED') and result.evidence_list and claim.predicate_components:
            comps = verifier.component_checks(claim.predicate_components, result.evidence_list)
            finalize(result, claim, record, components=[c['verdict'] for c in comps])
        rows.append({'id': p['id'], 'reference': p['reference'], 'gold': p['gold'] if p['gold'] == 'SUPPORTED' else 'INSUFFICIENT',
                     'predicted': result.final_verdict, 'semantic_completed': result.semantic_verification_completed,
                     'reason_code': result.reason_code, 'evidence_tier': result.evidence_tier, 'source_status': record['status'],
                     'decisive_passage': (result.evidence or '')[:300]})
    return {'metrics': compute(rows), 'rows': rows}


def part_e(report_path):
    if not report_path or not Path(report_path).exists():
        return None
    data = json.loads(Path(report_path).read_text(encoding='utf-8'))
    data = data.get('data', data)
    st = data['summary']['stats']
    refs = st['references']
    return {'report': str(report_path), 'processing_time_seconds': data['summary']['processing_time_seconds'],
            'full_text_retrieval_rate_references': ratio(refs['full_text_retrieved'], refs['references_resolved']),
            'identified_rate_references': ratio(refs['identified'], refs['references_resolved']),
            'evidence_coverage_pairs': st['evidence_coverage'], 'full_text_evidence_coverage_pairs': st['full_text_evidence_coverage'],
            'unresolved_rate_pairs': st['unresolved_rate'], 'support_rate': st['support_rate'],
            'verdict_counts': st['verdict_counts']}


async def main(args):
    out = {'dataset': 'evaluation/v21_labelled_set.json', 'caveat': 'Tiny agent-labelled sets; regression tracking only.',
           'label_provenance': DATA['label_provenance']}
    if not args.skip_network:
        out['A_identification'] = await part_a()
    out['B_decomposition_and_association'] = part_b()
    out['C_support_synthetic_benchmark'] = await part_c()
    if not args.skip_network:
        out['D_support_real_paper_pairs'] = await part_d()
    out['E_end_to_end'] = part_e(args.report)
    path = ROOT / 'evaluation' / 'v21_results.json'
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    summary = {k: ({kk: vv for kk, vv in v.items() if kk not in ('rows', 'details')} if isinstance(v, dict) else v)
               for k, v in out.items() if k not in ('label_provenance',)}
    for key in ('B_decomposition_and_association',):
        summary[key] = {s: {kk: vv for kk, vv in v.items() if kk != 'details'} for s, v in out[key].items()}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print('wrote', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='data/e2e-after-cold.json')
    ap.add_argument('--skip-network', action='store_true')
    asyncio.run(main(ap.parse_args()))
