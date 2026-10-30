"""Run the labelled benchmark through the real pipeline and report metrics.

    python scripts/run_benchmark.py --mode baseline    # no models: shows the safe-fallback regime
    python scripts/run_benchmark.py --mode neural      # requires prepared models (see docs/MODELS.md)

Metrics are only meaningful for the mode actually run. The 15 cases are far too few to claim accuracy.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def main(mode: str) -> int:
    from app.evaluation.metrics import compute
    from app.services.orchestrator import Orchestrator
    bench = json.loads((ROOT / 'evaluation' / 'citation_benchmark.json').read_text(encoding='utf-8'))
    orch = Orchestrator(use_models=(mode == 'neural'))
    rows, health = [], None
    for case in bench['cases']:
        sources = [{'name': 'bench-source', 'text': case['source_text'], 'markers': ['[1]']}] if case['source_text'] else []
        result = await orch.run_pipeline(target_text=case['claim'], source_texts=sources, document_title=case['id'])
        health = result.model_health
        claim = result.claims[0]
        rows.append({'id': case['id'], 'gold': case['gold'], 'predicted': claim.final_verdict,
                     'semantic_completed': claim.semantic_verification_completed, 'reason_code': claim.reason_code})
    out = {'mode': mode, 'model_health': health, 'metrics': compute(rows), 'rows': rows,
           'caveat': 'n=15 synthetic cases; not statistically meaningful.'}
    path = ROOT / 'evaluation' / f'benchmark_results_{mode}.json'
    path.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out['metrics'], indent=1))
    print('wrote', path)
    if mode == 'neural' and any(m['status'] != 'READY' for m in (health or {}).values()):
        print('WARNING: models were not READY; this run does not measure neural verification.')
        return 2
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['baseline', 'neural'], default='neural')
    sys.exit(asyncio.run(main(p.parse_args().mode)))
