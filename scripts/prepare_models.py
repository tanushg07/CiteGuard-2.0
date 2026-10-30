"""One-time model preparation. Run with network access; the server itself runs offline.

    python scripts/prepare_models.py              # download, probe with real inference, then re-verify OFFLINE
    python scripts/prepare_models.py --check      # only verify the already-cached models offline (no network needed)

Exit code 0 only if BOTH models load and pass their inference probes in offline mode - the condition the
server runs under. Set the same HF_HOME / MODEL_CACHE_DIR here and for the server, or the server will
report WEIGHTS_NOT_CACHED. See docs/MODELS.md.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def report(health: dict) -> bool:
    ok = True
    for name, st in health.items():
        if st['status'] == 'READY':
            print(f"  [READY]       {name:<14} device={st.get('device')} load={st.get('load_seconds')}s probe={st.get('probe_seconds')}s "
                  f"{('labels=' + str(st['details'].get('label_mapping'))) if st.get('details', {}).get('label_mapping') else ''}")
        else:
            ok = False
            print(f"  [UNAVAILABLE] {name:<14} cause={st.get('cause')} stage={st.get('stage')}\n"
                  f"                error : {st.get('error')}\n                remedy: {st.get('remedy')}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='verify cached models offline; do not download')
    args = parser.parse_args()

    from app.core import models
    if not args.check:
        os.environ['CITEGUARD_ALLOW_DOWNLOAD'] = '1'
        models.reset_registry_for_tests()
        print('Step 1/2: downloading (if needed) and probing models with real inference...')
        if not report(models.model_health(refresh=True)):
            print('\nFAILED: a model could not be downloaded, loaded, or passed its inference probe.')
            return 1
    os.environ['CITEGUARD_ALLOW_DOWNLOAD'] = '0'
    models.reset_registry_for_tests()
    print('Step 2/2: re-verifying in OFFLINE mode (what the server will do)...' if not args.check else 'Verifying cached models offline...')
    if not report(models.model_health(refresh=True)):
        print('\nFAILED: models are not usable offline. Check that HF_HOME / MODEL_CACHE_DIR match the server.')
        return 1
    print('\nOK: both models loaded offline and passed inference probes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
