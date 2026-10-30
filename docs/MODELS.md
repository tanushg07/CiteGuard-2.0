# Models: setup, readiness, offline use

CiteGuard uses two local models. Neither is required to *run* the app, but without **both** it cannot
verify support (see `docs/VERIFICATION_POLICY.md`).

| Role | Model id | Used for |
|---|---|---|
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking candidate passages (relevance only) |
| `nli` | `typeform/distilbert-base-uncased-mnli` | Entailment / contradiction / neutral between passage and claim |

## What "ready" means
A model is `READY` only after it loads **and** passes a real inference probe
(`app/core/models.py`): the cross-encoder must rank a relevant passage above an irrelevant one; the NLI model must
classify six hand-written entailment/contradiction/neutral pairs correctly. If the checkpoint exposes only generic
`LABEL_0..2` names, the label mapping is derived from behavioural probes and verified on a second set; if that is not
possible the model is rejected (we never guess which label means "entailment").

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # requirements-lock.txt is now UTF-8 (it was UTF-16 and unreadable by pip on Linux/macOS)
python scripts/prepare_models.py         # needs network ONCE: downloads, probes, then re-verifies OFFLINE
python scripts/prepare_models.py --check # later: confirm the cache still works, no network needed
uvicorn app.main:app --port 8000
curl localhost:8000/api/models/health    # loads + probes (once) and reports status; ?refresh=true retries
curl localhost:8000/api/health           # current state, includes `verification_ready`
```
`prepare_models.py` exits non-zero unless both models pass **in offline mode**, which is how the server runs.

## Failure causes (reported in `/api/models/health`, the dashboard warning, and the log)
| `cause` | Meaning | Fix |
|---|---|---|
| `MISSING_PACKAGE` | torch / transformers / sentence-transformers not importable | `pip install -r requirements.txt` in the **server's** environment |
| `WEIGHTS_NOT_CACHED` | downloads disabled and weights not in the cache the server reads | run `prepare_models.py`; make `HF_HOME` / `MODEL_CACHE_DIR` identical for the script and the server |
| `NETWORK_BLOCKED` | download attempted (`CITEGUARD_ALLOW_DOWNLOAD=1`) but the network failed | pre-download elsewhere and copy the cache (below) |
| `INIT_ERROR` | unexpected exception while loading (version mismatch, corrupt files, CUDA) | read the `error` field |
| `INFERENCE_FAILED` / `BAD_OUTPUT` | loaded, but probe raised / gave wrong answers | the model is **not used**; check versions against the lock file |

## Offline / air-gapped
1. On a connected machine: `MODEL_CACHE_DIR=/path/cache python scripts/prepare_models.py`
2. Copy `/path/cache` to the target; set the same `MODEL_CACHE_DIR` (or `HF_HOME`); leave `CITEGUARD_ALLOW_DOWNLOAD=0`.
3. `python scripts/prepare_models.py --check` on the target.

Models load once per process; failures are remembered (no retry storm). `GET /api/models/health?refresh=true`.

## Readiness versus actual inference (v2.1)
`READY` only proves the probe passed at load time. Every report also contains `model_usage`, which records what the models
did in **that run**: NLI `state` (`MODEL_LOADED` → `INFERENCE_STARTED` → `INFERENCE_COMPLETED` / `INFERENCE_FAILED`),
inference calls, passages scored, failures and seconds; cross-encoder rerank calls, reranked pairs and failures. A run in
which no passage was retrieved stays `MODEL_LOADED`, so a READY model can no longer be mistaken for completed verification.
Per passage, `evidence_list[i].nli` stores the model id, raw label scores, mapped entailment/neutral/contradiction scores,
reverse-direction contradiction (for contradiction candidates), processing time and any error. NLI runs batched per claim.
`CITEGUARD_TORCH_THREADS` sets CPU threads (default: min(8, CPU count)).

## OCR (optional)
Image-only PDFs fall back to OCR when an engine is installed; none is in `requirements-lock.txt`:
```bash
pip install rapidocr_onnxruntime      # CPU, no system binary   (or: pip install pytesseract + a tesseract binary on PATH)
```
Without an engine, such PDFs fail with `OCR_UNAVAILABLE` (extraction failure, never "empty source"). OCR passages are tagged
`extraction_method=ocr:<engine>`. `OCR_MODE=off` disables the fallback; `OCR_MAX_PAGES` bounds the work.
