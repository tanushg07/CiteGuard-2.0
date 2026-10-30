"""Model registry: load once, prove it works by running inference, remember why it failed.

Importing this module imports nothing heavy. Loaders and probes are injected, which is
what makes the failure paths unit-testable without torch/transformers.

A model is READY only if its loader returned AND its probe ran real inference and
produced sane output. "Loaded" is never reported as "works".
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


class ModelUnavailable(RuntimeError):
    def __init__(self, status: ModelStatus):
        super().__init__(f'{status.name} unavailable: {status.cause}: {status.error}')
        self.status = status


@dataclass
class ModelStatus:
    name: str
    model_id: str
    status: str = 'NOT_CHECKED'          # NOT_CHECKED | READY | UNAVAILABLE
    stage: str | None = None             # import | load | inference | None when READY
    cause: str | None = None             # MISSING_PACKAGE | WEIGHTS_NOT_CACHED | NETWORK_BLOCKED | INIT_ERROR | INFERENCE_FAILED | BAD_OUTPUT
    error: str | None = None             # underlying exception text (truncated)
    remedy: str | None = None
    offline_mode: bool = True
    device: str | None = None
    load_seconds: float | None = None
    probe_seconds: float | None = None
    checked_at: str | None = None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def classify_exception(exc: BaseException, offline: bool) -> tuple[str, str]:
    """Return (cause, remedy). Deliberately string-based: HF raises many exception types."""
    text = f'{type(exc).__name__}: {exc}'.lower()
    if isinstance(exc, ImportError):
        return 'MISSING_PACKAGE', 'Install pinned dependencies: pip install -r requirements.txt (see docs/MODELS.md).'
    cache_markers = ('local_files_only', 'not found in cache', 'cannot find the requested files',
                     'offline mode', 'does not appear to have a file named', 'couldn\'t find', 'no such file')
    if any(m in text for m in cache_markers) or (offline and isinstance(exc, OSError)):
        return 'WEIGHTS_NOT_CACHED', ('Weights are not in the local cache and downloads are disabled. Run '
                                      '`python scripts/prepare_models.py` once with network access, using the same '
                                      'HF_HOME / MODEL_CACHE_DIR as the server.')
    net_markers = ('connection', 'timed out', 'timeout', 'name or service', 'max retries', 'ssl', 'proxy',
                   '403', '429', '503')
    if any(m in text for m in net_markers):
        return 'NETWORK_BLOCKED', 'Model download failed on the network. Pre-download the weights on a connected machine and copy the cache.'
    return 'INIT_ERROR', 'Model initialisation raised an unexpected error; see the error text.'


@dataclass
class ModelSpec:
    name: str
    model_id: str
    loader: Callable[[], object]
    probe: Callable[[object], dict | None]   # runs real inference; raises or returns details; must raise on bad output


class ModelRegistry:
    """Thread-safe, load-once. Failures are cached too (no retry storm) until refresh()."""

    def __init__(self, specs: list[ModelSpec], offline: bool = True, clock: Callable[[], float] = time.monotonic):
        self._specs = {s.name: s for s in specs}
        self._offline = offline
        self._clock = clock
        self._models: dict[str, object] = {}
        self._status: dict[str, ModelStatus] = {s.name: ModelStatus(s.name, s.model_id, offline_mode=offline) for s in specs}
        self._lock = threading.RLock()
        self.load_attempts: dict[str, int] = {s.name: 0 for s in specs}

    def set_offline(self, offline: bool) -> None:
        with self._lock:
            self._offline = offline
            for st in self._status.values():
                st.offline_mode = offline

    def status(self, name: str) -> ModelStatus:
        return self._status[name]

    def all_status(self) -> dict[str, dict]:
        return {n: s.as_dict() for n, s in self._status.items()}

    def get(self, name: str):
        """Return the ready model or raise ModelUnavailable. Initialises at most once."""
        with self._lock:
            st = self._status[name]
            if st.status == 'NOT_CHECKED':
                self._initialise(name)
            if st.status != 'READY':
                raise ModelUnavailable(st)
            return self._models[name]

    def check_all(self) -> dict[str, dict]:
        for name in self._specs:
            try:
                self.get(name)
            except ModelUnavailable:
                pass
        return self.all_status()

    def refresh(self, name: str | None = None) -> dict[str, dict]:
        with self._lock:
            for n in ([name] if name else list(self._specs)):
                self._models.pop(n, None)
                self._status[n] = ModelStatus(n, self._specs[n].model_id, offline_mode=self._offline)
        return self.check_all()

    def _initialise(self, name: str) -> None:
        spec, st = self._specs[name], self._status[name]
        st.checked_at = datetime.now(UTC).isoformat(timespec='seconds')
        self.load_attempts[name] += 1
        t0 = self._clock()
        try:
            model = spec.loader()
        except BaseException as exc:  # noqa: BLE001 - report every failure mode, including SystemExit from libs
            if isinstance(exc, KeyboardInterrupt):
                raise
            cause, remedy = classify_exception(exc, self._offline)
            self._fail(st, 'import' if isinstance(exc, ImportError) else 'load', cause, exc, remedy)
            return
        st.load_seconds = round(self._clock() - t0, 3)
        t1 = self._clock()
        try:
            details = spec.probe(model) or {}
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            cause = 'BAD_OUTPUT' if isinstance(exc, AssertionError) else 'INFERENCE_FAILED'
            self._fail(st, 'inference', cause, exc,
                       'The model loaded but did not pass its inference check; it will not be used for verification.')
            return
        st.probe_seconds = round(self._clock() - t1, 3)
        st.details.update(details)
        st.device = str(details.get('device')) if details.get('device') else st.device
        st.status, st.stage, st.cause, st.error, st.remedy = 'READY', None, None, None, None
        self._models[name] = model

    @staticmethod
    def _fail(st: ModelStatus, stage: str, cause: str, exc: BaseException, remedy: str) -> None:
        st.status, st.stage, st.cause, st.remedy = 'UNAVAILABLE', stage, cause, remedy
        st.error = f'{type(exc).__name__}: {exc}'[:500]


# ---- NLI output normalisation (pure) -------------------------------------------------------

CANONICAL = ('ENTAILMENT', 'CONTRADICTION', 'NEUTRAL')


def normalize_nli_scores(raw: list[dict], id2label: dict | None = None,
                         derived_map: dict[str, str] | None = None) -> tuple[dict[str, float] | None, str]:
    """Map raw pipeline labels to ENTAILMENT/CONTRADICTION/NEUTRAL.

    Returns (scores, source). scores is None when the mapping cannot be trusted - callers must then
    treat NLI as unavailable instead of guessing which generic LABEL_n means entailment.
    """
    out: dict[str, float] = {}
    source = 'native'
    for pred in raw:
        label = str(pred['label']).upper()
        if label.startswith('LABEL_'):
            idx = int(label.split('_')[-1])
            mapped = str((id2label or {}).get(idx, (id2label or {}).get(str(idx), label))).upper()
            if mapped.startswith('LABEL_'):
                mapped = (derived_map or {}).get(label, mapped)
                source = 'probe_derived'
            else:
                source = 'config'
            label = mapped
        out[label] = float(pred['score'])
    if set(out) != set(CANONICAL):
        return None, 'unmapped'
    return out, source


def derive_label_map(probe_scores: dict[str, list[dict]]) -> dict[str, str] | None:
    """probe_scores: {'ENTAILMENT': raw_preds_for_clear_entailment_pair, 'CONTRADICTION': ..., 'NEUTRAL': ...}.

    Used only when the model config exposes generic LABEL_n names. Succeeds only if the three
    behavioural probes pick three distinct labels; otherwise None (=> model rejected).
    """
    chosen = {}
    for expected, raw in probe_scores.items():
        chosen[expected] = max(raw, key=lambda p: p['score'])['label'].upper()
    if len(set(chosen.values())) != 3:
        return None
    return {raw_label: expected for expected, raw_label in chosen.items()}
