"""Lazy, shared, health-checked model loading.

Importing this module never downloads weights or imports torch. Each model is
initialised at most once per process (success *and* failure are remembered), and is
only reported READY after a real inference probe passes. See docs/MODELS.md.
"""
import os

from app.core.model_health import (
    CANONICAL,
    ModelRegistry,
    ModelSpec,
    ModelUnavailable,
    derive_label_map,
    normalize_nli_scores,
)
from backend.config import get_settings

CROSS_ENCODER_ID = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
NLI_ID = 'typeform/distilbert-base-uncased-mnli'

# Two independent behavioural probes per class: the first set may be used to derive a label mapping when the
# checkpoint only exposes LABEL_0..2; the second set then verifies it. Wording is deliberately unambiguous.
NLI_PROBES = {
    'ENTAILMENT': [('A man is playing a guitar on a stage.', 'A person is playing an instrument.'),
                   ('The trial enrolled 200 patients across five hospitals.', 'The trial included patients from several hospitals.')],
    'CONTRADICTION': [('A man is playing a guitar.', 'Nobody is playing any instrument.'),
                      ('The store is closed on Sundays.', 'The store is open on Sundays.')],
    'NEUTRAL': [('A woman is reading a book in the park.', 'The woman is a university professor.'),
                ('The company released a new phone in March.', 'The phone sold more than a million units.')],
}
_nli_meta: dict = {'derived_map': None, 'mapping_source': None}


def _cache_kwargs() -> dict:
    cache = get_settings().model_cache_dir
    return {'cache_dir': cache} if cache else {}


def _local_only() -> bool:
    return not get_settings().citeguard_allow_download


def _configure_cpu():
    import torch
    torch.set_num_threads(max(1, int(os.getenv('CITEGUARD_TORCH_THREADS', str(min(8, os.cpu_count() or 4))))))
    return torch


def _device_of(obj) -> str | None:
    for path in (('device',), ('model', 'device')):
        cur = obj
        try:
            for attr in path:
                cur = getattr(cur, attr)
            return str(cur)
        except AttributeError:
            continue
    return None


def _versions() -> dict:
    out = {}
    for mod in ('torch', 'transformers', 'sentence_transformers'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - version reporting must never fail a health check
            out[mod] = None
    return out


# ---- cross-encoder ----------------------------------------------------------------------------

def _load_cross_encoder():
    _configure_cpu()
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_ID, max_length=512, local_files_only=_local_only(), **_cache_kwargs())


def _probe_cross_encoder(model) -> dict:
    q = 'How many people live in Berlin?'
    scores = model.predict([(q, 'Berlin has a population of about 3.7 million inhabitants.'),
                            (q, 'Bananas are a yellow tropical fruit rich in potassium.')])
    scores = [float(s) for s in scores]
    assert len(scores) == 2 and all(s == s for s in scores), f'non-finite scores {scores}'
    assert scores[0] > scores[1], f'relevant passage not ranked above irrelevant one: {scores}'
    return {'device': _device_of(model), 'versions': _versions(), 'probe_scores': scores}


# ---- NLI --------------------------------------------------------------------------------------

def _load_nli():
    _configure_cpu()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    local, extra = _local_only(), _cache_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(NLI_ID, local_files_only=local, **extra)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_ID, local_files_only=local, **extra)
    return pipeline('text-classification', model=model, tokenizer=tokenizer,
                    top_k=None, truncation=True, max_length=512)


def _run_nli(pipe, premise: str, hypothesis: str) -> list[dict]:
    raw = pipe({'text': premise, 'text_pair': hypothesis})
    return raw[0] if raw and isinstance(raw[0], list) else raw


def _probe_nli(pipe) -> dict:
    id2label = dict(getattr(getattr(pipe.model, 'config', None), 'id2label', {}) or {})
    first = {cls: _run_nli(pipe, *pairs[0]) for cls, pairs in NLI_PROBES.items()}
    derived = None
    if any(str(p['label']).upper().startswith('LABEL_') for p in first['ENTAILMENT']) and \
            all(str(v).upper().startswith('LABEL_') for v in id2label.values()):
        derived = derive_label_map(first)
        assert derived, 'checkpoint exposes only generic labels and behavioural probes could not establish a mapping'
    source = None
    for cls, pairs in NLI_PROBES.items():
        for premise, hypothesis in pairs:
            scores, source = normalize_nli_scores(_run_nli(pipe, premise, hypothesis), id2label, derived)
            assert scores is not None, f'labels {id2label} cannot be mapped to {CANONICAL}'
            top = max(scores, key=scores.get)
            assert top == cls, f'probe expecting {cls} was classified {top}: {scores}'
    _nli_meta.update(derived_map=derived, mapping_source=source)
    return {'device': _device_of(pipe), 'versions': _versions(), 'label_mapping': source, 'probes_passed': 6}


# ---- registry ---------------------------------------------------------------------------------

_registry: ModelRegistry | None = None


def registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry([ModelSpec('cross_encoder', CROSS_ENCODER_ID, _load_cross_encoder, _probe_cross_encoder),
                                   ModelSpec('nli', NLI_ID, _load_nli, _probe_nli)],
                                  offline=_local_only())
    return _registry


def cross_encoder():
    """Ready cross-encoder or ModelUnavailable (never a half-initialised model)."""
    return registry().get('cross_encoder')


def nli_pipeline():
    """Ready NLI pipeline or ModelUnavailable."""
    return registry().get('nli')


def nli_label_map() -> dict | None:
    return _nli_meta['derived_map']


def model_health(refresh: bool = False) -> dict:
    reg = registry()
    return reg.refresh() if refresh else reg.check_all()


def reset_registry_for_tests() -> None:
    global _registry
    _registry = None
    _nli_meta.update(derived_map=None, mapping_source=None)


__all__ = ['ModelUnavailable', 'cross_encoder', 'nli_pipeline', 'nli_label_map', 'model_health', 'registry']
