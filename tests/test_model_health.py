"""Model readiness without torch: failures, caching, probes, label mapping (cases 7 & 8)."""
import unittest

from app.core.model_health import (
    ModelRegistry,
    ModelSpec,
    ModelUnavailable,
    classify_exception,
    derive_label_map,
    normalize_nli_scores,
)


def spec(name, loader=lambda: object(), probe=lambda m: {'device': 'cpu'}):
    return ModelSpec(name, 'org/' + name, loader, probe)


class Registry(unittest.TestCase):
    def test_ready_only_after_probe_passes(self):
        reg = ModelRegistry([spec('nli')])
        reg.get('nli')
        self.assertEqual(reg.status('nli').status, 'READY')
        self.assertEqual(reg.status('nli').device, 'cpu')

    def test_loaded_but_failing_probe_is_not_ready(self):
        def bad_probe(m): raise AssertionError('relevant passage not ranked above irrelevant one')
        reg = ModelRegistry([spec('cross_encoder', probe=bad_probe)])
        with self.assertRaises(ModelUnavailable):
            reg.get('cross_encoder')
        st = reg.status('cross_encoder')
        self.assertEqual((st.status, st.stage, st.cause), ('UNAVAILABLE', 'inference', 'BAD_OUTPUT'))

    def test_missing_package_reports_cause(self):                      # case 7/8 root cause path
        def loader(): raise ImportError("No module named 'torch'")
        reg = ModelRegistry([spec('nli', loader=loader)])
        with self.assertRaises(ModelUnavailable):
            reg.get('nli')
        st = reg.status('nli')
        self.assertEqual((st.stage, st.cause), ('import', 'MISSING_PACKAGE'))
        self.assertIn('requirements', st.remedy)

    def test_offline_missing_weights(self):
        def loader(): raise OSError("We couldn't connect ... local_files_only=True and files not found in cache")
        reg = ModelRegistry([spec('nli', loader=loader)], offline=True)
        with self.assertRaises(ModelUnavailable):
            reg.get('nli')
        self.assertEqual(reg.status('nli').cause, 'WEIGHTS_NOT_CACHED')
        self.assertIn('prepare_models', reg.status('nli').remedy)

    def test_network_failure_when_downloads_allowed(self):
        cause, _ = classify_exception(ConnectionError('Max retries exceeded: timed out'), offline=False)
        self.assertEqual(cause, 'NETWORK_BLOCKED')

    def test_unknown_error_is_init_error(self):
        self.assertEqual(classify_exception(RuntimeError('CUDA kernel image'), False)[0], 'INIT_ERROR')

    def test_load_once_success_and_failure_cached(self):
        calls = {'ok': 0, 'bad': 0}
        def ok():
            calls['ok'] += 1
            return object()
        def bad():
            calls['bad'] += 1
            raise OSError('x')
        reg = ModelRegistry([spec('a', loader=ok), spec('b', loader=bad)])
        for _ in range(5):
            reg.get('a')
            with self.assertRaises(ModelUnavailable):
                reg.get('b')
        self.assertEqual(calls, {'ok': 1, 'bad': 1})   # no re-initialisation, no retry storm

    def test_refresh_retries(self):
        state = {'fail': True, 'n': 0}
        def loader():
            state['n'] += 1
            if state['fail']:
                raise OSError('no cache')
            return object()
        reg = ModelRegistry([spec('nli', loader=loader)])
        with self.assertRaises(ModelUnavailable):
            reg.get('nli')
        state['fail'] = False
        reg.refresh()
        self.assertEqual(reg.status('nli').status, 'READY')
        self.assertEqual(state['n'], 2)

    def test_unavailable_error_message_is_preserved(self):
        reg = ModelRegistry([spec('nli', loader=lambda: (_ for _ in ()).throw(ValueError('boom detail')))])
        with self.assertRaises(ModelUnavailable) as ctx:
            reg.get('nli')
        self.assertIn('boom detail', str(ctx.exception))

    def test_check_all_never_raises(self):
        reg = ModelRegistry([spec('a', loader=lambda: 1 / 0), spec('b')])
        out = reg.check_all()
        self.assertEqual((out['a']['status'], out['b']['status']), ('UNAVAILABLE', 'READY'))


class Labels(unittest.TestCase):
    NATIVE = [{'label': 'ENTAILMENT', 'score': .7}, {'label': 'NEUTRAL', 'score': .2}, {'label': 'CONTRADICTION', 'score': .1}]

    def test_native_labels(self):
        scores, src = normalize_nli_scores(self.NATIVE)
        self.assertEqual((src, scores['ENTAILMENT']), ('native', .7))

    def test_generic_labels_use_config_when_informative(self):
        raw = [{'label': 'LABEL_0', 'score': .6}, {'label': 'LABEL_1', 'score': .3}, {'label': 'LABEL_2', 'score': .1}]
        scores, src = normalize_nli_scores(raw, {0: 'contradiction', 1: 'neutral', 2: 'entailment'})
        self.assertEqual((src, scores['CONTRADICTION']), ('config', .6))

    def test_generic_labels_without_mapping_are_rejected_not_guessed(self):
        raw = [{'label': 'LABEL_0', 'score': .6}, {'label': 'LABEL_1', 'score': .3}, {'label': 'LABEL_2', 'score': .1}]
        scores, _ = normalize_nli_scores(raw, {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})
        self.assertIsNone(scores)

    def test_probe_derived_mapping(self):
        def preds(top):
            return [{'label': f'LABEL_{i}', 'score': .8 if i == top else .1} for i in range(3)]
        mapping = derive_label_map({'ENTAILMENT': preds(2), 'CONTRADICTION': preds(0), 'NEUTRAL': preds(1)})
        self.assertEqual(mapping, {'LABEL_2': 'ENTAILMENT', 'LABEL_0': 'CONTRADICTION', 'LABEL_1': 'NEUTRAL'})
        scores, src = normalize_nli_scores(preds(2), {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}, mapping)
        self.assertEqual((src, scores['ENTAILMENT']), ('probe_derived', .8))

    def test_ambiguous_probes_give_no_mapping(self):
        same = [{'label': 'LABEL_0', 'score': .9}, {'label': 'LABEL_1', 'score': .05}, {'label': 'LABEL_2', 'score': .05}]
        self.assertIsNone(derive_label_map({'ENTAILMENT': same, 'CONTRADICTION': same, 'NEUTRAL': same}))

    def test_incomplete_label_set_rejected(self):
        scores, _ = normalize_nli_scores([{'label': 'ENTAILMENT', 'score': .99}])
        self.assertIsNone(scores)


if __name__ == '__main__':
    unittest.main()
