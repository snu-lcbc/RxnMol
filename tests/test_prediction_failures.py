"""Regression checks for prediction failure handling without a GPU or model files."""
import unittest
from unittest.mock import Mock

# Match the public run.py entry point's initialization order.
import rxnmol.core
from rxnmol.solutions.fragments_route import FragmentRouteSpec


class PredictionFailureTests(unittest.TestCase):
    def setUp(self):
        self.spec = FragmentRouteSpec.__new__(FragmentRouteSpec)
        self.spec.rxn_predictor = Mock()
        self.spec.reaction_cache = Mock()

    def test_fatal_batch_error_stops_without_sequential_retry(self):
        self.spec.rxn_predictor.predict_batch.side_effect = RuntimeError('CUDA error: device-side assert triggered')
        with self.assertRaisesRegex(RuntimeError, 'Unrecoverable device fault'):
            self.spec._predict_missing_with_fallback(['a', 'b'])
        self.spec.rxn_predictor.predict.assert_not_called()

    def test_out_of_memory_can_use_sequential_fallback(self):
        self.spec.rxn_predictor.predict_batch.side_effect = RuntimeError('CUDA out of memory')
        self.spec.rxn_predictor.predict.side_effect = ['CC', 'CO']
        self.assertEqual(self.spec._predict_missing_with_fallback(['a', 'b']), ['CC', 'CO'])

    def test_input_failure_does_not_discard_other_predictions(self):
        self.spec.rxn_predictor.predict_batch.side_effect = ValueError('invalid input')
        self.spec.rxn_predictor.predict.side_effect = [ValueError('invalid input'), 'CO']
        self.assertEqual(self.spec._predict_missing_with_fallback(['a', 'b']), ['FAILED', 'CO'])

    def test_fatal_error_during_fallback_stops_remaining_inputs(self):
        self.spec.rxn_predictor.predict_batch.side_effect = ValueError('batch failed')
        self.spec.rxn_predictor.predict.side_effect = ['CC', RuntimeError('illegal memory access'), 'CO']
        with self.assertRaisesRegex(RuntimeError, 'Unrecoverable device fault'):
            self.spec._predict_missing_with_fallback(['a', 'b', 'c'])
        self.assertEqual(self.spec.rxn_predictor.predict.call_count, 2)

    def test_complete_fallback_failure_stops_without_cache_write(self):
        self.spec.reaction_cache.get_batch.return_value = [None, None]
        self.spec.rxn_predictor.predict_batch.side_effect = ValueError('batch failed')
        self.spec.rxn_predictor.predict.side_effect = ValueError('model unavailable')
        with self.assertRaisesRegex(RuntimeError, 'All 2 reaction predictions failed'):
            self.spec._batch_predict_with_cache(['a', 'b'])
        self.spec.reaction_cache.set_batch.assert_not_called()

    def test_failed_predictions_are_not_cached_and_order_is_preserved(self):
        self.spec.reaction_cache.get_batch.return_value = ['CC', None, None]
        self.spec.rxn_predictor.predict_batch.return_value = ['FAILED', 'CO']
        self.assertEqual(self.spec._batch_predict_with_cache(['cached', 'bad', 'good']), ['CC', 'FAILED', 'CO'])
        self.spec.reaction_cache.set_batch.assert_called_once_with(['good'], ['CO'])

    def test_all_cached_predictions_skip_model(self):
        self.spec.reaction_cache.get_batch.return_value = ['CC', 'CO']
        self.assertEqual(self.spec._batch_predict_with_cache(['a', 'b']), ['CC', 'CO'])
        self.spec.rxn_predictor.predict_batch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
