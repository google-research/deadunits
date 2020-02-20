# coding=utf-8
# Copyright 2021 The Deadunits Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Tests for `deadunits.pruner`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deadunits import pruner
from deadunits import train_utils
from deadunits.generic_convnet import GenericConvnet
import mock
from six.moves import zip
import tensorflow.compat.v1 as tf


class ProbePruningTest(tf.test.TestCase):

  @mock.patch('deadunits.pruner.process_layers2prune')
  @mock.patch('deadunits.pruner.get_pruning_measurements')
  @mock.patch('deadunits.pruner.prune_model_with_scores')
  @mock.patch('deadunits.train_utils.cross_entropy_loss')
  @mock.patch('tensorflow.compat.v2.summary.scalar')
  def testDefault(self, mock_summary, mock_loss, mock_prune_model,
                  mock_pruning_measurements, mock_process_l2p):
    mock_getitem = mock.Mock()
    mock_scores = mock.Mock(__getitem__=mock_getitem)
    mock_pruning_measurements.return_value = (mock_scores, None, None)
    mock_loss.return_value = (0.5, None, None)
    model = mock.Mock()
    f_retrain = mock.Mock()
    d_val = d_val2 = d_test = None
    selected_units, _, _ = pruner.probe_pruning(model, d_val, d_val2, d_test,
                                                f_retrain)
    self.assertEqual(len(selected_units), 1)
    self.assertEqual(model.clone.call_count, 2)
    self.assertEqual(mock_loss.call_count, 2)
    mock_process_l2p.assert_called_once()
    mock_pruning_measurements.assert_called_once()
    mock_prune_model.assert_called_once()
    mock_getitem.assert_called_once()
    mock_getitem.assert_called_with('norm')
    f_retrain.assert_not_called()
    mock_summary.assert_any_call('pruning_penalty_bp_norm', 0.5)
    mock_summary.assert_any_call('pruning_penalty_bp_norm_test', 0.5)
    self.assertEqual(mock_summary.call_count, 2)

  @mock.patch('deadunits.pruner.process_layers2prune')
  @mock.patch('deadunits.pruner.get_pruning_measurements')
  @mock.patch('deadunits.pruner.prune_model_with_scores')
  @mock.patch('deadunits.train_utils.cross_entropy_loss')
  @mock.patch('tensorflow.compat.v2.summary.scalar')
  def testNRetrain(self, mock_summary, mock_loss, mock_prune_model,
                   mock_pruning_measurements, mock_process_l2p):
    mock_getitem = mock.Mock()
    mock_scores = mock.Mock(__getitem__=mock_getitem)
    mock_pruning_measurements.return_value = (mock_scores, None, None)
    mock_loss.return_value = (0.5, None, None)
    model = mock.Mock()
    f_retrain = mock.Mock()
    d_val = d_val2 = d_test = None
    pruner.probe_pruning(model, d_val, d_val2, d_test, f_retrain, n_retrain=100)
    self.assertEqual(model.clone.call_count, 2)
    self.assertEqual(mock_loss.call_count, 2)
    mock_process_l2p.assert_called_once()
    mock_pruning_measurements.assert_called_once()
    mock_prune_model.assert_called_once()
    mock_getitem.assert_called_once()
    mock_getitem.assert_called_once_with('norm')
    f_retrain.assert_called_once()

    # Negative values are same as 0.
    f_retrain.reset_mock()
    pruner.probe_pruning(model, d_val, d_val2, d_test, f_retrain, n_retrain=-4)
    f_retrain.assert_not_called()

  @mock.patch('deadunits.pruner.process_layers2prune')
  @mock.patch('deadunits.pruner.get_pruning_measurements')
  @mock.patch('deadunits.pruner.prune_model_with_scores')
  @mock.patch('deadunits.train_utils.cross_entropy_loss')
  @mock.patch('tensorflow.contrib.summary.scalar')
  def testPruningMethods(self, mock_summary, mock_loss, mock_prune_model,
                         mock_pruning_measurements, mock_process_l2p):
    mock_getitem = mock.Mock()
    mock_scores = mock.Mock(__getitem__=mock_getitem)
    mock_pruning_measurements.return_value = (mock_scores, None, None)
    mock_loss.return_value = (0.5, None, None)
    model = mock.Mock()
    f_retrain = mock.Mock()
    d_val = d_val2 = d_test = None
    p_methods = (('mrs', True), ('mrs', False), ('rs', True))
    selected_units, _, _ = pruner.probe_pruning(model, d_val, d_val2, d_test,
                                                f_retrain,
                                                pruning_methods=p_methods)
    self.assertEqual(len(selected_units), 2)
    len_pm = len(p_methods)
    self.assertEqual(model.clone.call_count, 1+len_pm)
    self.assertEqual(mock_loss.call_count, 2*len_pm)
    mock_process_l2p.assert_called_once()
    mock_pruning_measurements.assert_called_once()
    self.assertEqual(mock_prune_model.call_count, len_pm)
    self.assertEqual(mock_getitem.call_count, len_pm)
    for (args, _), p_method in zip(mock_getitem.call_args_list, p_methods):
      self.assertEqual(args[0], p_method[0])

    f_retrain.assert_not_called()
    for score, is_bp in p_methods:
      tag = 'pruning_penalty%s_%s' % ('_bp' if is_bp else '', score)
      mock_summary.assert_any_call(tag, 0.5)
      mock_summary.assert_any_call(tag+'_test', 0.5)
    # Called twice for each combination.
    self.assertEqual(mock_summary.call_count, len_pm * 2)


class PruneModelTest(tf.test.TestCase):

  @mock.patch('deadunits.utils.mask_and_broadcast')
  def testLayerPruneWithBiasProp(self, mock_maskandbroadcast):
    combinations = [['conv_1'],
                    ['conv_1', 'conv_2', 'dense_1'],
                    ['conv_2', 'dense_1']]
    for layers2prune in combinations:
      model = GenericConvnet(use_masked_layers=True)
      # Initialize model.
      model(tf.ones((2, 32, 32, 3)))
      mock_bp = mock.Mock()
      model.propagate_bias = mock_bp
      # Reset call counts.
      mock_maskandbroadcast.reset_mock()
      scores = {}
      mean_vals = {}
      inp_shapes = {}
      for l_name in layers2prune:
        c_shape = getattr(model, l_name).get_layer().weights[0].shape
        scores[l_name] = tf.range(c_shape[-1])
        mean_vals[l_name] = tf.range(c_shape[-1])
        inp_shapes[l_name] = None
      is_bp = True
      pruning_count, pruning_factor = None, 0.1
      selected_units = pruner.prune_model_with_scores(
          model, scores, is_bp, layers2prune, pruning_factor, pruning_count,
          mean_vals, inp_shapes)
      for l_name in layers2prune:
        l = getattr(model, l_name)
        self.assertAllEqual(l.mask_bias.numpy(),
                            selected_units[l_name][1])
        self.assertAllEqual(l.mask_weight.numpy(),
                            tf.broadcast_to(selected_units[l_name][1],
                                            l.get_layer().weights[0].shape))
        self.assertAllEqual(scores[l_name], selected_units[l_name][0])
      self.assertEqual(mock_maskandbroadcast.call_count, len(layers2prune))
      self.assertEqual(mock_bp.call_count, len(layers2prune))

  @mock.patch('deadunits.utils.mask_and_broadcast')
  def testLayerPruneWithoutBiasProp(self, mock_maskandbroadcast):
    combinations = [['conv_2'],
                    ['conv_1', 'conv_2', 'dense_1'],
                    ['conv_1', 'dense_1']]
    for layers2prune in combinations:
      model = GenericConvnet(use_masked_layers=True)
      # Initialize model.
      model(tf.ones((2, 32, 32, 3)))
      mock_bp = mock.Mock()
      model.propagate_bias = mock_bp
      # Reset call counts.
      mock_maskandbroadcast.reset_mock()
      scores = {}
      mean_vals = {}
      inp_shapes = {}
      for l_name in layers2prune:
        c_shape = getattr(model, l_name).get_layer().weights[0].shape
        scores[l_name] = tf.range(c_shape[-1])
        mean_vals[l_name] = tf.range(c_shape[-1])
        inp_shapes[l_name] = None
      is_bp = False
      pruning_count, pruning_factor = None, 0.1
      selected_units = pruner.prune_model_with_scores(
          model, scores, is_bp, layers2prune, pruning_factor, pruning_count,
          mean_vals, inp_shapes)
      for l_name in layers2prune:
        l = getattr(model, l_name)
        self.assertAllEqual(l.mask_bias.numpy(),
                            selected_units[l_name][1])
        self.assertAllEqual(l.mask_weight.numpy(),
                            tf.broadcast_to(selected_units[l_name][1],
                                            l.get_layer().weights[0].shape))
        self.assertAllEqual(scores[l_name], selected_units[l_name][0])
      mock_maskandbroadcast.assert_not_called()
      mock_bp.assert_not_called()


class ProcessLayers2PruneTest(tf.test.TestCase):

  def _get_model_with_masked_layers(self, list_of_masked_layers):
    model = mock.Mock()
    with mock.patch(
        'deadunits.layers.MaskedLayer', autospec=True) as mock_masked:
      for l_name in list_of_masked_layers:
        setattr(model, l_name, mock_masked(mock.Mock()))
    return model

  def testStringAll(self):
    combinations = [['conv_1', 'conv_2', 'dense_1'],
                    ['dense_1'],
                    ['conv_1']]
    for layers2prune in combinations:
      model = self._get_model_with_masked_layers(layers2prune)
      mock_get_layer_keys = mock.Mock(return_value=layers2prune)
      model.get_layer_keys = mock_get_layer_keys
      layers2prune_processed = pruner.process_layers2prune('all', model)
      self.assertAllEqual(layers2prune, layers2prune_processed)
      mock_get_layer_keys.assert_called_once()
      mock_get_layer_keys.reset_mock()

  def testStringFirstConvDense(self):
    model = self._get_model_with_masked_layers(['conv_1'])
    layers2prune_processed = pruner.process_layers2prune('firstconv', model)
    self.assertAllEqual(['conv_1'], layers2prune_processed)
    model = self._get_model_with_masked_layers(['dense_1'])
    layers2prune_processed = pruner.process_layers2prune('firstdense', model)
    self.assertAllEqual(['dense_1'], layers2prune_processed)

  def testStringCustom(self):
    combinations = [['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                    ['conv_1', 'conv_2', 'conv_3', 'conv_4'],
                    ['conv_1', 'conv_2', 'conv_3'],
                    ['conv_1']]
    expected_results = [['conv_3', 'conv_5'],
                        ['conv_2', 'conv_4'],
                        ['conv_2', 'conv_3'],
                        ['conv_1', 'conv_1']]
    for res, layers2prune in zip(expected_results, combinations):
      model = self._get_model_with_masked_layers(layers2prune)
      model.layer_name_counter = {'C': (len(layers2prune) + 1)}
      layers2prune_processed = pruner.process_layers2prune('midconv', model)
      self.assertAllEqual([res[0]], layers2prune_processed)
      layers2prune_processed = pruner.process_layers2prune('lastconv', model)
      self.assertAllEqual([res[1]], layers2prune_processed)

  def testStringMidLast(self):
    all_layers = ['conv_1', 'conv_2', 'conv_3']
    model = self._get_model_with_masked_layers(all_layers)
    for l_name in all_layers:
      layers2prune_processed = pruner.process_layers2prune(l_name, model)
      self.assertAllEqual([l_name], layers2prune_processed)

  def testList(self):
    all_layers = ['conv_1', 'conv_2', 'conv_3',
                  'dense_1', 'dense_2', 'dense_3']
    test_lists = [['conv_1', 'dense_1'],
                  ['conv_2', 'dense_3'],
                  ['conv_3', 'dense_3'],
                  ['conv_2'],
                  ['conv_2', 'conv_2']]
    model = self._get_model_with_masked_layers(all_layers)
    for test_list in test_lists:
      layers2prune_processed = pruner.process_layers2prune(test_list, model)
      self.assertAllEqual(test_list, layers2prune_processed)

  def testAssertion(self):
    all_layers = ['conv_1', 'conv_2', 'conv_3',
                  'dense_1', 'dense_2', 'dense_3']
    test_args = [['conv_0', 'dense_1'],
                 ['output_1', 'dense_3'],
                 ['conv_4'],
                 ['conv_0'],
                 'firstConv',
                 'LastConv']
    model = self._get_model_with_masked_layers(all_layers)
    for test_arg in test_args:
      with self.assertRaises(AssertionError):
        _ = pruner.process_layers2prune(test_arg, model)


class GetPruningMeasurementsTest(tf.test.TestCase):

  def _get_model_with_ts_layers(self, list_of_layers):
    model = mock.Mock()
    with mock.patch(
        'deadunits.layers.TaylorScorer', autospec=True) as mock_taylor:
      with mock.patch(
          'deadunits.layers.MaskedLayer', autospec=True) as mock_masked:
        for l_name in list_of_layers:
          setattr(model, l_name, mock_masked(mock.Mock()))
          setattr(model, l_name + '_ts', mock_taylor(mock.Mock()))
    return model

  @mock.patch('deadunits.train_utils.cross_entropy_loss')
  @mock.patch('deadunits.unitscorers.norm_score')
  @mock.patch('deadunits.unitscorers.random_score')
  def testGeneric(self, mock_random_scorer, mock_norm_scorer,
                  mock_cross_entropy):
    combinations = [['conv_1', 'conv_2', 'dense_1'],
                    ['dense_1'],
                    ['conv_1']]
    for layers2score in combinations:
      model = self._get_model_with_ts_layers(layers2score)
      measurements = pruner.get_pruning_measurements(model, mock.Mock(),
                                                     layers2score)
      scores, mean_vals, l2_norms = measurements
      self.assertAllEqual(set(mean_vals.keys()), set(layers2score))
      self.assertAllEqual(set(l2_norms.keys()), set(layers2score))
      for k in ['abs_mrs', 'abs_rs', 'mrs', 'rs', 'norm', 'rand']:
        self.assertAllEqual(set(scores[k].keys()), set(layers2score))
      self.assertEqual(len(scores), 6)
      self.assertEqual(mock_cross_entropy.call_count, 2)
      self.assertEqual(mock_random_scorer.call_count, len(layers2score))
      self.assertEqual(mock_norm_scorer.call_count, len(layers2score))
      mock_cross_entropy.reset_mock()
      mock_random_scorer.reset_mock()
      mock_norm_scorer.reset_mock()


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
