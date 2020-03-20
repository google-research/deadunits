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
"""Tests for `deadunits.generic_convnet`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deadunits import generic_convnet
from deadunits import layers
import tensorflow.compat.v2 as tf


class GenericConvnetTest(tf.test.TestCase):

  def testDefaultConstructor(self):
    m = generic_convnet.GenericConvnet(name='test')
    # m.forward_chain should be like
    # ['conv_1', 'conv_1_a', 'maxpool_1', 'conv_2', 'conv_2_a', 'maxpool_2',
    # 'flatten_1', 'dense_1', 'dense_1_a', 'output_1'])
    self.assertEqual(len(m.forward_chain), 10)
    self.assertEqual(m.name, 'test')
    self.assertIsInstance(m.conv_1, tf.keras.layers.Conv2D)
    self.assertEqual(m.conv_1_a, tf.keras.activations.relu)
    self.assertIsInstance(m.maxpool_1, tf.keras.layers.MaxPool2D)
    self.assertIsInstance(m.conv_2, tf.keras.layers.Conv2D)
    self.assertEqual(m.conv_2_a, tf.keras.activations.relu)
    self.assertIsInstance(m.maxpool_2, tf.keras.layers.MaxPool2D)
    self.assertIsInstance(m.flatten_1, tf.keras.layers.Flatten)
    self.assertIsInstance(m.dense_1, tf.keras.layers.Dense)
    self.assertEqual(m.dense_1_a, tf.keras.activations.relu)
    self.assertIsInstance(m.output_1, tf.keras.layers.Dense)

  def testInvalidModelArchitectures(self):
    with self.assertRaises(AssertionError):
      # Needs to be non-zero length iterable of collections.MutableSequence.
      generic_convnet.GenericConvnet(model_arch='test')
    with self.assertRaises(AssertionError):
      generic_convnet.GenericConvnet(model_arch=[['C', 16, [5, 2, 5], {}]])
    with self.assertRaises(AssertionError):
      generic_convnet.GenericConvnet(model_arch=[['C', 16, [5, '5'], {}]])
    with self.assertRaises(AssertionError):
      # Empty kwargs missing
      generic_convnet.GenericConvnet(model_arch=[['C', 16, 4]])
    with self.assertRaises(AssertionError):
      generic_convnet.GenericConvnet(model_arch=[['MP', 16, [3, 5], {}]])
    with self.assertRaises(AssertionError):
      # Dense only requires one int (out_units).
      generic_convnet.GenericConvnet(model_arch=[['D', 16, 4]])
    with self.assertRaises(AssertionError):
      # Second argument should be n_units. Activations are given with a flag.
      generic_convnet.GenericConvnet(model_arch=[['0', 'norelu']])
    with self.assertRaises(AssertionError):
      # Flatten should not have other elements in the list
      generic_convnet.GenericConvnet(model_arch=[['F', 16]])
    with self.assertRaises(AssertionError):
      # GlobalAveragePooling should not have other elements in the list
      generic_convnet.GenericConvnet(model_arch=[['GA', 16]])
    with self.assertRaises(AssertionError):
      # Second argument should be n_units. Activations are given with a flag.
      generic_convnet.GenericConvnet(model_arch=[['D', 'relu']])
    with self.assertRaises(AssertionError):
      # Second argument should be dropout rate between 0,1.
      generic_convnet.GenericConvnet(model_arch=[['DO', 0.0]])
    with self.assertRaises(AssertionError):
      # Second argument should be dropout rate between 0,1.
      generic_convnet.GenericConvnet(model_arch=[['DO', '0.5']])

  def testValidModelArchitectures(self):
    # 3rd argument for 'C' is the filter shape, which can be an int or list
    # of two integers.
    generic_convnet.GenericConvnet(model_arch=[['C', 16, [3, 5], {}]])
    generic_convnet.GenericConvnet(model_arch=[['C', 16, 5, {}]])
    # Check that **kwargs is working.
    m = generic_convnet.GenericConvnet(
        model_arch=[['C', 3, 5, {
            'padding': 'same'
        }]])
    x = tf.random.uniform((5, 32, 32, 3))
    y = m(x)
    self.assertAllEqual(x.shape, y.shape)
    generic_convnet.GenericConvnet(model_arch=[['MP', 2, [3, 5]]])
    generic_convnet.GenericConvnet(model_arch=[['MP', [2, 3], 5]])

  def testMaskedLayer(self):
    m = generic_convnet.GenericConvnet(name='test', use_masked_layers=True)
    self.assertIsInstance(m.conv_1, layers.MaskedLayer)
    self.assertIsInstance(m.conv_2, layers.MaskedLayer)
    self.assertIsInstance(m.dense_1, layers.MaskedLayer)
    self.assertNotIsInstance(m.flatten_1, layers.MaskedLayer)

  def testUseTaylorScorer(self):
    model_arch = [['C', 16, [3, 5], {}], ['F'], ['D', 16]]
    m = generic_convnet.GenericConvnet(
        name='test', model_arch=model_arch, use_taylor_scorer=True)
    m(tf.random.uniform((2, 10, 10, 3)))
    correct_chain = [
        'conv_1', 'conv_1_a', 'conv_1_ts', 'flatten_1', 'dense_1', 'dense_1_a',
        'dense_1_ts'
    ]
    self.assertAllEqual(m.forward_chain, correct_chain)
    self.assertIsInstance(m.conv_1_ts, layers.TaylorScorer)
    self.assertNotIsInstance(m.conv_1, layers.TaylorScorer)
    self.assertIsInstance(m.dense_1_ts, layers.TaylorScorer)
    self.assertNotIsInstance(m.dense_1, layers.TaylorScorer)

  def testUseMeanReplacer(self):
    model_arch = [['C', 16, [3, 5], {}], ['GA'], ['D', 16]]
    m = generic_convnet.GenericConvnet(
        name='test', model_arch=model_arch, use_mean_replacer=True)
    m(tf.random.uniform((2, 10, 10, 3)))
    correct_chain = [
        'conv_1', 'conv_1_a', 'conv_1_mr', 'gap_1', 'dense_1', 'dense_1_a',
        'dense_1_mr'
    ]
    self.assertAllEqual(m.forward_chain, correct_chain)
    self.assertIsInstance(m.conv_1_mr, layers.MeanReplacer)
    self.assertNotIsInstance(m.conv_1, layers.MeanReplacer)
    self.assertIsInstance(m.dense_1_mr, layers.MeanReplacer)
    self.assertNotIsInstance(m.dense_1, layers.MeanReplacer)

  def testUseDropOut(self):
    model_arch = [['C', 16, [3, 5], {}], ['F'], ['D', 16]]
    m = generic_convnet.GenericConvnet(
        name='test', model_arch=model_arch, use_dropout=True)
    m(tf.random.uniform((2, 10, 10, 3)))
    correct_chain = [
        'conv_1', 'conv_1_a', 'conv_1_dr', 'flatten_1', 'dense_1', 'dense_1_a',
        'dense_1_dr'
    ]
    self.assertAllEqual(m.forward_chain, correct_chain)
    self.assertIsInstance(m.conv_1_dr, tf.keras.layers.Dropout)
    self.assertIsInstance(m.dense_1_dr, tf.keras.layers.Dropout)

  def testDropoutInjection(self):
    model_arch = [['C', 16, [3, 5], {}], ['DO', 0.5], ['F'], ['D', 16]]
    m = generic_convnet.GenericConvnet(
        name='test', model_arch=model_arch, use_dropout=False)
    m(tf.random.uniform((2, 10, 10, 3)))
    correct_chain = [
        'conv_1', 'conv_1_a', 'dropout_1', 'flatten_1', 'dense_1', 'dense_1_a'
    ]
    self.assertAllEqual(m.forward_chain, correct_chain)
    self.assertIsInstance(m.dropout_1, tf.keras.layers.Dropout)

  def testGetAllLayerKeys(self):
    m = generic_convnet.GenericConvnet(
        name='test', use_masked_layers=True, use_batchnorm=True)
    returned_set = set(m.get_layer_keys(layers.MaskedLayer))
    correct_set = set(['conv_1', 'conv_2', 'dense_1'])
    self.assertSetEqual(returned_set, correct_set)
    returned_set = set(m.get_layer_keys(tf.keras.layers.BatchNormalization))
    correct_set = set(['conv_1_bn', 'conv_2_bn', 'dense_1_bn'])
    self.assertSetEqual(returned_set, correct_set)
    returned_set = set(
        m.get_layer_keys(
            tf.keras.layers.BatchNormalization,
            name_filter=lambda n: not n.startswith('dense')))
    correct_set = set(['conv_1_bn', 'conv_2_bn'])
    self.assertSetEqual(returned_set, correct_set)

  def testClone(self):
    m = generic_convnet.GenericConvnet(name='test', use_masked_layers=True)
    # Initilizes the params.
    m(tf.random.uniform((4, 32, 32, 3)))
    m2 = m.clone()
    self.assertAllEqual(m2.conv_2.weights[0].numpy(),
                        m.conv_2.weights[0].numpy())
    self.assertAllEqual(m2.dense_1.weights[0].numpy(),
                        m.dense_1.weights[0].numpy())
    self.assertNotEqual(m2.conv_2, m.conv_2)

  def testPropagateBiasErrors(self):
    m = generic_convnet.GenericConvnet(name='test', use_masked_layers=True)
    with self.assertRaises(AssertionError):
      m.propagate_bias('conv_1', tf.ones((32, 24, 24, 3), dtype=tf.int16))
    with self.assertRaises(ValueError):
      # Layer name misspelled.
      m.propagate_bias('conv1', tf.ones((32, 24, 24, 3), dtype=tf.float32))
    with self.assertRaises(ValueError):
      # There is no other layer to propagate.
      m.propagate_bias('output_1', tf.ones((32, 24, 24, 3), dtype=tf.float32))

  def testReturnNodes(self):
    model_arch = [['C', 16, [3, 5], {}], ['F'], ['D', 16]]
    m = generic_convnet.GenericConvnet(
        name='test', model_arch=model_arch, use_taylor_scorer=True)
    # forward_chain = [
    #    'conv_1', 'conv_1_a', 'conv_1_ts', 'flatten_1', 'dense_1', 'dense_1_a',
    #    'dense_1_ts'
    # ]
    x = tf.random.uniform((2, 10, 10, 3))
    y = m(x)
    nodes = set(['conv_1_a', 'dense_1_a'])
    y2, res_dict = m(x, return_nodes=nodes)
    self.assertAllEqual(y, y2)
    ts_layer = getattr(m, 'conv_1_ts')
    self.assertAllClose(tf.reduce_mean(res_dict['conv_1_a'], axis=[0, 1, 2]),
                        ts_layer.get_saved_values('mean'), atol=1e-4)
    ts_layer = getattr(m, 'dense_1_ts')
    self.assertAllClose(tf.reduce_mean(res_dict['dense_1_a'], axis=0),
                        ts_layer.get_saved_values('mean'), atol=1e-4)

  def testPropagateBias(self):
    for use_masked_layers in [True, False]:
      m = generic_convnet.GenericConvnet(
          name='test', use_masked_layers=use_masked_layers)
      dummy_input = tf.ones((32, 28, 28, 3), dtype=tf.float32)
      # Initialize model parameters.
      m(dummy_input)
      layer_conv_1 = m.conv_1.layer if use_masked_layers else m.conv_1
      layer_conv_2 = m.conv_2.layer if use_masked_layers else m.conv_2
      n_units = layer_conv_1.filters
      l_out = m.conv_1(dummy_input)
      zeros_with_a_single_one = [0] * (n_units - 1) + [1]
      mean_values = tf.cast(
          tf.broadcast_to(zeros_with_a_single_one, l_out.shape), tf.float32)
      # Default initialization for bias is all zeros.
      self.assertEqual(
          tf.math.count_nonzero(layer_conv_2.weights[1]).numpy(), 0)
      correct_propagated_tensor = m.conv_2(
          tf.keras.activations.relu(mean_values))
      # Since we have constant tensors `mean_values[:,:,:,i]` for each i, each
      # of the resulting channels should be equal to each other.
      self.assertAllEqual(correct_propagated_tensor[0, 0, 0, :],
                          correct_propagated_tensor[0, 0, 1, :])
      # Since all values in the last dimension same, we can take a single one,
      # this is equal to the mean anyway.
      correct_new_bias = correct_propagated_tensor[0, 0, 1, :]
      m.propagate_bias('conv_1', mean_values)
      self.assertAllClose(
          layer_conv_2.weights[1].numpy(), correct_new_bias.numpy(), atol=1e-04)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
