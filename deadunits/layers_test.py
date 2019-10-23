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
"""Tests for `deadunits.layers`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
from deadunits import layers
from six.moves import range
from six.moves import zip
import tensorflow as tf


tf.enable_eager_execution()
FLAGS = flags.FLAGS


class MeanReplacerTest(tf.test.TestCase):

  def testArgs(self):
    l = layers.MeanReplacer(name='test', is_replacing=False)
    self.assertFalse(hasattr(l, 'n_units'))
    self.assertEqual(l.name, 'test')
    self.assertEqual(l.is_replacing, False)
    x = tf.ones((4, 3, 2))
    y = l(x)
    self.assertEqual(l.n_units, 2)
    self.assertAllEqual(y, x)

  def testSetActiveUnitsFailedAssertions(self):
    l = layers.MeanReplacer(is_replacing=True)
    x = tf.random_uniform((3, 5))
    #  Layer not yet built.
    with self.assertRaises(AssertionError):
      l.set_active_units([0])
    l(x)
    with self.assertRaises(AssertionError):
      l.set_active_units([-1])
    # 5 not valid
    with self.assertRaises(AssertionError):
      l.set_active_units([0, 5])
    # It should be a list
    with self.assertRaises(AssertionError):
      l.set_active_units(2)

  def testIsReplacing(self):
    l = layers.MeanReplacer(is_replacing=True)
    x = tf.random_uniform((3, 5))
    x_mean = tf.broadcast_to(tf.reduce_mean(x, axis=0), x.shape)
    y = l(x)
    l.set_active_units([0, 0, 2])
    self.assertSetEqual(set(l._active_units), set([0, 2]))
    l.set_active_units([4, 2])
    self.assertSetEqual(set(l._active_units), set([4, 2]))
    y = l(x)
    self.assertAllEqual(y[:, 0], x[:, 0])
    self.assertAllEqual(y[:, 1], x[:, 1])
    self.assertAllClose(y[:, 2], x_mean[:, 2])
    self.assertAllEqual(y[:, 3], x[:, 3])
    self.assertAllClose(y[:, 4], x_mean[:, 4])

  def testGetConfig(self):
    l = layers.MeanReplacer()
    expected_config = {'_active_units': [], 'is_replacing': False}
    self.assertDictContainsSubset(expected_config, l.get_config())
    l(tf.random_uniform((4, 5)))
    l.set_active_units([3, 2])
    expected_config = {'_active_units': [2, 3], 'is_replacing': False}
    self.assertDictContainsSubset(expected_config, l.get_config())


class TaylorScorerTest(tf.test.TestCase):

  def testArgs(self):
    l = layers.TaylorScorer(
        name='test',
        compute_removal_saliency=False,
        compute_mean_replacement_saliency=True)
    self.assertEqual(l.name, 'test')
    self.assertFalse(l.compute_removal_saliency)
    self.assertTrue(l.compute_mean_replacement_saliency)
    self.assertTrue(l.is_abs)
    self.assertFalse(l.save_l2norm)

  def testIdentity(self):
    l = layers.TaylorScorer(
        compute_removal_saliency=False, compute_mean_replacement_saliency=False)
    a = tf.random_uniform((3, 5))
    self.assertAllEqual(l(a), a)
    a = tf.random_uniform((3, 5, 5, 2))
    self.assertAllEqual(l(a), a)

  def testAggregationRS(self):
    l = layers.TaylorScorer(compute_removal_saliency=False,
                            compute_mean_replacement_saliency=False)
    x1 = tf.Variable(tf.random_uniform((3, 5)))
    with tf.GradientTape() as tape:
      y = l(x1, compute_removal_saliency=True)
      loss = tf.reduce_sum(y)
    # After forward pass it needs to be set to None
    tape.gradient(loss, x1)
    first_rs = l.get_saved_values('rs')
    # This should remove the previos rs, mrs, mean values.
    y = l(x1)
    self.assertIsNone(l.get_saved_values('rs'))
    # Another input
    x2 = tf.Variable(tf.random_uniform((3, 5)))
    with tf.GradientTape() as tape:
      y = l(x2, compute_removal_saliency=True)
      loss = tf.reduce_sum(y)
    # After forward pass it needs to be set to None
    tape.gradient(loss, x2)
    second_rs = l.get_saved_values('rs')
    # Aggregating once
    with tf.GradientTape() as tape:
      y = l(x1, compute_removal_saliency=True, aggregate_values=True)
      loss = tf.reduce_sum(y)
    tape.gradient(loss, x1)
    self.assertAllClose((first_rs + second_rs) / 2, l.get_saved_values('rs'))
    # Aggregating twice.
    with tf.GradientTape() as tape:
      y = l(x1, compute_removal_saliency=True, aggregate_values=True)
      loss = tf.reduce_sum(y)
    tape.gradient(loss, x1)
    self.assertAllClose((first_rs + first_rs + second_rs) / 3,
                        l.get_saved_values('rs'))

  def testAggregationMean(self):
    l = layers.TaylorScorer(compute_removal_saliency=False,
                            compute_mean_replacement_saliency=False)
    x1 = tf.random_uniform((3, 5))
    l(x1)
    first_mean = l.get_saved_values('mean')
    self.assertEqual(len(l._mean), 2)
    x2 = tf.random_uniform((6, 5))
    # Removing the previous one
    l(x2)
    second_mean = l.get_saved_values('mean')
    self.assertEqual(len(l._mean), 2)
    l(x1, aggregate_values=True)
    self.assertAllClose((first_mean + second_mean * 2) / 3,
                        l.get_saved_values('mean'))

  def testL2Norm(self):
    l = layers.TaylorScorer()
    x1 = tf.random_uniform((3, 5))
    l(x1)
    self.assertIsNone(l.get_saved_values('l2norm'))
    self.assertIsNone(l._l2norm)
    l(x1, save_l2norm=True)
    correct_l2normsquared = tf.square(tf.norm(x1, axis=0)) / x1.shape[0].value
    self.assertAllClose(l.get_saved_values('l2norm'),
                        correct_l2normsquared)
    x2 = tf.random_uniform((3, 5))
    l(x2, save_l2norm=True, aggregate_values=True)
    correct_l2normsquared2 = tf.square(tf.norm(x2, axis=0)) / x2.shape[0].value
    self.assertAllClose(l.get_saved_values('l2norm'),
                        (correct_l2normsquared+correct_l2normsquared2) / 2)

  def testIsMrsTrue(self):
    values = [tf.random_uniform((3, 5)),
              tf.random_uniform((3, 5, 5, 4))]
    for inp in values:
      n_dim = len(inp.shape)
      l = layers.TaylorScorer(compute_removal_saliency=False,
                              compute_mean_replacement_saliency=True)
      ones_channel = tf.ones(inp.shape.as_list()[:-1] + [1])
      inp_concat = tf.concat((inp, ones_channel), axis=n_dim-1)
      x = tf.Variable(inp_concat)
      x_mean = tf.reduce_mean(x, axis=0)
      with tf.GradientTape() as tape:
        y = l(x)
        loss = tf.reduce_sum(y)
      # After forward pass it needs to be set to None
      self.assertIsNone(l.get_saved_values('mrs'))
      dx = tape.gradient(loss, x)
      # RS should be set to None.
      self.assertIsNone(l.get_saved_values('rs'))
      # dy is just 1's.
      self.assertAllEqual(dx, tf.ones_like(inp_concat))
      # Normalize the sum.
      avg_change = x_mean - x
      if n_dim > 2:
        avg_change = tf.reduce_sum(avg_change, axis=list(range(1, n_dim - 1)))
      correct_mrs = tf.reduce_sum(tf.abs(avg_change),
                                  axis=0) / int(tf.size(x[Ellipsis, 0]))
      self.assertAllClose(correct_mrs, l.get_saved_values('mrs'))
      # Since last unit is just ones, replacing it with its mean has 0 penalty.
      self.assertEqual(l.get_saved_values('mrs')[-1].numpy(), 0.0)
      self.assertAllEqual(l(inp), inp)

  def testGetMeanValues(self):
    l = layers.TaylorScorer(compute_removal_saliency=False,
                            compute_mean_replacement_saliency=False)
    x = tf.random_uniform((3, 5))
    l(x)
    x_mean = tf.reduce_mean(x, axis=0)
    self.assertAllEqual(x_mean, l.get_saved_values('mean'))
    self.assertAllEqual(tf.broadcast_to(x_mean, x.shape),
                        l.get_saved_values('mean',
                                           broadcast_to_input_shape=True))
    rand_mask = tf.cast(tf.random_uniform(x_mean.shape[:1],
                                          dtype=tf.int32,
                                          maxval=2),
                        tf.float32)
    self.assertAllEqual(rand_mask * x_mean,
                        l.get_saved_values('mean', unit_mask=rand_mask))
    self.assertAllEqual(tf.broadcast_to(rand_mask * x_mean, x.shape),
                        l.get_saved_values('mean', unit_mask=rand_mask,
                                           broadcast_to_input_shape=True))

  def testGetMeanValuesAggregated(self):
    l = layers.TaylorScorer(compute_removal_saliency=False,
                            compute_mean_replacement_saliency=False)
    x1 = tf.random_uniform((3, 5))
    l(x1)
    x2 = tf.random_uniform((6, 5))
    l(x2, aggregate_values=True)

    correct_mean = tf.reduce_mean(tf.concat([x1, x2], 0), axis=0)
    self.assertAllClose(l.get_saved_values('mean'),
                        correct_mean)

  def testIsRsTrue(self):
    values = [tf.random_uniform((3, 5)),
              tf.random_uniform((3, 5, 5, 4))]
    for inp in values:
      n_dim = len(inp.shape)
      l = layers.TaylorScorer(compute_removal_saliency=True,
                              compute_mean_replacement_saliency=False)
      zeros_channel = tf.zeros(inp.shape.as_list()[:-1] + [1])
      inp_concat = tf.concat((inp, zeros_channel), axis=n_dim-1)
      x = tf.Variable(inp_concat)
      x_mean = tf.reduce_mean(x, axis=list(range(n_dim - 1)))
      with tf.GradientTape() as tape:
        y = l(x)
        loss = tf.reduce_sum(y)
      # After forward pass it needs to be set to None
      self.assertIsNone(l.get_saved_values('rs'))
      dx = tape.gradient(loss, x)
      # RS should be set to None.
      self.assertIsNone(l.get_saved_values('mrs'))
      # dy is just 1's.
      self.assertAllEqual(dx, tf.ones_like(inp_concat))
      # Normalize the sum.
      avg_change = -x
      if n_dim > 2:
        avg_change = tf.reduce_sum(avg_change, axis=list(range(1, n_dim - 1)))
      correct_rs = tf.reduce_sum(tf.abs(avg_change),
                                 axis=0) / int(tf.size(x[Ellipsis, 0]))
      self.assertAllClose(correct_rs, l.get_saved_values('rs'))
      # Since last unit is just ones, replacing it with its mean has 0 penalty.
      self.assertEqual(l.get_saved_values('rs')[-1].numpy(), 0.0)
      # We still expect the mean to be calculated
      self.assertAllEqual(x_mean, l.get_saved_values('mean'))
      self.assertAllEqual(l(inp), inp)

  def testIsAbsTrue(self):
    l = layers.TaylorScorer(is_abs=False, compute_removal_saliency=True)
    a = tf.constant([[-1, 0, 1],
                     [1, 0, 1]], dtype=tf.float32)
    x = tf.Variable(a)
    x_mean = tf.reduce_mean(x, axis=0)
    with tf.GradientTape() as tape:
      y = l(x)
      loss = tf.reduce_sum(y)
    # Before backward pass it is None.
    self.assertIsNone(l.get_saved_values('rs'))
    dx = tape.gradient(loss, x)
    # dy is just 1's.
    correct_rs = tf.constant([0, 0, -1])
    self.assertAllEqual(dx, tf.ones_like(a))
    self.assertAllEqual(correct_rs, l.get_saved_values('rs'))
    # We still expect the mean to be calculated
    self.assertAllEqual(x_mean, l.get_saved_values('mean'))

    # Lets do the same with  and get non_zero rs.
    with tf.GradientTape() as tape:
      y = l(x, is_abs=True)
      loss = tf.reduce_sum(y)
    # Before backward pass it is None.
    self.assertIsNone(l.get_saved_values('rs'))
    tape.gradient(loss, x)
    correct_rs = tf.constant([1, 0, 1])
    self.assertAllEqual(correct_rs, l.get_saved_values('rs'))

  def IsMrsTrueInModel(self):
    l = layers.TaylorScorer(
        compute_removal_saliency=True, compute_mean_replacement_saliency=True)
    l_before = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
    model = tf.keras.Sequential([
        l_before, l,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            32, activation=lambda x: tf.nn.log_softmax(x, axis=1))
    ])
    # Building the model. Don't need the return value.
    model(tf.random_uniform((3, 5)))
    l_before.weights[0].assign(
        tf.concat([l_before.weights[0][:, 1:],
                   tf.zeros((5, 1))], axis=1))

    x = tf.Variable(tf.random_uniform((3, 5)))
    a_mean = tf.reduce_mean(l_before(x), axis=0)
    with tf.GradientTape() as tape:
      y = model(x)
      loss = tf.reduce_sum(y)
    # Don't need the gradient itself, this would accumulate mrs_score.
    tape.gradient(loss, x)

    self.assertAllEqual(a_mean, l.get_saved_values('mean')[0])
    self.assertAllEqual(a_mean, l.get_saved_values('mean')[1])
    self.assertAllEqual(a_mean, l.get_saved_values('mean')[2])
    # Since last unit creates just whatever its bias is, it should be zero
    self.assertEqual(l.get_saved_values('mrs')[-1].numpy(), 0.0)

  def testGetConfig(self):
    l = layers.TaylorScorer()
    expected_config = {'is_abs': True, 'compute_removal_saliency': False,
                       'compute_mean_replacement_saliency': False,
                       'save_l2norm': False, 'trainable': False}
    self.assertDictContainsSubset(expected_config, l.get_config())


class MaskedLayerTest(parameterized.TestCase, tf.test.TestCase):

  def testNoMasking(self):
    layers_to_test = [(tf.keras.layers.Dense(12), tf.random_uniform((3, 5))),
                      (tf.keras.layers.Conv2D(12, 5),
                       tf.random_uniform((3, 10, 10, 5)))]

    for l, x in layers_to_test:
      ml = layers.MaskedLayer(l, name='test')
      self.assertFalse(hasattr(ml, 'mask_weight'))
      self.assertFalse(hasattr(ml, 'mask_bias'))
      self.assertAllEqual(l(x), ml(x))
      self.assertEqual(str(ml), 'MaskedLayer object, name=test')
      with tf.GradientTape() as tape1:
        y = l(x)
        loss1 = tf.reduce_sum(y)
      with tf.GradientTape() as tape2:
        yy = ml(x)
        loss2 = tf.reduce_sum(yy)
      # After forward pass it needs to be set to None
      g1 = tape1.gradient(loss1, x)
      g2 = tape2.gradient(loss2, x)
      self.assertAllEqual(g1, g2)

  def testSetWeightMasking(self):
    l = tf.keras.layers.Dense(12)
    ml = layers.MaskedLayer(l, name='test')
    with self.assertRaises(AssertionError):
      ml.set_mask(tf.zeros(4))
    x = tf.random_uniform((3, 5))
    # Bulding the layer and initializing the parameters.
    ml(x)
    with self.assertRaises(AssertionError):
      # Wrong mask_shape
      ml.set_mask(tf.zeros(3, 12))
    l_weights = ml.layer.weights[0]
    w_mask = tf.random_uniform(l_weights.shape, maxval=2, dtype=tf.int32)
    # To get pruned parameters.
    w_mask_not_bool = tf.logical_not(tf.cast(w_mask, tf.bool))
    ml.set_mask(w_mask)
    self.assertIsInstance(ml.mask_weight, tf.Variable)
    self.assertAllEqual(w_mask, ml.mask_weight.numpy())
    self.assertEqual(l_weights.dtype, ml.mask_weight.dtype)
    # Check the assign works.
    w_mask = tf.random_uniform(l_weights.shape, maxval=2, dtype=tf.int32)
    ml.set_mask(w_mask)
    self.assertAllEqual(w_mask, ml.mask_weight.numpy())
    self.assertAllEqual(ml.mask_bias.numpy(),
                        tf.ones_like(ml.mask_bias))
    # weights are not masked yet
    self.assertNotEqual(
        tf.count_nonzero(tf.boolean_mask(l_weights, w_mask_not_bool)).numpy(),
        0)

  def testSetBiasMasking(self):
    l = tf.keras.layers.Dense(12, bias_initializer='glorot_uniform')
    ml = layers.MaskedLayer(l, name='test')
    with self.assertRaises(AssertionError):
      ml.set_mask(tf.zeros(10), is_bias=True)
    x = tf.random_uniform((3, 5))
    # Bulding the layer and initializing the parameters.
    ml(x)
    with self.assertRaises(AssertionError):
      # Wrong mask_shape
      ml.set_mask(tf.zeros(5, 12), is_bias=True)
    l_bias = ml.layer.weights[1]
    b_mask = tf.random_uniform(l_bias.shape, maxval=2, dtype=tf.int32)
    # To get pruned parameters.
    b_mask_not_bool = tf.logical_not(tf.cast(b_mask, tf.bool))
    ml.set_mask(b_mask, is_bias=True)
    self.assertIsInstance(ml.mask_bias, tf.Variable)
    self.assertAllEqual(b_mask, ml.mask_bias.numpy())
    self.assertEqual(l_bias.dtype, ml.mask_bias.dtype)
    # Check the assign works.
    b_mask = tf.random_uniform(l_bias.shape, maxval=2, dtype=tf.int32)
    ml.set_mask(b_mask, is_bias=True)
    self.assertAllEqual(b_mask, ml.mask_bias.numpy())
    self.assertAllEqual(ml.mask_weight.numpy(),
                        tf.ones_like(ml.mask_weight))
    # weights are not masked yet
    self.assertNotEqual(
        tf.count_nonzero(tf.boolean_mask(l_bias, b_mask_not_bool)).numpy(), 0)

  def testMaskingForwardWeights(self):
    l = tf.keras.layers.Dense(12)
    ml = layers.MaskedLayer(l, name='test')
    x = tf.random_uniform((3, 5))
    # Bulding the layer and initializing the parameters.
    ml(x)
    l_weights = ml.layer.weights[0]
    w_mask = tf.random_uniform(l_weights.shape, maxval=2, dtype=tf.int32)
    w_mask_not_bool = tf.logical_not(tf.cast(w_mask, tf.bool))
    ml.set_mask(w_mask)
    with tf.GradientTape() as tape:
      y = ml(x)
      # All weights under the mask expected to be zero after forward call.
      self.assertEqual(
          tf.count_nonzero(tf.boolean_mask(l_weights, w_mask_not_bool)).numpy(),
          0)
      loss = tf.reduce_sum(y)
    grads = tape.gradient(loss, l.variables)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer.apply_gradients(list(zip(grads, l.variables)))
    # Weights are updated and they are not necesarrily zero anymore.
    self.assertNotEqual(
        tf.count_nonzero(tf.boolean_mask(l_weights, w_mask_not_bool)).numpy(),
        0)
    # All weights under the mask expected to be zero after forward call.
    # Don't need the return value.
    ml(x)
    self.assertEqual(
        tf.count_nonzero(tf.boolean_mask(l_weights, w_mask_not_bool)).numpy(),
        0)

  def testMaskingForwardBias(self):
    l = tf.keras.layers.Dense(12)
    ml = layers.MaskedLayer(l, name='test')
    x = tf.random_uniform((3, 5))
    # Bulding the layer and initializing the parameters.
    ml(x)
    l_bias = ml.layer.weights[1]
    b_mask = tf.random_uniform(l_bias.shape, maxval=2, dtype=tf.int32)
    # To get pruned parameters.
    b_mask_not_bool = tf.logical_not(tf.cast(b_mask, tf.bool))
    ml.set_mask(b_mask, is_bias=True)
    with tf.GradientTape() as tape:
      y = ml(x)
      # All weights under the mask expected to be zero after forward call.
      self.assertEqual(
          tf.count_nonzero(tf.boolean_mask(l_bias, b_mask_not_bool)).numpy(), 0)
      loss = tf.reduce_sum(y)
    grads = tape.gradient(loss, l.variables)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer.apply_gradients(list(zip(grads, l.variables)))
    # Weights are updated and they are not necesarrily zero anymore.
    self.assertNotEqual(
        tf.count_nonzero(tf.boolean_mask(l_bias, b_mask_not_bool)).numpy(), 0)
    # All weights under the mask expected to be zero after forward call.
    # Don't need the return value.
    ml(x)
    self.assertEqual(
        tf.count_nonzero(tf.boolean_mask(l_bias, b_mask_not_bool)).numpy(), 0)

  @parameterized.named_parameters(
      ('_1', tf.ones(6), True, True, 0.0),
      ('_2', tf.ones((2, 6)), False, True, 0.0),
      # There are 18 parameters in total, 6 of them being masked: 6/18 = 1/3.
      ('_3', tf.zeros(6), True, False, 1/3.),
      ('_4', tf.zeros(6), True, True, 0.0),
      ('_5', tf.zeros((2, 6)), False, False, 2/3),
      ('_6', tf.zeros((2, 6)), False, True, 1.0),
      ('_7', tf.constant([1, 0, 1, 0, 0, 1]), True, False, 1/6.))  # 3/18 = 1/6
  def testGetSparsity(self, mask, is_bias, weight_only, sparsity):
    layer = tf.keras.layers.Dense(6, bias_initializer='glorot_uniform')
    masked_layer = layers.MaskedLayer(layer, name='test')
    x = tf.random_uniform((3, 2))
    # Bulding the layer and initializing the parameters.
    masked_layer(x)
    masked_layer.set_mask(mask, is_bias=is_bias)
    self.assertEqual(masked_layer.get_sparsity(weight_only=weight_only),
                     sparsity)

  def testCheckpoint(self):
    test_path = FLAGS.test_tmpdir
    masked_layer_1 = layers.MaskedLayer(tf.layers.Dense(4))
    checkpoint = tf.train.Checkpoint(model=masked_layer_1)
    x = tf.ones((2, 5))
    y_1 = masked_layer_1(x)
    custom_mask = tf.constant([1, 1, 0, 1])
    masked_layer_1.set_mask(custom_mask, is_bias=True)
    c_path = checkpoint.save(test_path)

    # Loading
    masked_layer_2 = layers.MaskedLayer(tf.layers.Dense(4),
                                        mask_initializer=tf.initializers.zeros)
    checkpoint = tf.train.Checkpoint(model=masked_layer_2)
    checkpoint.restore(c_path)

    y_2 = masked_layer_2(x)
    self.assertAllEqual(y_2, y_1)
    self.assertAllEqual(masked_layer_2.mask_weight.numpy(),
                        masked_layer_1.mask_weight.numpy())
    self.assertAllEqual(masked_layer_2.mask_bias.numpy(),
                        masked_layer_1.mask_bias.numpy())
    self.assertAllEqual(masked_layer_2.weights[0].numpy(),
                        masked_layer_1.weights[0].numpy())
    self.assertAllEqual(masked_layer_2.weights[1].numpy(),
                        masked_layer_1.weights[1].numpy())
if __name__ == '__main__':
  tf.test.main()
