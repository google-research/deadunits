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
"""Tests for `deadunits.train_utils`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deadunits import train_utils
import mock
from six.moves import range
import tensorflow.compat.v2 as tf


class PruningScheduleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_1', 5, [('layer_1', 0.3, 0.1), ('layer_2', 0.5, 0.3)], 10,
       {5: [('layer_1', 0.1)], 15: [('layer_2', 0.3)],
        25: [('layer_1', 0.2)], 35: [('layer_2', 0.5)],
        45: [('layer_1', 0.3)]}),
      ('_2', 5, [('layer_1', 0.3, 0.1), ('layer_2', 0.5, 0.3)], 0,
       {5: [('layer_1', 0.1), ('layer_2', 0.3), ('layer_1', 0.2),
            ('layer_2', 0.5), ('layer_1', 0.3)]}),
      ('_3', 5, [('layer_1', 0.25, 0.1)], 1,
       {5: [('layer_1', 0.1)], 6: [('layer_1', 0.2)],
        7: [('layer_1', 0.25)]}),
      ('_4', 2, [('layer_1', 0.25, 0.5)], 1,
       {2: [('layer_1', 0.25)]}))
  def testSchedules(self, start_iteration, target_fractions, n_finetune,
                    expected_results):
    generated_results = train_utils.pruning_schedule(start_iteration,
                                                     target_fractions,
                                                     n_finetune)
    self.assertDictEqual(expected_results, generated_results)

  def testEmptyTargetSet(self):
    generated_results = train_utils.pruning_schedule(5, [], 2)
    self.assertDictEqual(generated_results, {})


class CrossEntropyLossTest(tf.test.TestCase):

  def get_logits(self, n_sample, n_out):
    return tf.reshape(
        tf.range(int(n_sample) * int(n_out), dtype=tf.float32),
        (n_sample, n_out))

  def _create_mock_model(self, n_out=10):

    def f_side_effect(x, **_):
      return self.get_logits(x.shape[0], n_out)

    model = mock.Mock(side_effect=f_side_effect)
    model.get_layer_keys = mock.Mock(return_value=['conv_1', 'conv_2'])
    return model

  def testDefaultSingleBatch(self):
    # Model returns logits for 3 samples and 5 classes.
    n_sample, n_out = 8, 10
    model = self._create_mock_model(n_out=n_out)
    x = tf.ones((n_sample, 2))
    y = tf.ones((n_sample,), dtype=tf.int32)
    loss, acc, total_samples = train_utils.cross_entropy_loss(model, (x, y))
    self.assertEqual(total_samples, n_sample)
    self.assertIsNone(acc)
    logits = self.get_logits(n_sample, n_out)
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    true_loss = cce(y, logits)
    self.assertAllClose(loss, true_loss)
    model.assert_called_once_with(
        x,
        training=False,
        compute_mean_replacement_saliency=False,
        compute_removal_saliency=False,
        is_abs=True,
        aggregate_values=False)

  def testDefaultAccuracy(self):
    # Model returns logits for 3 samples and 5 classes.
    n_sample, n_out = 8, 10
    model = self._create_mock_model(n_out=n_out)
    x = tf.ones((n_sample, 2))
    y = tf.ones((n_sample,), dtype=tf.int32)
    _, acc, total_samples = train_utils.cross_entropy_loss(
        model, (x, y), calculate_accuracy=True)
    self.assertEqual(total_samples, n_sample)
    logits = self.get_logits(n_sample, n_out)
    predictions = tf.cast(tf.argmax(logits, 1), y.dtype)
    acc_obj = tf.keras.metrics.Accuracy()
    acc_obj.update_state(tf.squeeze(y), predictions)
    true_acc = acc_obj.result().numpy()
    self.assertAllClose(acc, true_acc)

  def testSingleBatch(self):
    # Model returns logits for 3 samples and 5 classes.
    n_sample, n_out = 8, 10
    model = self._create_mock_model(n_out=n_out)
    x = tf.ones((n_sample, 2))
    y = tf.ones((n_sample,), dtype=tf.int32)
    loss, acc, total_samples = train_utils.cross_entropy_loss(
        model, (x, y), calculate_accuracy=True)
    model2 = self._create_mock_model(n_out=n_out)
    d = tf.data.Dataset.from_tensor_slices((x, y))
    loss2, acc2, total_samples2 = train_utils.cross_entropy_loss(
        model2, d.batch(n_sample), calculate_accuracy=True)
    self.assertEqual(total_samples2, total_samples)
    self.assertAllClose(acc, acc2)
    self.assertAllClose(loss, loss2)

  def testDefaultBatch(self):
    # Model returns logits for 3 samples and 5 classes.
    n_sample, chunk_size, n_out = 42, 10, 10
    model = self._create_mock_model(n_out=n_out)
    x_all = tf.ones((n_sample, 2))
    y_all = tf.ones((n_sample,), dtype=tf.int32)
    d = tf.data.Dataset.from_tensor_slices((x_all, y_all))
    loss, acc, total_samples = train_utils.cross_entropy_loss(
        model, d.batch(chunk_size))
    self.assertEqual(total_samples, n_sample)
    self.assertIsNone(acc)
    all_logits = []
    for i in range(0, n_sample, chunk_size):
      c_size = min(n_sample, i + chunk_size) - i
      all_logits.append(self.get_logits(c_size, n_out))
    logits = tf.concat(all_logits, 0)
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    true_loss = cce(y_all, logits)
    self.assertAllClose(loss, true_loss, atol=1e-4)
    self.assertEqual(model.call_count, 5)
    model.get_layer_keys.assert_not_called()
    train_utils.cross_entropy_loss(model, d.batch(chunk_size),
                                   aggregate_values=True)
    model.get_layer_keys.assert_called_once()
    model.conv_1.reset_saved_values.assert_called_once()
    model.conv_2.reset_saved_values.assert_called_once()


class GetOptimizerTest(tf.test.TestCase):

  def testNoSchedule(self):
    optimizer = train_utils.get_optimizer(0)
    self.assertEqual(optimizer.learning_rate, 0.01)
    optimizer = train_utils.get_optimizer(55)
    self.assertEqual(optimizer.learning_rate, 0.01)

    optimizer = train_utils.get_optimizer(22, lr=0.1)
    self.assertEqual(optimizer.learning_rate, 0.1)

  def testSchedule(self):
    test_schedules = [[[3, 0.5], [6, 0.25]],
                      [[3, 0.5], [6, 1]],
                      [[2, 0.1]],
                      [[0, 0.1]]]
    lr = 0.1
    for schedule in test_schedules:
      test_results = {i: lr for i in range(10)}
      for epoch_j, factor in schedule:
        for i in range(epoch_j, 10):
          test_results[i] = factor*lr
      for i in range(10):
        optimizer = train_utils.get_optimizer(i, lr=lr, schedule=schedule)
        self.assertEqual(optimizer.learning_rate, test_results[i])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
