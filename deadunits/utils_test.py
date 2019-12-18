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
"""Tests for `deadunits.utils`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deadunits import utils
import mock
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class Score2BinaryMaskTest(parameterized.TestCase, tf.test.TestCase):

  def testCorrectness(self):
    score = tf.range(5)
    m = utils.create_binary_mask_from_scores(score, f=0.5)
    self.assertAllEqual(m, tf.constant([0, 0, 1, 1, 1]))

    m = utils.create_binary_mask_from_scores(score, n_zeros=3)
    self.assertAllEqual(m, tf.constant([0, 0, 0, 1, 1]))

    score = tf.range(4, 0, -1)
    m = utils.create_binary_mask_from_scores(score, f=0.5)
    self.assertAllEqual(m, tf.constant([1, 1, 0, 0]))

    score = tf.constant([0.1, 0.2, -0.5, 0.23, -5.1, 3])
    m = utils.create_binary_mask_from_scores(score, f=0.35)
    self.assertAllEqual(m, tf.constant([1, 1, 0, 1, 0, 1]))

  @parameterized.named_parameters(
      ('fraction0.5', 0.5, [[1, 0, 0], [1, 1, 0]]),
      ('fraction0.7', 0.7, [[0, 0, 0], [1, 1, 0]]),
      ('fraction0.2', 0.2, [[1, 1, 0], [1, 1, 1]]))
  def test2DScoresFractions(self, f, expected_mask):
    score = tf.constant([[0.6, 0.5, 0.1],
                         [0.9, 0.7, 0.4]])
    m = utils.create_binary_mask_from_scores(score, f=f)
    self.assertAllEqual(m, tf.constant(expected_mask))

  @parameterized.named_parameters(
      ('nzeros_2', 2, [[1, 1, 0], [1, 1, 0]]),
      ('nzeros_0', 0, [[1, 1, 1], [1, 1, 1]]),
      ('nzeros_6', 6, [[0, 0, 0], [0, 0, 0]]))
  def test2DScoresNZeros(self, n_zeros, expected_mask):
    score = tf.constant([[0.6, 0.5, 0.1],
                         [0.9, 0.7, 0.4]])
    m = utils.create_binary_mask_from_scores(score, n_zeros=n_zeros)
    self.assertAllEqual(m, tf.constant(expected_mask))

  def testInvalidArgs(self):
    score = tf.constant([0.1, 0.2, -0.5, 0.23, -5.1, 3])
    with self.assertRaises(AssertionError):
      utils.create_binary_mask_from_scores(score, 0)
    with self.assertRaises(AssertionError):
      utils.create_binary_mask_from_scores(score, 1)
    with self.assertRaises(AssertionError):
      utils.create_binary_mask_from_scores(score, -0.5)
    with self.assertRaises(AssertionError):
      utils.create_binary_mask_from_scores(tf.ones((3, 5)), 0)


class MaskAndBroadCastTest(tf.test.TestCase):

  def testCorrectnessNoBroadcast(self):
    vals = tf.range(1, 6)
    mask = tf.constant([0, 1, 1, 0, 0])
    res = utils.mask_and_broadcast(vals, mask)
    self.assertAllEqual(res, tf.constant([0, 2, 3, 0, 0]))
    res = utils.mask_and_broadcast(vals, mask, invert_mask=True)
    self.assertAllEqual(res, tf.constant([1, 0, 0, 4, 5]))

  def testCorrectnessWithBroadcast(self):
    vals = tf.range(1, 4)
    mask = tf.constant([0, 1, 1])
    res = utils.mask_and_broadcast(vals, mask, out_shape=[2, 3])
    self.assertAllEqual(res, tf.constant([[0, 2, 3], [0, 2, 3]]))
    res = utils.mask_and_broadcast(vals, mask, out_shape=[2, 3],
                                   invert_mask=True)
    self.assertAllEqual(res, tf.constant([[1, 0, 0], [1, 0, 0]]))


class BindGinParamsTest(tf.test.TestCase):

  @mock.patch('gin.bind_parameter')
  @mock.patch('gin.unlock_config')
  def testDefault(self, unlock_mock, bind_param_mock):
    c_dict = {'fun1.arg1': 1,
              'fun2.arg2': (2, 3)}
    utils.bind_gin_params(c_dict)
    self.assertEqual(bind_param_mock.call_count, len(c_dict))
    self.assertEqual(unlock_mock.call_count, len(c_dict))
    for k, v in c_dict.items():
      bind_param_mock.assert_any_call(k, v)

if __name__ == '__main__':
  tf.test.main()
