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
"""Tests for `deadunits.unitscorers`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deadunits import unitscorers
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.eager.python import tfe as contrib_eager
tfe = contrib_eager
tf.enable_eager_execution()


class NormScoreTest(tf.test.TestCase):

  def testDefaultL2(self):
    a = tf.reshape(tf.range(30, dtype=tf.float32), (3, 2, 5))
    v = tfe.Variable(a)
    scora_a = unitscorers.norm_score(a)
    score_v = unitscorers.norm_score(v)
    self.assertAllEqual(scora_a, score_v)
    l2_norm_i0 = sum([a * a for a in [0, 5, 10, 15, 20, 25]]) / 6.0
    self.assertAllClose(scora_a[0], l2_norm_i0, atol=1e-3)

  def testL1Norm(self):
    a = tf.reshape(tf.range(60, dtype=tf.float32), (3, 2, 2, 5)) - 30.0
    l1_norm_i3 = np.sum(np.abs(a[:, :, :, 3].numpy())) / 12.0
    scora_a = unitscorers.norm_score(a, order=1)
    self.assertAllClose(l1_norm_i3, scora_a[3], atol=1e-3)
    scora_1d = unitscorers.norm_score(tf.reshape(tf.range(5), (5, 1)), order=1)
    self.assertAllClose(2, scora_1d[0], atol=1e-3)

  def testShapeAssert(self):
    with self.assertRaises(AssertionError):
      unitscorers.norm_score(tf.ones(10))
    with self.assertRaises(AssertionError):
      unitscorers.norm_score(tf.zeros(1))


class RandomScoreTest(tf.test.TestCase):

  def testBasic(self):
    a = tf.reshape(tf.range(30, dtype=tf.float32), (3, 2, 5))
    score_a = unitscorers.random_score(a)
    score_b = unitscorers.random_score(a)
    self.assertNotAllClose(score_a, a[Ellipsis, -1])
    self.assertNotAllClose(score_b, score_a)
    self.assertAllEqual(score_a.get_shape().as_list(),
                        a.get_shape().as_list()[-1:])

if __name__ == '__main__':
  tf.test.main()
