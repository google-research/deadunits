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

"""Tests for `deadunits.model_defs`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deadunits import model_defs
from deadunits.generic_convnet import GenericConvnet
import tensorflow as tf
tf.enable_eager_execution()


class ModelDefsTest(tf.test.TestCase):

  def testSmallConvs(self):
    x = tf.random_uniform((32, 32, 32, 3))
    cifar10_small = model_defs.small_conv + [['O', 10]]
    m = GenericConvnet(model_arch=cifar10_small)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])
    cifar10_medium = model_defs.small_conv + [['O', 10]]
    m = GenericConvnet(model_arch=cifar10_medium)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])

  def testVggs(self):
    x = tf.random_uniform((32, 32, 32, 3))
    vgg_11 = model_defs.vgg_11 + [['O', 10]]
    m = GenericConvnet(model_arch=vgg_11)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])
    vgg_13 = model_defs.vgg_13 + [['O', 10]]
    m = GenericConvnet(model_arch=vgg_13)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])
    vgg_16 = model_defs.vgg_16 + [['O', 10]]
    m = GenericConvnet(model_arch=vgg_16)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])
    m = GenericConvnet(model_arch=vgg_16, use_batchnorm=True)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])

  def testAlexNet(self):
    x = tf.random_uniform((32, 32, 32, 3))
    alexnet_cifar10 = model_defs.alexnet_cifar10 + [['O', 10]]
    m = GenericConvnet(model_arch=alexnet_cifar10)
    y = m(x)
    self.assertAllEqual(y.get_shape().as_list(), [32, 10])

if __name__ == '__main__':
  tf.test.main()
