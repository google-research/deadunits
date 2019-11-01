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
"""Tests for `deadunits.model_load_test`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from absl.testing import parameterized
from deadunits import layers
from deadunits import model_defs
from deadunits import model_load
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe as contrib_eager
tfe = contrib_eager
tf.enable_eager_execution()


class GetModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_cub200', 'small_conv', 'cub200', 200),
      ('_cifar10', 'medium_conv', 'cifar10', 10),
      ('_imagenet', 'small_conv', 'imagenet', 1000))
  def testInit(self, model_arch_name, dataset_name, n_classes):
    model = model_load.get_model(model_arch_name=model_arch_name,
                                 dataset_name=dataset_name)
    # Check output layer has correct number of units.
    self.assertEqual(model.output_1.units, n_classes)
    # Lets check the number of channels in the first conv layer.
    arch_definition = getattr(model_defs, model_arch_name)
    n_units_in_first_layer = arch_definition[0][1]
    self.assertEqual(model.conv_1.filters,
                     n_units_in_first_layer)

  @parameterized.named_parameters(
      ('_invalid_data', 'cifar100', 'small_conv'),  # Not valid dataset
      ('_invalid_model_name', 'cifar10', 'foo_16'))  # Invalid model name
  def testValueError(self, dataset_name, model_arch_name):
    with self.assertRaises(ValueError):
      _ = model_load.get_model(model_arch_name=model_arch_name,
                               dataset_name=dataset_name)

  def _create_and_save_model(self, model):
    save_path = self.get_temp_dir()
    shutil.rmtree(save_path, ignore_errors=True)
    os.mkdir(save_path)
    checkpoint = tfe.Checkpoint(model=model)
    checkpoint.save(os.path.join(save_path, 'ckpt'))
    load_path = tf.train.latest_checkpoint(save_path)
    return load_path

  def testSaveAndLoad(self):
    model_arch_name = 'small_conv'
    dataset_name = 'cifar10'
    # Let's create and save model.
    model = model_load.get_model(model_arch_name=model_arch_name,
                                 dataset_name=dataset_name)
    load_path = self._create_and_save_model(model)
    model_loaded = model_load.get_model(model_arch_name=model_arch_name,
                                        dataset_name=dataset_name,
                                        load_path=load_path)
    # If loaded correctly all should be equal.
    self.assertAllClose(model.conv_1.weights[0].numpy(),
                        model_loaded.conv_1.weights[0].numpy())

  @parameterized.named_parameters(('_True', True,), ('_False', False,))
  def testPreparePruning(self, is_prepared):
    # Arrange
    model_arch_name = 'small_conv'
    dataset_name = 'cifar10'
    # Let's create and save model.
    model = model_load.get_model(model_arch_name=model_arch_name,
                                 dataset_name=dataset_name)
    load_path = self._create_and_save_model(model)
    # Act
    model_loaded = model_load.get_model(model_arch_name=model_arch_name,
                                        dataset_name=dataset_name,
                                        load_path=load_path,
                                        prepare_for_pruning=is_prepared)
    # The architecture defined in `model_defs.small_conv` would generate the
    # 3 conv layers and 2 dense layers with following attribute names.
    all_layers = ['conv_1', 'conv_2', 'conv_3', 'dense_1', 'dense_2']
    # Assert, whether the new layers are injected or not
    for l_name in all_layers:
      self.assertEqual(('%s_ts' % l_name) in model_loaded.forward_chain,
                       is_prepared)
      self.assertEqual(isinstance(getattr(model_loaded, l_name),
                                  layers.MaskedLayer),
                       is_prepared)

if __name__ == '__main__':
  tf.test.main()
