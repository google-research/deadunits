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
"""Implements various utility functions for loading and transforming models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from deadunits import data
from deadunits import generic_convnet
from deadunits import model_defs
import gin
from six.moves import zip
import tensorflow as tf


tfe = tf.contrib.eager
INPUT_SHAPES = {'cub200': (2, 224, 224, 3),
                'cifar10': (2, 32, 32, 3),
                'imagenet': (2, 224, 224, 3)}


@gin.configurable
def get_model(model_arch_name=gin.REQUIRED,
              dataset_name=gin.REQUIRED,
              load_path=None,
              prepare_for_pruning=False):
  """Creates or loads the model and returns it.

  If the model does not match with the saved, version, usually no error or
    warning is made, so be careful, CHECK YOUR VARIABLES.

  Args:
    model_arch_name: str, definition from .model_defs.py file.
    dataset_name: str, either 'cifar10' or 'imagenet'.
    load_path: str, checkpoint name/path to be load.
    prepare_for_pruning: bool, if True the loaded model is copied in-to one with
      TaylorScorer layer and layers are wrapped with MaskedLayer.

  Returns:
    generic_convnet.GenericConvnet, initialized or loaded model.
  Raises:
    ValueError: when the args doesn't match the specs.
    IOError: when there is no checkpoint found at the path given.
  """
  if dataset_name not in INPUT_SHAPES:
    raise ValueError('Dataset_name: %s is not one of %s' %
                     (dataset_name, list(INPUT_SHAPES.keys())))
  if not hasattr(model_defs, model_arch_name):
    raise ValueError('Model name: %s...not in model_defs.py' % model_arch_name)
  n_classes = data.N_CLASSES_BY_DATASET[dataset_name]
  model_arch = (
      getattr(model_defs, model_arch_name) + [['O', n_classes]])
  model = generic_convnet.GenericConvnet(
      model_arch=model_arch, name=model_arch_name)
  dummy_var = tf.zeros(INPUT_SHAPES[dataset_name])
  # Initializing model.
  model(dummy_var)
  if load_path is not None:
    checkpoint = tfe.Checkpoint(model=model)
    if not tf.train.checkpoint_exists(load_path):
      raise IOError('No checkpoint at: %s' % load_path)
    checkpoint.restore(load_path)
    if prepare_for_pruning:
      old_model = model
      model = generic_convnet.GenericConvnet(
          model_arch=model_arch, name=model_arch_name,
          use_taylor_scorer=True,
          use_masked_layers=True)
      model(dummy_var)
      for v1, v2 in zip(old_model.trainable_variables,
                        model.trainable_variables):
        v2.assign(v1)
  return model
