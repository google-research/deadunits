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
"""Implements various utility functions.

This module has the following helper utility function:
 - score2binary_mask: Given a score tensor and a fraction, returns binary mask.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import gin
from six.moves import cPickle as pickle
import tensorflow.compat.v1 as tf
from tensorflow.contrib.eager.python import tfe as contrib_eager
tfe = contrib_eager


def create_binary_mask_from_scores(score_tensor, f=None, n_zeros=None):
  """Given an arbitrary tensor and a fraction returns the binary(0-1) tensor.

  Given a numerical tensor with any shape and N elements it returns a 0-1 tensor
  with same shape where a fraction `f` of the smallest values are set to zero.
  The indices which are set to 0 are selected according to the values provided
  in the `score_tensor`. We select smallest M=floor(N*f) indices.

  One should use either `f` or `n_zeros`; not together.
  Args:
    score_tensor: an arbitrary numerical tensor.
    f: fraction of zeros to be set: a number 0<f<1.
    n_zeros: int, number of zeros to be set: n_zeros<n_elements.
  Returns:
    a binary Tensor with a same shape as the `score_tensor` and same-type.
    It should have n_zero = floor(n_elements*f) many zeros.
  """
  # Assert only one of the name arguments is in use.
  assert (f is None) ^ (n_zeros is None)
  n_elements = tf.size(score_tensor).numpy()
  if f is not None:
    assert f > 0 and f < 1
    n_ones = n_elements - int(math.floor(n_elements * f))
  else:
    assert isinstance(n_zeros, int)
    n_ones = n_elements - n_zeros
  flat_score_tensor = (tf.reshape(score_tensor, [-1])
                       if len(score_tensor.shape) > 1 else score_tensor)
  mask = tf.Variable(tf.zeros_like(flat_score_tensor))
  _, indices = tf.nn.top_k(flat_score_tensor, n_ones)
  tf.scatter_update(mask, indices, 1)
  res = mask.read_value()
  # Reshaping back to the original shape.
  if len(score_tensor.shape) > 1:
    res = tf.reshape(res, score_tensor.shape)
  return res


def  mask_and_broadcast(values, mask, out_shape=None, invert_mask=False):
  """Returns mask*values broadcasted to the out_shape.

  Args:
    values: Tensor, values to be masked.
    mask: Tensor, consisting of {0,1}'s.
    out_shape: Tuple, this needs to be valid for `tf.broadcast_to` call.
    invert_mask: bool, if True it inverts the mask.
  Returns:
    Tensor, with shape `out_shape`.
  """
  if invert_mask:
    mask = mask * -1 + 1
  masked_vals = tf.multiply(values, mask)
  res = (masked_vals if out_shape is None
         else tf.broadcast_to(masked_vals, out_shape))
  return res


def pickle_object(obj, path):
  """Saves/Overwrites the given object creating a pickle file at `path`.

  Args:
    obj: object, any pickle-able python object.
    path: str, folder needs to exists
  """
  if tf.gfile.Exists(path):
    # This should happen when the process is preempted and continued
    # afterwards.
    tf.logging.warning('The file:%s exists, overwriting' % path)
  with tf.gfile.GFile(path, 'w') as f:
    pickle.dump(obj, f)


def bind_gin_params(xm_params):
  """Binding parameters from the given dictionary.

  Args:
    xm_params: dict, <key,value> pairs where key is a valid gin parameter.
  """
  tf.logging.info('xm_pararameters:\n')
  for param_name, param_value in xm_params.items():
    # Quote non-numeric values.
    tf.logging.info('%s=%s\n' % (param_name, param_value))
    with gin.unlock_config():
      gin.bind_parameter(param_name, param_value)
