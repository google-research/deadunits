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
"""Has basic functions that return various scores for given variables.

This module has the following scoring functions for units of a given layer.
 - norm_scorer: calculates returns the norm of the weights.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def norm_score(tensor, order=2, **_):
  """Given an n_dim > 1 Tensor/Variable returns the norms on the last dimension.

  Norm is taken to the `order`th power and divided by number of elements

  If the n_dim > s, than it flattens the tensor (Nx...xMxK-> N*...*MxK).

  Args:
    tensor: an arbitrary 1d or 2d numerical tensor.
    order: order of the norm.
      Check valid ones from `tf.norm` for `ord` argument.
  Returns:
    1d Tensor with size same as the last dimension of `tensor`
    and same-type as `tensor`.
  """
  assert len(tensor.shape) > 1
  if len(tensor.shape) > 2:
    tensor = tf.reshape(tensor, (-1, tensor.shape[-1]))
  return tf.math.pow(tf.norm(tensor, axis=0, ord=order),
                     order) / tensor.shape[0]


def random_score(tensor, **_):
  """Given an n_dim > 1 Tensor/Variable returns random numbers.

  The random numbers are generated through tf.random_uniform.
  If a tensor with shape Nx...xMxK provided return shape would be K.

  Args:
    tensor: an arbitrary numerical tensor.
  Returns:
    1d Tensor sampled uniformly from range 0-1 with size same as the last
    dimension of `tensor`. The tensor is tf.float32 dtype.
  """
  assert len(tensor.shape) > 1
  return tf.cast(tf.random_uniform(tensor.shape[-1:]), dtype=tf.float32)
