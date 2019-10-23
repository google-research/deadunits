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
"""Implements common architectures in a generic way using tf.keras.Model.

Each generic model inherits from `tf.keras.Model`.
You can use following generic_models for now:

- GenericConvnet: sequential models include Conv2D's + Dense's.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from deadunits.layers import MaskedLayer
from deadunits.layers import MeanReplacer
from deadunits.layers import TaylorScorer
import gin
from six.moves import range
import tensorflow as tf

_default_generic_convnet_arch = [['C', 16, 5, {}], ['MP', 2, 2],
                                 ['C', 32, 5, {}], ['MP', 2, 2], ['F'],
                                 ['D', 256], ['O', 10]]


@gin.configurable
class GenericConvnet(tf.keras.Model):
  """Creates a tf.keras.Model from according to the flags and arch provided.

  """
  _allowed_layers = {
      'C': 'conv',
      'D': 'dense',
      'MP': 'maxpool',
      'DO': 'dropout',
      'O': 'output',
      'F': 'flatten',
      'GA': 'gap',
  }

  # Each layer should have the following form in the arch definition.
  # 'C': Conv2D layer in the form ['C', n_units, filter_shape, **kwargs]
  # 'MP': MaxPool2D layer in the form ['MP', pool_size, strides, **kwargs]
  # 'D': Dense layer in the form ['D', n_units]
  # 'DO': Dropout layer in the form ['DO', rate]
  # 'F': Flatten layer in the form ['F']
  # 'GA': Global average pooling 2D in the form ['GA']
  # 'O': Dense layer with no activation in the form ['O', n_units]
  def __init__(self,
               model_arch=None,
               name='GenericCifarConvnet',
               f_activation=tf.keras.activations.relu,
               use_batchnorm=False,
               bn_is_affine=False,
               use_dropout=False,
               dropout_rate=0.5,
               use_mean_replacer=False,
               use_taylor_scorer=False,
               use_masked_layers=False):
    """Initializes GenericConvnet instance with correct layers.

    Args:
      model_arch: list, consists of lists defining the cascaded network. refer
        to `GenericConvnet._allowed_layers`.
      name: str, name of the model.
      f_activation: function, from tf.keras.activations
      use_batchnorm: bool, if True BatchNormalization layer is used.
      bn_is_affine: bool, if True BatchNormalization performs affine
        transformation after the normalization.
      use_dropout: bool, if True Dropout layer is used.
      dropout_rate: float, dropout fraction for the Dropout layer.
      use_mean_replacer: bool, if True MeanReplacer layer is used after each
        layer.
      use_taylor_scorer: bool, if True TaylorScorer layer is used after each
        layer.
      use_masked_layers: bool, if True each layer is wrapped with MaskedLayer.

    Raises:
      AssertionError: when the provided `model_arch` is not valid.
    """
    if model_arch is None:
      model_arch = _default_generic_convnet_arch
    self._check_arch(model_arch)
    super(GenericConvnet, self).__init__(name=name)
    # Initial configration is saved to be able to clone the model.
    self.init_config = dict([('model_arch', model_arch), ('name', name),
                             ('f_activation', f_activation),
                             ('use_batchnorm', use_batchnorm),
                             ('bn_is_affine', bn_is_affine),
                             ('use_dropout', use_dropout),
                             ('dropout_rate', dropout_rate),
                             ('use_mean_replacer', use_mean_replacer),
                             ('use_taylor_scorer', use_taylor_scorer),
                             ('use_masked_layers', use_masked_layers)])
    # Wrap the layers if asked.
    wrapper = lambda l: MaskedLayer(l) if use_masked_layers else l
    # Forward chain has the attribute names in order and used to orchestrate
    # the forward pass.
    forward_chain = []
    for t in model_arch:
      # The order is:
      # Layer + bn + Activation + taylorScorer + meanReplacer + Dropout
      l_type = t[0]
      l_name = self._get_layer_name(l_type)
      forward_chain.append(l_name)
      # If F(flatten) or O(output), we don't have extra layers(dropout,bn,etc..)
      if l_type == 'F':
        setattr(self, l_name, tf.keras.layers.Flatten())
      elif l_type == 'GA':
        setattr(self, l_name, tf.keras.layers.GlobalAvgPool2D())
      elif l_type == 'MP':
        setattr(self, l_name, tf.keras.layers.MaxPool2D(t[1], t[2]))
      elif l_type == 'O':
        setattr(self, l_name, tf.keras.layers.Dense(t[1], activation=None))
      elif l_type == 'DO':
        setattr(self, l_name, tf.keras.layers.Dropout(t[1]))
      else:
        if l_type == 'C':
          setattr(
              self, l_name,
              wrapper(
                  tf.keras.layers.Conv2D(t[1], t[2], activation=None, **t[3])))
        elif l_type == 'D':
          setattr(self, l_name,
                  wrapper(tf.keras.layers.Dense(t[1], activation=None)))

        if use_batchnorm:
          c_name = l_name + '_bn'
          setattr(
              self, c_name,
              tf.keras.layers.BatchNormalization(
                  center=bn_is_affine, scale=bn_is_affine))
          forward_chain.append(c_name)
        # Add activation
        c_name = l_name + '_a'
        setattr(self, c_name, f_activation)
        forward_chain.append(c_name)
        if use_taylor_scorer:
          c_name = l_name + '_ts'
          setattr(self, c_name, TaylorScorer())
          forward_chain.append(c_name)
        if use_mean_replacer:
          c_name = l_name + '_mr'
          setattr(self, c_name, MeanReplacer())
          forward_chain.append(c_name)
        if use_dropout:
          c_name = l_name + '_dr'
          setattr(self, c_name, tf.keras.layers.Dropout(dropout_rate))
          forward_chain.append(c_name)
    self.forward_chain = forward_chain

  def call(self,
           inputs,
           training=False,
           compute_mean_replacement_saliency=False,
           compute_removal_saliency=False,
           is_abs=True,
           aggregate_values=False,
           is_replacing=False,
           return_nodes=None):
    # We need to save the first_input for initiliazing our clone (see .clone()).
    if not hasattr(self, 'first_input'):
      self.first_input = inputs
    x = inputs
    return_dict = {}
    for l_name in self.forward_chain:
      node = getattr(self, l_name)
      if isinstance(node, MeanReplacer):
        x = node(x, is_replacing=is_replacing)
      elif isinstance(node, TaylorScorer):
        x = node(
            x,
            compute_mean_replacement_saliency=compute_mean_replacement_saliency,
            compute_removal_saliency=compute_removal_saliency,
            is_abs=is_abs,
            aggregate_values=aggregate_values)
      elif isinstance(
          node, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
        x = node(x, training=training)
      else:
        x = node(x)
      if return_nodes and l_name in return_nodes:
        return_dict[l_name] = x
    if return_nodes:
      return x, return_dict
    else:
      return x

  def propagate_bias(self, l_name, input_tensor):
    """Propagates the given input to the bias of the next unit.

    We expect `input_tensor` having constant values at `input_tensor[...,i]` for
      every unit `i`. However this is not checked and if it is not constant,
      mean of the all values are used to update the bias.

    If input_tensor casted into same type as the parameters of the `l_name`.

    Args:
      l_name: str, name of a MaskedLayer such that `hasattr(self, l_name)` is
        True.
      input_tensor: Tensor, same shape as the output shape of the l_name. It
        should also be a float type. i.e. tf.float16/32/64.

    Raises:
      ValueError: when the l_name is not in the `self.forward_chain` or if
        there is no parameterized layer exists after `l_name`.
      AssertionError: when the input_tensor is not float type.
    """
    assert (input_tensor.dtype in [tf.float16, tf.float32, tf.float64])
    current_i = self.forward_chain.index(l_name) + 1
    if current_i == len(self.forward_chain):
      raise ValueError('Output layer cannot propagate bias')
    next_layer = getattr(self, self.forward_chain[current_i])
    forward_tensor = input_tensor
    # Including `tf.keras.layers.Dense`, too; since the output layer(Dense)
    # is not wrapped with `MaskedLayer`.
    parametered_layers = (MaskedLayer, tf.keras.layers.Dense,
                          tf.keras.layers.Conv2D)
    while not isinstance(next_layer, parametered_layers):
      forward_tensor = next_layer(forward_tensor)
      current_i += 1
      if current_i == len(self.forward_chain):
        raise ValueError('No appropriate layer exists after'
                         '%s to propagate bias.' % l_name)
      next_layer = getattr(self, self.forward_chain[current_i])
    # So now we have propageted bias + currrent_bias. This should be our new
    # bias.
    forward_tensor = next_layer(forward_tensor)
    # During Mean Replacement, forward_tensor[...,i] should be a constant
    # tensor, but it is not verified.
    bias2add = tf.reduce_mean(
        forward_tensor, axis=list(range(forward_tensor.shape.ndims - 1)))
    if isinstance(next_layer, MaskedLayer):
      next_layer.layer.weights[1].assign(bias2add)
    else:
      next_layer.weights[1].assign(bias2add)

  def get_allowed_layer_keys(self):
    return list(self._allowed_layers.keys())

  def get_layer_keys(self, layer_type, name_filter=lambda _: True):
    """Returns a list of layer_names matching the type and passing the filter.

    `self.forward_chain` is filtered by type and layer_name.
    Args:
      layer_type: layer class to be matched.
      name_filter: function, returning bool given a layer_name.
    """
    res = []
    for l_name in self.forward_chain:
      if name_filter(l_name) and isinstance(getattr(self, l_name), layer_type):
        res.append(l_name)
    return res

  def _get_layer_name(self, l_type):
    """Returns names for different layers by incrementing the counter.

    Args:
      l_type: str from self._allowed_layers.keys()

    Returns:
      attr_name: str unique attr name for the layer
    """
    if not hasattr(self, 'layer_name_counter'):
      self.layer_name_counter = {k: 1 for k in self._allowed_layers.keys()}
    i = self.layer_name_counter[l_type]
    self.layer_name_counter[l_type] += 1
    return '%s_%d' % (self._allowed_layers[l_type], i)

  def clone(self):
    new_model = GenericConvnet(**self.init_config)
    # Initilize the new_model params.
    new_model(self.first_input)
    new_model.set_weights(self.get_weights())
    return new_model

  def _check_arch(self, arch):
    """Checks the arch provided has the right form.

    For some reason tensorflow wraps every list/dict to make it checkpointable.
    For that reason we are using the super classes from collections module.
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpointable/data_structures.py
    Args:
      arch: architecture list.

    Raises:
      AssertionError: If the architecture list is not in the right format.
    """
    assert arch is not None
    assert len(arch) >= 1
    for t in arch:
      assert isinstance(t, collections.MutableSequence)
      assert len(t) >= 1
      assert t[0] in self.get_allowed_layer_keys()
      if t[0] == 'C':
        assert len(t) == 4
        assert isinstance(t[1], int)
        # t[2] can be an int or list of two integers.
        assert (isinstance(t[2], int) or
                (isinstance(t[2], collections.MutableSequence) and
                 len(t[2]) == 2) and all(isinstance(x, int) for x in t[2]))
        assert isinstance(t[3], collections.MutableMapping)
      if t[0] == 'MP':
        assert len(t) == 3
        assert (isinstance(t[1], int) or
                (isinstance(t[1], collections.MutableSequence) and
                 len(t[1]) == 2) and all(isinstance(x, int) for x in t[1]))
        assert (isinstance(t[2], int) or
                (isinstance(t[2], collections.MutableSequence) and
                 len(t[2]) == 2) and all(isinstance(x, int) for x in t[2]))
      if t[0] in ('F', 'GA'):
        assert len(t) == 1
      if t[0] in ('D', 'O'):
        assert len(t) == 2
        assert isinstance(t[1], int)
      if t[0] == 'DO':
        assert len(t) == 2
        assert isinstance(t[1], float) and 0 < t[1] and t[1] < 1
