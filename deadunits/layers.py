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
"""Has some auxiliary layer definitions needed for pruning and mean replacement.

This file implements two layers:
  - MeanReplacer
  - Masked Layer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range
import tensorflow as tf

tfe = tf.contrib.eager


class MaskedLayer(tf.keras.layers.Wrapper):
  """This layer wraps keras.layers for applying binary masks to parameters.

  Example Usage:
    `ml = MaskedLayer(tf.keras.layers.Dense(32))`

  To ensure masks are saved properly, initiate tfe.Checkpoint object after
    masks are generated.
  This layer can be used for two tasks.
  1. Pruning
    One can prune a layer by setting the binary mask.
    `ml.set_mask(binary_weight_mask)`

  2. Dynamic_training: using a part of the layer. controlled dropout
     TODO
     This part is NOT yet needed and therefore not supported.
  """

  def __init__(self, layer, mask_initializer=tf.initializers.ones, **kwargs):
    """Initilizes the MaskedLayer.

    MaskedLayer assumes that the `tf.keras.layers.Layer` that it is wrapping has
    their weights at `layer.weights[0]` and bias at `layer.weights[1]`. This
    is the current API for `tf.keras.layers.Conv2D` or `tf.keras.layers.Dense`.
    Args:
      layer: tf.keras.layers.Layer that is going to be wrapped.
      mask_initializer: func, used to initialize the mask.
      **kwargs: any remaining named arguments passed to the super(MaskedLayer).

    Raises:
      ValueError: if the `layer` passed is not a keras layer.
    """
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError('Please initialize `MaskedLayer` layer with a '
                       '`Layer` instance. You passed: %s' % layer)
    super(MaskedLayer, self).__init__(layer, **kwargs)
    self.enabled = True
    self.mask_initializer = mask_initializer

  def build(self, input_shape):
    tf.logging.debug('MaskedLayer generated with shape: %s' % input_shape)
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = True
    # This is the case for now, we only support layers with weights+bias
    assert len(self.layer.weights) == 2
    self.mask_weight = self.add_variable('mask_weight',
                                         shape=self.layer.weights[0].shape,
                                         initializer=self.mask_initializer,
                                         trainable=False,
                                         dtype=self.layer.weights[0].dtype)
    self.mask_bias = self.add_variable('mask_bias',
                                       shape=self.layer.weights[1].shape,
                                       initializer=self.mask_initializer,
                                       trainable=False,
                                       dtype=self.layer.weights[1].dtype)
    super(MaskedLayer, self).build(input_shape)
    self.built = True

  def call(self, inputs, **kwargs):
    # We need to apply the mask to ensure the masked parameters are zero.
    # This is due to the fact that the gradient update might change the masked
    # parameters.

    # Note: Another way to implement this is to mask gradient. This would remove
    # the need for setting masked parameters zero before each forward pass.
    self.apply_masks()
    return self.layer(inputs, **kwargs)

  def set_mask(self, mask, is_bias=False):
    """Verifies the shape of the mask and sets it.

    Args:
      mask: tf.Tensor, same size as the weights (`self.layer.weights[0]`) or
        bias (`self.layer.weights[1]`) if `is_bias=True`.
      is_bias: sets the mask for the bias (layer.weights[1]) parameters if True.
    """
    # One shouldn't set the mask before building the layer /init.
    assert self.built
    current_mask = self.mask_bias if is_bias else self.mask_weight
    assert isinstance(mask, tf.Tensor)
    assert current_mask.shape == mask.shape
    mask = tf.cast(mask, current_mask.dtype)
    current_mask.assign(mask)

  def apply_masks(self):
    assert self.built
    if self.enabled:
      self.layer.weights[0].assign(self.layer.weights[0] * self.mask_weight)
      self.layer.weights[1].assign(self.layer.weights[1] * self.mask_bias)

  # Needs to be implemented (from tf.keras.layers.Layer)
  def compute_output_shape(self, input_shape):
    return input_shape

  def __repr__(self):
    return 'MaskedLayer object, name=%s' % self.name

  # This is needed to be able to mock nicely.
  def get_layer(self):
    return self.layer

  def get_sparsity(self, weight_only=True):
    # Returns the sparsity of the layer counting the 0's in the mask.
    total_param = tf.size(self.layer.weights[0], out_type=tf.int32)
    def get_sparse_weight_count():
      return total_param - tf.count_nonzero(self.mask_weight, dtype=tf.int32)
    pruned_param = 0 if self.mask_weight is None else get_sparse_weight_count()

    if not weight_only:
      total_bias = tf.size(self.layer.weights[1], out_type=tf.int32)
      def get_sparse_bias_count():
        return total_bias - tf.count_nonzero(self.mask_bias, dtype=tf.int32)
      pruned_bias = 0 if self.mask_bias is None else get_sparse_bias_count()
      total_param += total_bias
      pruned_param += pruned_bias

    sparsity = float(pruned_param) / float(total_param)
    return sparsity


class TaylorScorer(tf.keras.layers.Layer):
  """This layer uses forward activations to calculate two different scores.

  `compute_mean_replacement_saliency` and `compute_removal_saliency`
  has defaults and overwritten with `call` arguments.

  There are two main scores you can calculate.
    - Removal Saliency (RS): activated with `compute_removal_saliency` and if
    activated, first order approximation of the change in the loss calculated
    when the unit is replaced with zeros.
    - Mean Replacement Saliency (MRS): activated with
    `compute_mean_replacement_saliency` and if activated, first order
    approximation of the change in the loss calculated when the unit is replaced
    with its mean value calculated over the batch.

  References for MRS:
      - [Detecting Dead Weights and Units in Neural Networks]
        (https://arxiv.org/abs/1806.06068)
  """
  _saved_values_set = ['mean', 'l2norm', 'mrs', 'rs']

  def __init__(self,
               name=None,
               compute_removal_saliency=False,
               compute_mean_replacement_saliency=False,
               is_abs=True,
               save_l2norm=False,
               **kwargs):
    """Initilizes the MeanReplacer.

    Args:
      name: str, layer name that is passed to the super-class of TaylorScorer
      compute_removal_saliency: bool, activates Removal Saliency(RS)
        calculation.
      compute_mean_replacement_saliency: bool, if True, activates Mean
        Replacement Saliency (MRS) calculation.
      is_abs: bool, if True the first order approximation is aggregated through
        l1 norm. Otherwise reduce_mean is going to be used. Absolute value
        around delta_loss means we want to penalize negative changes as much as
        the positive changes.
      save_l2norm: bool, if True the normalized l2_norm calculated during
        forward pass.
      **kwargs: Any other named argument is passed to the super-class.
    Input shape: Arbitrary. The last dimension is assumed to have the output of
      separate units. In other words, an input of shape N * M * K * C would have
      an output of C units. We would calculate C different MRS scores or replace
      the M[:,:,:,i] with the corresponding mean.
    Output shape: Same shape as input.
    """
    super(TaylorScorer, self).__init__(name=name,
                                       trainable=False,
                                       **kwargs)
    self.compute_removal_saliency = compute_removal_saliency
    self.compute_mean_replacement_saliency = compute_mean_replacement_saliency
    self.is_abs = is_abs
    self.save_l2norm = save_l2norm
    self._mrs = None
    self._rs = None
    self._mean = None
    # TODO make this optional. This is used to save activations
    self._l2norm = None

  def set_or_aggregate(self, attr_name, val, n_elements):
    """Given an attr_name sets it to val if it is None or aggregates.

    Args:
      attr_name: str, must be one of 'mean', 'mrs', 'rs' or 'l2norm'.
      val: Tensor, to be set or aggregate. It should be normalized value by
        the n_elements.
      n_elements: int, number of elements in current batch.

    Raises:
      AssertionError: if attr_name is not valid.
    """
    assert attr_name in TaylorScorer._saved_values_set
    attr_name = '_' + attr_name
    prev_val = getattr(self, attr_name)
    if prev_val is None:
      setattr(self, attr_name, (val, n_elements))
    else:
      # We do running mean by keeping the (current_mean, total_n_elements)
      prev_val, prev_n_elements = prev_val
      c_n_elements = prev_n_elements + n_elements
      c_val = (prev_val * prev_n_elements + val * n_elements) / c_n_elements
      setattr(self, attr_name, (c_val, c_n_elements))

  def build(self, _):
    # We need to save the change in the activation (`c_delta`) in forward pass
    # and multiply that with output gradient to get the first order
    # approximation of mean replacement: MRS.
    @tf.custom_gradient
    def taylor_calc(x, compute_mean_replacement_saliency,
                    compute_removal_saliency, is_abs, save_l2norm):
      """Identity function calculating mrs, rs or both as side effects.

      `compute_mean_replacement_saliency` and `compute_removal_saliency`
      accepted as arguments to ensure consistency between
      forward and backward pass. Reading from
      `self.compute_mean_replacement_saliency` may cause inconsistencies, if
      those fields are updated between forward and backward call.
      Args:
        x: input tensor of any_shape(>=1d)
        compute_mean_replacement_saliency: Whether to calculate MRS during
          gradient calculation.
        compute_removal_saliency: Whether to calculate RS during gradient
          calculation.
        is_abs: bool, if True change in the loss penalized in both direction.
        save_l2norm: bool, if True saves the average squared activations.
      Returns:
        the output, gradient
      """
      self.xshape = x.shape
      n_dims = len(x.shape)
      n_elements = int(x.shape[0])
      c_mean = tf.reduce_mean(x, axis=list(range(n_dims - 1)))
      self.set_or_aggregate('mean', c_mean, n_elements)
      if save_l2norm:
        reshaped_inp = tf.reshape(x, [-1, x.shape[-1]])
        l2norm = tf.reduce_sum(tf.square(reshaped_inp),
                               axis=0) / reshaped_inp.shape[0].value
        self.set_or_aggregate('l2norm', l2norm, n_elements)

      def grad(dy):
        """Implements the calculation of two different saliencies.

        Mean Replacement Saliency (MRS)
        Removal Saliency (RS)

        This function is an identity gradient function. If
        `compute_mean_replacement_saliency`
        is True, it calculates the MRS score and stores it at `self._mrs`. If
        `compute_removal_saliency` is True, it calculates/aggregates the RS
        score and stores it at `self._rs`.

        Args:
          dy: Output gradient.

        Returns:
          dy: input itself.
        """
        if compute_mean_replacement_saliency:
          # Using c_mean, not the saved value.
          brodcasted_mean = tf.broadcast_to(c_mean, x.shape)
          c_delta = brodcasted_mean - x
          c_mrs = tf.multiply(c_delta, dy)

          # Reduce the tensor through sum if it has more than 2 dimensions (e.g.
          # output of Conv2D layer).
          # For example: if 4D (output of Conv2D), it sums over axis=[1, 2].
          for _ in range(n_dims - 2):
            c_mrs = tf.reduce_mean(c_mrs, axis=1)
          # Following is for approximating abs change.
          if is_abs:
            c_mrs = tf.abs(c_mrs)
          c_mrs = tf.reduce_mean(c_mrs, axis=0)
          self.set_or_aggregate('mrs', c_mrs, n_elements)
        if compute_removal_saliency:
          c_rs = tf.multiply(-x, dy)
          # Reduce the tensor through sum if it has more than 2 dimensions (e.g.
          # output of Conv2D layer).
          # For example: if 4D (output of Conv2D), it sums over axis=[1, 2].
          for _ in range(n_dims - 2):
            c_rs = tf.reduce_mean(c_rs, axis=1)
          # Following is for approximating abs change.
          if is_abs:
            c_rs = tf.abs(c_rs)
          c_rs = tf.reduce_mean(c_rs, axis=0)
          self.set_or_aggregate('rs', c_rs, n_elements)
        return dy

      return tf.identity(x), grad

    self.custom_forward_fun = taylor_calc

  def call(self,
           inputs,
           compute_removal_saliency=None,
           compute_mean_replacement_saliency=None,
           is_abs=None,
           save_l2norm=None,
           aggregate_values=False):
    """Forward call for the layer implementing custom behaviours.

    There are two modes, that cannot be on at the same time.
      - `self.compute_mean_replacement_saliency`: if True, MRS is calculated
      during backprop.
      - `self.compute_removal_saliency`: if True, RS is calculated during
      backprop.
     These two modes can be overwritten by the named_arguments.

    Args:
      inputs: tf.Tensor with at least one element.
      compute_removal_saliency: overwrites the `self.compute_removal_saliency`
        event for current call.
      compute_mean_replacement_saliency: overwrites the
        `self.compute_mean_replacement_saliency` event for current call.
      is_abs: bool, if True change in the loss penalized in both direction.
      save_l2norm: bool, if True overwrites the flag provided at initiliazation
        for this specific call.
      aggregate_values: bool, if True it aggregate the previous values with the
        new ones.

    Returns:
      output: Same as `inputs`.
    """
    if compute_removal_saliency is None:
      compute_removal_saliency = self.compute_removal_saliency
    if compute_mean_replacement_saliency is None:
      compute_mean_replacement_saliency = self.compute_mean_replacement_saliency
    if is_abs is None:
      is_abs = self.is_abs
    if save_l2norm is None:
      save_l2norm = self.save_l2norm

    if not aggregate_values:
      # Remove the previous mrs/rs if it exists.
      self.reset_saved_values()
    output = self.custom_forward_fun(
        inputs,
        compute_mean_replacement_saliency=compute_mean_replacement_saliency,
        compute_removal_saliency=compute_removal_saliency,
        is_abs=is_abs,
        save_l2norm=save_l2norm)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def reset_saved_values(self):
    self._mean = None
    self._l2norm = None
    self._mrs = None
    self._rs = None

  def get_saved_values(self,
                       attr_name,
                       broadcast_to_input_shape=False,
                       unit_mask=None):
    """Returns the saved values of the most recent forward pass.

    All of 'mean', 'l2norm', 'mrs' and 'rs' have the same shape and here we
    define common getter operation for them.
    Args:
      attr_name: str, 'mean', 'l2norm', 'mrs' or 'rs'.
      broadcast_to_input_shape: bool, if True the values are broadcast to the
        input shape.
      unit_mask: Tensor, same shape as `self._<attr_name>` and it is multiplied
        with the saved tensor before broadcast operation.

    Returns:
      Tensor or None: None if there is no saved value exists.
    Raises:
      ValueError: when the `attr_name` is not valid.
    """
    if attr_name not in TaylorScorer._saved_values_set:
      raise ValueError('attr_name: %s is not valid. ' % attr_name)
    attr_name = '_'+ attr_name
    if getattr(self, attr_name) is None:
      return None

    val, _ = getattr(self, attr_name)
    # TODO maybe get rid of this part. It doesn't belong here.
    if unit_mask is None:
      possibly_masked_mean = val
    else:
      tf.assert_equal(val.shape, unit_mask.shape)
      possibly_masked_mean = tf.multiply(val, unit_mask)
    if broadcast_to_input_shape:
      return tf.broadcast_to(possibly_masked_mean, self.xshape)
    else:
      return possibly_masked_mean

  def get_config(self):
    config = {
        'compute_removal_saliency':
            self.compute_removal_saliency,
        'compute_mean_replacement_saliency':
            self.compute_mean_replacement_saliency,
        'is_abs': self.is_abs,
        'save_l2norm': self.save_l2norm
    }
    # Only add TensorFlow-specific parameters if they are set, so as to preserve
    # model compatibility with external Keras.
    base_config = super(TaylorScorer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def __repr__(self):
    return ('TaylorScorer object, name=%s, '
            'compute_removal_saliency=%s, compute_mean_replacement_saliency=%s,'
            ' save_l2norm:%s, is_abs:%s'
           ) % (self.name, self.compute_removal_saliency,
                self.compute_mean_replacement_saliency, self.save_l2norm,
                self.is_abs)


class MeanReplacer(tf.keras.layers.Layer):
  """This layer replaces some of the units asked with their mean.

  `is_replacing` has default and can be overwritten with arguments during
    `call`.
  """

  def __init__(self, name=None, is_replacing=False, **kwargs):
    """Initilizes the MeanReplacer.

    Args:
      name: str, layer name that is passed to the super-class of MeanReplacer
      is_replacing: bool, if True replaces the output of each unit (last
        dimension) with the mean values calculated over the batch and filters
        (everything but the last dim). If false, no side effects.
      **kwargs: Any other named argument is passed to the super-class.
    Input shape: Arbitrary. The last dimension is assumed to have the output of
      separate units. In other words, an input of shape N * M * K * C would have
      an output of C units. We would calculate C different MRS scores or replace
      the M[:,:,:,i] with the corresponding mean.
    Output shape: Same shape as input.
    """
    super(MeanReplacer, self).__init__(name=name,
                                       trainable=False,
                                       **kwargs)
    self.is_replacing = is_replacing
    self._active_units = []
    self.mrs = None

  def set_active_units(self, active_units):
    """Checks validity of active units and sets it.

    The layer should be built before calling this function.
    Args:
      active_units: list, of int. Every int should be in the range
        0<i<input_shape[-1]. Note that input_shape[-1] is equal to number of
        units. Duplicates are removed.
    """
    assert self.built
    # active_units returns True if not empty
    assert (isinstance(active_units, list) and active_units)
    for i in active_units:
      assert (isinstance(i, int) and 0 <= i and i < self.n_units)
    # Making sure we have a copy of the list and unique indices.
    self._active_units = sorted(list(set(active_units)))

  def build(self, input_shape):
    # We need `n_units` for `self.set_active_units` function.
    self.n_units = input_shape[-1]

  def call(self, inputs, is_replacing=None):
    """Forward call for the mean replacing layer.

      `self.is_replacing`: if True, the output is calculated through taking
      the mean over the batch for each unit. Units for which we perform this
      calculation is declared with self._active_units and set with
      `set_active_units` function.
       Replacing mode can be overwritten by the is_replacing named argument.

    Args:
      inputs: tf.Tensor with at least one element.
      is_replacing: overwrites the `self.is_replacing` event for current call.

    Returns:
      output: Same as `inputs` if self.is_replacing` is False. If it is True,
        all input channels except last dimension replaced with the mean for the
        `active_units`.

    Raises:
      AssertionError: When the `input` and `n_units` doesn't match. We expect to
        process same shape `input`. If another `input` shape is required, create
        a new MeanReplacer.
    """
    if is_replacing is None:
      is_replacing = self.is_replacing
    output = tfe.Variable(inputs)
    if is_replacing:
      # If active units empty give a warning and return the input.
      if not self._active_units:
        tf.logging.warning('From %s: is_replacing=True, but there are no active'
                           'units.' % self)
        return inputs
      assert inputs.shape[-1] == self.n_units

      n_dims = len(inputs.shape)
      c_mean = tf.reduce_mean(inputs, axis=list(range(n_dims - 1)))
      for i in self._active_units:
        tf.assign(output[Ellipsis, i], tf.broadcast_to(c_mean[i], output.shape[:-1]))
    return output.read_value()

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'is_replacing': self.is_replacing,
        '_active_units': self._active_units,
    }
    # Only add TensorFlow-specific parameters if they are set, so as to preserve
    # model compatibility with external Keras.
    base_config = super(MeanReplacer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def __repr__(self):
    return ('MeanReplacer object, name=%s, '
            'is_replacing=%s, active_units=%s') % (self.name, self.is_replacing,
                                                   self._active_units)
