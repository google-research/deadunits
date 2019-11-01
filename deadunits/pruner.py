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
"""This module has UnitPruner object and some pruning related functions.

- `UnitPruner`: gets a model and a dataset and has two functions.
  1. `prune_layer`: Prunes a single layer.
  2. `prune_one_unit`: Prunes a single unit using global ordering among layers.

Other functions:

- `probe_pruning`: Performs multiple pruning experiments efficiently cloning the
     model each time.
- `prune_model_with_scores`: utility function used by the UnitPruner.
     There are six scoring functions used: 'mrs', 'abs_mrs', 'rs',
     'abs_rs', 'rand' and 'norm'.
- `process_layers2prune`: Binds the string arguments to the layer names and
     validates the layer names.
- `get_pruning_measurements`: Using data on the model calculates various
     scoring functions for units.

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from deadunits import layers
from deadunits import train_utils
from deadunits import unitscorers
from deadunits import utils
import gin
import six
import tensorflow as tf
from tensorflow.contrib import summary as contrib_summary

ALL_SCORING_FUNCTIONS = ['norm', 'abs_mrs', 'abs_rs', 'mrs', 'rs', 'rand']


@gin.configurable(blacklist=['model', 'subset_val'])
class UnitPruner(object):
  """Class for supporting various pruning tasks."""

  def __init__(self, model, subset_val, pruning_method, is_bp):
    """Prunes a copy of the network and calculates change in the loss.

    By default calculates mrs, rs and mean values.
    Args:
      model: tf.keras.Model.
      subset_val: tf.data.Dataset, used for loss calculation.
      pruning_method: str, from ['norm', 'mrs', 'rs', 'rand', 'abs_mrs', 'rs'].
        'norm': unitscorers.norm_score
        '{abs_}mrs': `compute_mean_replacement_saliency=True` for `TaylorScorer`
          layer. if {abs_} prefix exists absolute value used before aggregation.
          i.e. is_abs=True.
        '{abs_}rs': `compute_removal_saliency=True` for `TaylorScorer` layer. if
          {abs_} prefix exists absolute value used before aggregation. i.e.
          is_abs=True.
        'rand': unitscorers.random_score is_bp; bool, if True, mean value of the
          units are propagated to the next layer prior to the pruning. `bp` for
          bias_propagation.
     is_bp: bool, if True Mean Replacement Pruning is used and bias propagation
       is made.
    """
    self.model = model
    self.subset_val = subset_val
    self.pruning_method = pruning_method
    self.is_bp = is_bp

  @gin.configurable(blacklist=['layer_name', 'pruning_factor', 'pruning_count'])
  def prune_layer(self, layer_name, pruning_factor=0.1, pruning_count=None,
                  pruning_method=None, is_bp=None):
    """Prunes a single layer using the given scoring function.

    Args:
      layer_name: str, layer name to prune.
      pruning_factor: float, 0 < pruning_factor < 1.
      pruning_count: int, if not None, sets the pruning_factor to None. This is
        because you can either prune a fraction or a number of units.
        pruning_count is used to determine how many units to prune per layer.
      pruning_method: str, from ['norm', 'mrs', 'rs', 'rand', 'abs_mrs', 'rs'].
        If given, overwrites the default value.
     is_bp: bool, if True Mean Replacement Pruning is used and bias propagation
       is made. If given, overwrites the default value.

    Raises:
      ValueError: if the arguments provided doesn't match specs.
    """
    pruning_method = pruning_method if pruning_method else self.pruning_method
    is_bp = is_bp if is_bp else self.is_bp
    if pruning_method not in ALL_SCORING_FUNCTIONS:
      raise ValueError(
          '%s is not one of %s' % (pruning_method, ALL_SCORING_FUNCTIONS))
    # Need to wrap up the layer in a list for
    # `pruner.prune_model_with_scores()` call.
    layers2prune = [layer_name]
    # If pruning_count exists invalidate the pruning_factor.
    if pruning_count is not None:
      if not isinstance(pruning_count, int):
        raise ValueError('pruning_count: %s should be an int' % pruning_count)
      elif pruning_count < 1:
        raise ValueError(
            'pruning_count: %d should be greater than 1.' % pruning_count)
      pruning_factor = None
    # Validate pruning_factor.
    elif pruning_factor == 0:
      return
    elif pruning_factor <= 0 or pruning_factor >= 1:
      raise ValueError(
          'pruning_factor: %s should be in (0, 1)' % pruning_factor)

    tf.logging.info('Pruning layer `%s` with: %s, f:%.2f',
                    layer_name, pruning_method, pruning_factor)
    input_shapes = {layer_name: getattr(self.model, layer_name + '_ts').xshape}

    # Calculating the scoring function/mean value.
    is_abs = pruning_method.startswith('abs')
    is_mrs = pruning_method.endswith('mrs')
    is_rs = pruning_method.endswith('rs') and not is_mrs
    train_utils.cross_entropy_loss(
        self.model,
        self.subset_val,
        training=False,
        compute_mean_replacement_saliency=is_mrs,
        compute_removal_saliency=is_rs,
        is_abs=is_abs,
        aggregate_values=True,
        run_gradient=True)
    scores = {}
    mean_values = {}
    # `layer_ts` stands for TaylorScorer layer.
    layer_ts = getattr(self.model, layer_name + '_ts')
    masked_layer = getattr(self.model, layer_name)
    masked_layer.apply_masks()
    if pruning_method == 'rand':
      scores[layer_name] = unitscorers.random_score(
          masked_layer.get_layer().weights[0])
    elif pruning_method == 'norm':
      scores[layer_name] = unitscorers.norm_score(
          masked_layer.get_layer().weights[0])
    else:
      # mrs or rs.
      score_name = 'rs' if is_rs else 'mrs'
      scores[layer_name] = layer_ts.get_saved_values(score_name)
    mean_values[layer_name] = layer_ts.get_saved_values('mean')
    prune_model_with_scores(
        self.model, scores, is_bp, layers2prune, pruning_factor, pruning_count,
        mean_values, input_shapes)

  @gin.configurable(blacklist=['pruning_pool', 'baselines'])
  def prune_one_unit(self, pruning_pool, baselines=None, normalized_scores=True,
                     pruning_method=None, is_bp=None):
    """Picks a layer and prunes a single unit using the scoring function.

    Args:
      pruning_pool: list, of layers that are considered for pruning.
      baselines: dict, if exists, subtracts the given constant from the scores
        of individual layers. The keys should a subset of pruning_pool.
      normalized_scores: bool, if True the scores are normalized with l2 norm.
      pruning_method: str, from ['norm', 'mrs', 'rs', 'rand', 'abs_mrs', 'rs'].
        If given, overwrites the default value.
      is_bp: bool, if True Mean Replacement Pruning is used and bias propagation
        is made. If given, overwrites the default value.

    Raises:
      AssertionError: if the arguments provided doesn't match specs.
    """
    pruning_method = pruning_method if pruning_method else self.pruning_method
    is_bp = is_bp if is_bp else self.is_bp
    if pruning_method not in ALL_SCORING_FUNCTIONS:
      raise ValueError(
          '%s is not one of %s' % (pruning_method,
                                   ALL_SCORING_FUNCTIONS))
    if baselines is None:
      baselines = {}
    tf.logging.info('Prunning with: %s, is_bp: %s', pruning_method, is_bp)

    # Calculating the scoring function/mean value.
    is_abs = pruning_method.startswith('abs')
    is_mrs = pruning_method.endswith('mrs')
    is_rs = pruning_method.endswith('rs') and not is_mrs
    is_grad = is_mrs or is_rs
    train_utils.cross_entropy_loss(
        self.model,
        self.subset_val,
        training=False,
        compute_mean_replacement_saliency=is_mrs,
        compute_removal_saliency=is_rs,
        is_abs=is_abs,
        aggregate_values=True,
        run_gradient=is_grad)
    scores = {}
    mean_values = {}
    smallest_score = None
    smallest_l_name = None
    smallest_nprune = None

    for l_name in pruning_pool:
      l_ts = getattr(self.model, l_name + '_ts')
      l = getattr(self.model, l_name)
      mean_values[l_name] = l_ts.get_saved_values('mean')
      # Make sure the masks are applied after last gradient update. Note
      # that this is necessary for `norm` functions, since it doesn't call the
      # model and therefore the masks are not applied.
      l.apply_masks()
      if pruning_method == 'rand':
        scores[l_name] = unitscorers.random_score(l.get_layer().weights[0])
      elif pruning_method == 'norm':
        scores[l_name] = unitscorers.norm_score(l.get_layer().weights[0])
      else:
        # mrs or rs.
        score_name = 'rs' if is_rs else 'mrs'
        scores[l_name] = l_ts.get_saved_values(score_name)
      if normalized_scores:
        scores[l_name] /= tf.norm(scores[l_name])
      baseline_score = baselines.get(l_name, 0)
      if baseline_score != 0:
        # Regularizing the scores with c_flop weights.
        scores[l_name] -= baseline_score
      # If there is an existing mask we have to make sure pruned connections
      # are indicated. Let's set them to very small negative number (-1e10).
      # Note that the elements of `l.mask_bias` consist of zeros and ones only.
      if l.mask_bias is not None:
        # Setting the scores of the pruned units to zero.
        scores[l_name] = scores[l_name] * l.mask_bias
        # Setting the scores of the pruned units to -1e10.
        scores[l_name] += -1e10 * (1 - l.mask_bias)
        # Number of previously pruned units.
        n_pruned = tf.count_nonzero(l.mask_bias - 1).numpy()
        layer_smallest_score = tf.reduce_min(
            tf.boolean_mask(scores[l_name], l.mask_bias)).numpy()
        # Do not prune the last unit.
        if tf.equal(n_pruned + 1, tf.size(l.mask_bias)):
          continue
      else:
        n_pruned = 0
        layer_smallest_score = tf.reduce_min(scores[l_name]).numpy()

      tf.logging.info('Layer:%s, min:%f', l_name, layer_smallest_score)
      if smallest_score is None or (layer_smallest_score < smallest_score):
        smallest_score = layer_smallest_score
        smallest_l_name = l_name
        # We want to prune one more than before.
        smallest_nprune = n_pruned + 1
    tf.logging.info('UNIT_PRUNED, layer:%s, n_pruned:%d',
                    smallest_l_name, smallest_nprune)
    mean_values = {smallest_l_name: mean_values[smallest_l_name]}
    scores = {smallest_l_name: scores[smallest_l_name]}
    input_shapes = {
        smallest_l_name: getattr(self.model, smallest_l_name + '_ts').xshape}
    layers2prune = [smallest_l_name]
    prune_model_with_scores(self.model, scores, is_bp, layers2prune, None,
                            smallest_nprune, mean_values, input_shapes)


@gin.configurable('pruning_config', blacklist=['model', 'subset_val',
                                               'subset_val2', 'subset_test',
                                               'f_retrain'])
def probe_pruning(model,
                  subset_val,
                  subset_val2,
                  subset_test,
                  f_retrain,
                  baselines=(0.0, 0.0),
                  layers2prune='all',
                  n_retrain=0,
                  pruning_factor=0.1,
                  pruning_count=None,
                  pruning_methods=(('norm', True),)):
  """Prunes a copy of the network and calculates change in the loss.

  By default calculates mrs,rs and mean values.
  Args:
    model: tf.keras.Model
    subset_val: tf.data.Dataset, used for loss calculation.
    subset_val2: tf.data.Dataset, used for pruning scoring.
    subset_test: tf.data.Dataset, from test set.
    f_retrain: function, used for retraining with 2 arguments `copied_model` and
      `n_retrain`.
    baselines: tuple, <val_loss, test_loss> Baselines to subtract from loss,
    layers2prune: list or str, each elemenet `name` in the list should be a
      valid MasketLayer under model. model.name->MaskedLayer. One can also
      provide following tokens:
        `all`: searches model finds all MaskedLayers's and prunes them all.
        `firstconv`: prunes the first conv_layer in the `forward_chain`.
        `midconv`: prunes the `mid=n_conv//2` conv_layer in the `forward_chain`.
        `lastconv`: prunes the last conv_layer in the `forward_chain`.
        `firstdense`: prunes the first dense layer in the `forward_chain`.
    n_retrain: int, Number of retraining updates to perform after pruning.
      If n_retrain<=0, then nothing happens.
    pruning_factor: float, 0<pruning_factor<1
    pruning_count: int, if not None, sets the pruning_factor to None. This is
      because you can either prune a fraction or a number of units.
      pruning_count is used to determine how many units to prune per layer.
    pruning_methods: iterator of tuples, (scoring, is_bp) where `is_bp`
      is a boolean indicating the usage of Mean Replacement Pruning. Scoring is
      a string from ['norm', 'mrs', 'rs', 'rand', 'abs_mrs', 'rs'].
        'norm': unitscorers.norm_score
        '{abs_}mrs': `compute_mean_replacement_saliency=True` for `TaylorScorer`
          layer. if {abs_} prefix exists absolute value used before aggregation.
          i.e. is_abs=True.
        '{abs_}rs': `compute_removal_saliency=True` for `TaylorScorer` layer. if
          {abs_} prefix exists absolute value used before aggregation. i.e.
          is_abs=True.
        'rand': unitscorers.random_score is_bp; bool, if True, mean value of the
          units are propagated to the next layer prior to the pruning. `bp` for
          bias_propagation.

  Raises:
    AssertionError: if the arguments provided doesn't match specs.

  Returns:
    selected_units: dict, keys coming from `layers2prune` and each value is a
      tuple of (score, mask), where score is the pruning scores for each units
      and mask is the corresponding binary masks created for the given fraction.
    mean_values: dict, keys coming from `layers2prune` and each value is the
      mean activation under training batch.
  """
  # Check validity of `layers2prune` and process.
  layers2prune = process_layers2prune(layers2prune, model)
  # If pruning_count exists invalidate the pruning_factor
  if pruning_count is not None:
    assert isinstance(pruning_count, int) and pruning_count >= 1
    pruning_factor = None
  copied_model = model.clone()
  loss_val, loss_test = baselines
  selected_units = {}
  input_shapes = {
      l_name: getattr(model, l_name + '_ts').xshape for l_name in layers2prune
  }
  measurements = get_pruning_measurements(copied_model, subset_val2,
                                          layers2prune)
  (scores, mean_values, l2_norms) = measurements
  for scoring, is_bp in pruning_methods:
    assert (scoring in ['norm', 'abs_mrs', 'abs_rs', 'mrs', 'rs', 'rand'])
    copied_model = model.clone()
    scalar_summary_tag = 'pruning_penalty%s_%s' % ('_bp' if is_bp else '',
                                                   scoring)
    tf.logging.info('Pruning following layers: %s, '
                    'Using %s %s bias_prop.' % (layers2prune, scoring,
                                                'with' if is_bp else 'without'))
    selected_units_scoring = prune_model_with_scores(
        copied_model, scores[scoring], is_bp, layers2prune, pruning_factor,
        pruning_count, mean_values, input_shapes)
    # There's going to be two pass for a given `scoring` with and without `bp`.
    # We will record them only once, since they are equal.
    if scoring not in selected_units:
      selected_units[scoring] = selected_units_scoring
    if n_retrain > 0:
      f_retrain(copied_model, n_retrain)
    # Setting training=True since test uses running average and revives pruned,
    # units.
    loss_new, _, _ = train_utils.cross_entropy_loss(copied_model, subset_val,
                                                    training=True)
    contrib_summary.scalar(scalar_summary_tag, loss_new - loss_val)
    # Setting training=True, otherwise BatchNorm uses the accumulated mean and
    # std during forward propagation, which causes pruned units to generate
    # non-zero constants.
    loss_new, _, _ = train_utils.cross_entropy_loss(
        copied_model, subset_test, training=True)
    contrib_summary.scalar(scalar_summary_tag + '_test', loss_new - loss_test)
  return selected_units, mean_values, l2_norms


def prune_model_with_scores(model, scores, is_bp, layers2prune, pruning_factor,
                            pruning_count, mean_values, input_shapes):
  """Prunes the model with or without bias propagation.

  Args:
    model: tf.keras.Model, model to be pruned.
    scores: dict, with keys matching the elements of the `layers2prune`.
    is_bp: bool, if True, bias propagation is performed.
    layers2prune: list, list of `MaskedLayer` names to be pruned.
    pruning_factor: float or None, if float, 0<pruning_factor<1
    pruning_count: int or None, if int it needs to be smaller than the number
      of units in the smallest layer getting pruned.
    mean_values: dict, with keys matching the elements of the `layers2prune` and
      values with the mean activations.
    input_shapes: dict, with keys matching the elements of the `layers2prune`
      and values with the original activation shapes. Used to broadcast the mean
      values to the right shape and perform the bias propagation.
  Returns:
    dict, with dict.keys=layers2prune and dict.values = (scores, mask).
  """
  selected_units = {}
  for l_name in layers2prune:
    l = getattr(model, l_name)
    binary_unit_mask = utils.create_binary_mask_from_scores(
        scores[l_name], f=pruning_factor, n_zeros=pruning_count)
    if is_bp:
      mean_vals = mean_values[l_name]
      inp_shape = input_shapes[l_name]
      broadcast_mean_vals = utils.mask_and_broadcast(
          mean_vals, binary_unit_mask, inp_shape, invert_mask=True)
      model.propagate_bias(l_name + '_ts', broadcast_mean_vals)
    selected_units[l_name] = [scores[l_name], binary_unit_mask]
    # binary_unit_mask.shape : (M,)
    # binary_mask : (C, M); Dense layer / (H, W, C, M); Conv2D
    binary_mask = tf.broadcast_to(binary_unit_mask,
                                  l.get_layer().weights[0].shape)
    l.set_mask(binary_mask)
    l.set_mask(binary_unit_mask, is_bias=True)
  return selected_units


def process_layers2prune(layers2prune, model):
  """Processes and checks the validity of `layers2prune` argument.

  Args:
    layers2prune: list or str, each elemenet `name` in the list should be a
      valid MasketLayer under model. model.name->MaskedLayer. One can also
      provide following tokens:
        `all`: searches model finds all MaskedLayers's.
        `firstconv`: gets the first conv_layer in the `forward_chain`.
        `midconv`: gets the `mid=n_conv//2` conv_layer in the `forward_chain`.
        `lastconv`: gets the last conv_layer in the `forward_chain`.
        `firstdense`: gets the first dense layer in the `forward_chain`.
    model: tf.keras.Model, model to check whether the layers2prune are valid
     `MaskedLayer`s.
  Returns:
    list<str>, processed and verified layer names.
  """
  if isinstance(layers2prune, six.string_types):
    if layers2prune == 'all':
      layers2prune = model.get_layer_keys(layers.MaskedLayer)
    elif layers2prune == 'firstconv':
      # If there is a Conv layer in GenericConvnet, it should be `conv_1`.
      layers2prune = ['conv_1']
    elif layers2prune == 'midconv':
      mid_i = model.layer_name_counter['C'] // 2
      layers2prune = ['conv_%d' % mid_i]
    elif layers2prune == 'lastconv':
      last_i = (model.layer_name_counter['C'] - 1)
      layers2prune = ['conv_%d' % last_i]
    elif layers2prune == 'firstdense':
      layers2prune = ['dense_1']
    else:
      layers2prune = [layers2prune]
  for l_name in layers2prune:
    assert hasattr(model, l_name) and isinstance(
        getattr(model, l_name), layers.MaskedLayer)
  return layers2prune


def get_pruning_measurements(model, subset_val, layers2prune):
  """Returns 6 different pruning scores and some other measurements.

  Args:
    model: tf.keras.Model, model to be pruned.
    subset_val:  tf.data.Dataset, to calculate data-dependent scoring functions.
    layers2prune: list<str>, of layers which will be scored.
  Returns:
    dict, scores with 6 keys `mrs`, `abs_mrs`, `rs`, `abs_rs`, `norm`, `rand`.
      and score dictionary values. Each Score dictionary has scorings for each
      layer provided in `layers2prune`.
    dict, mean unit activations for each layer in layers2prune.
    dict, l2norms, average square activations for each unit.
      see `deadunits.layers.TaylorScorer` for further details.
  """
  # Run once and get mrs, rs and mean calculated
  train_utils.cross_entropy_loss(
      model,
      subset_val,
      training=False,
      compute_mean_replacement_saliency=True,
      compute_removal_saliency=True,
      is_abs=True,
      aggregate_values=True,
      run_gradient=True)
  scores = collections.defaultdict(dict)
  mean_values = {}
  l2_norms = {}
  for l_name in layers2prune:
    l_ts = getattr(model, l_name + '_ts')
    scores['abs_mrs'][l_name] = l_ts.get_saved_values('mrs')
    scores['abs_rs'][l_name] = l_ts.get_saved_values('rs')
    mean_values[l_name] = l_ts.get_saved_values('mean')
    l2_norms[l_name] = l_ts.get_saved_values('l2norm')

  # Run again to get without abs.
  train_utils.cross_entropy_loss(
      model,
      subset_val,
      training=False,
      compute_mean_replacement_saliency=True,
      compute_removal_saliency=True,
      is_abs=False,
      aggregate_values=True,
      run_gradient=True)

  for l_name in layers2prune:
    l_ts = getattr(model, l_name + '_ts')
    l = getattr(model, l_name)
    scores['mrs'][l_name] = l_ts.get_saved_values('mrs')
    scores['rs'][l_name] = l_ts.get_saved_values('rs')
    # The reson we calculate them here to reduce the amount of the code.
    # `weights[0]` is the weight, where `weights[1]` is the bias.
    scores['norm'][l_name] = unitscorers.norm_score(l.get_layer().weights[0])
    scores['rand'][l_name] = unitscorers.random_score(l.get_layer().weights[0])
  return (scores, mean_values, l2_norms)
