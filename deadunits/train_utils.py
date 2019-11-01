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

"""This file has training utils for classification and pruning tasks.

  - `cross_entropy_loss`: Calculating loss and accuracy. Called by many
    functions, including but not limited to `probe_pruning`, `retrain_model`,
    `get_pruning_measurements`. It can split batches into chunks to prevent
    Out-of-Memory errors.
  - `pruning_schedule`: Returns a pruning schedule.
  - `log_loss_acc`: Calculates and logs loss and accuracy on two splits.
  - `log_sparsity`: Calculates and logs sparsity of layers of the given network.
  - `get_optimizer`: Returns gradient descent optimizer with correct
    learning according to the schedule given.
  - `retrain_model`:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

from deadunits import layers
import gin
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import summary as contrib_summary


def cross_entropy_loss(model,
                       dataset,
                       calculate_accuracy=False,
                       training=False,
                       compute_mean_replacement_saliency=False,
                       compute_removal_saliency=False,
                       is_abs=True,
                       aggregate_values=False,
                       run_gradient=False):
  """Cross Entropy Classification Loss.

  If Dataset is provided the result calculated recursively.
  Args:
    model: GenericConvnet.
    dataset: tuple or tf.data.Dataset, if tuple, it is (x, y) input, output
      pair. If tf.data.Dataset, each iterator value should be an x, y pair.
    calculate_accuracy: bool, if True accuracy is calculated and returned.
    training: flag to propagate to some layers(e.g. batch norm)
    compute_mean_replacement_saliency: flag to pass to the `model.__call__`.
    compute_removal_saliency: flag to pass to the `model.__call__`.
    is_abs: flag to pass to the `model.__call__`.
    aggregate_values: Passed to the GenericConvnet, such that it can be passed
      to the Taylor Scorer Layer. Set this true if you are aggregating saved
      values.
    run_gradient: bool, if True runs the gradient with respect to the given
      input. This might be useful for side effects like score calculation.

  Returns:
    loss: Tensor, scalar.
    acc: Tensor or None, A scalar Tensor if `calculate_accuracy` is True.
    total_sample: int, number of samples used during calculations
  """

  def calculate_loss_and_accuracy(x, y):
    """To reduce the code size we define this common call as a function.

    Note that the named_args are binded to the original ones during creation.
    Args:
      x: Tensor, input.
      y: Tensor, output.

    Returns:
      loss: avg loss.
      acc: acc.
    """
    logits = model(
        x,
        training=training,
        compute_mean_replacement_saliency=compute_mean_replacement_saliency,
        compute_removal_saliency=compute_removal_saliency,
        is_abs=is_abs,
        aggregate_values=aggregate_values)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    if calculate_accuracy:
      # Calculate accuracy.
      predictions = tf.cast(tf.argmax(logits, 1), y.dtype)
      acc = contrib_metrics.accuracy(tf.squeeze(y), predictions)
    else:
      acc = None
    return loss, acc

  if not isinstance(dataset, tf.data.Dataset):
    # Then it should be an x,y tuple. If not, the following might throw error.
    x, y = dataset
    total_samples = int(x.shape[0])
    if run_gradient:
      with tf.GradientTape() as tape:
        loss, acc = calculate_loss_and_accuracy(x, y)
      tape.gradient(loss, x)
    else:
      loss, acc = calculate_loss_and_accuracy(x, y)
  else:
    # We are dealing with a tf.data.Dataset, so lets iterate.
    # Reset TaylorScorer values if we are going to aggregate over Dataset:
    if aggregate_values:
      for l_name in model.get_layer_keys(layers.TaylorScorer):
        getattr(model, l_name).reset_saved_values()
    sum_loss, sum_acc = 0.0, 0.0
    total_samples = 0
    for x, y in dataset:
      c_size = int(x.shape[0])
      if run_gradient:
        with tf.GradientTape() as tape:
          c_loss, c_acc = calculate_loss_and_accuracy(x, y)
        tape.gradient(c_loss, x)
      else:
        c_loss, c_acc = calculate_loss_and_accuracy(x, y)
      # Running Mean
      sum_loss = sum_loss + c_loss * c_size
      if calculate_accuracy:
        sum_acc = sum_acc + c_acc * c_size
      total_samples += c_size
    loss = sum_loss / total_samples
    if calculate_accuracy:
      acc = sum_acc / total_samples
    else:
      acc = None
  return loss, acc, total_samples


@gin.configurable
def pruning_schedule(start_iteration=gin.REQUIRED,
                     target_fractions=gin.REQUIRED,
                     n_finetune=gin.REQUIRED):
  """Returns a schedule as a dictionary of <iteration, (layer_name, fraction)>.

  Schedule is a dictionary of (iteration, pruning_task) pairs. Each pruning
  task tell us how much sparsity a given layer have after pruning.

  `target_fractions` specify which layers will be pruned. For every layer
  there are `target_sparsity` and `increment` information provided.
  Each layer is pruned iteratively with given increments in round robin fashion
  (all layers are pruned once before we start the second round) until each of
  them reach to their targets. Therefore, the final pruning task for each layer
  is (layer_name, target_sparsity).

  For example:
    start_iteration=5
    target_fractions=[('layer_1', 0.3, 0.1), ('layer_2', 0.5, 0.3)]
    n_finetune=10
  Should return:
    {5: [('layer_1', 0.1)],
     15: [('layer_2', 0.3)],
     25: [('layer_1', 0.2)],
     35: [('layer_2', 0.5)],
     45: [('layer_1', 0.3)]}
  If n_finetune==0:
    {5: [('layer_1', 0.1), ('layer_2', 0.3), ('layer_1', 0.2),
    ('layer_2', 0.5), ('layer_1', 0.3)]}
  Args:
    start_iteration: int, First iteration to start with.
    target_fractions: list, of (layer_name, target_sparsity, fraction increment)
      tuples. Schedule is created in the same order as the tuples given.
    n_finetune: int, iterations between successive pruning steps.

  Returns:
    dict<(k,v)>, where the k is the iteration number (int) and v is the
      list of pruning tasks, where a task is a tuple of (layer_name, fraction).
  """
  if not target_fractions: return {}
  schedule = collections.defaultdict(list)
  c_iter = start_iteration
  layer_names = [t[0] for t in target_fractions]
  layer_constants = {layer_name: (target, increment)
                     for layer_name, target, increment in target_fractions}
  remaining_layers = collections.deque(
      [(layer_name, 0.0) for layer_name in layer_names])
  # Add pruning tasks `(layer_name, c_fraction)` to the schedule(list) in
  # round robin fashion until the target sparsities for each layer is reached.
  while remaining_layers:
    layer_name, previous_fraction = remaining_layers.popleft()
    target_sparsity, f_increment = layer_constants[layer_name]
    c_fraction = min(previous_fraction + f_increment, target_sparsity)
    schedule[c_iter].append((layer_name, c_fraction))
    c_iter += n_finetune
    if c_fraction < target_sparsity:
      remaining_layers.append((layer_name, c_fraction))
  return schedule


def retrain_model(model, total_iter, dataset_train, optimizer,
                  global_step=None):
  """Trains the model with data and optimizer given for `total_iter` steps.

  Args:
    model: tf.keras.Model, model to be trained.
    total_iter: int, number of updates to perform.
    dataset_train:  tf.data.Dataset, with at least N <x,y> samples where
      N>=`total_iter`.
    optimizer: tf.train.Optimizer.
    global_step: tf.Variable, variable to increment at each step.
  """
  for (i, (x, y)) in enumerate(dataset_train):
    if i == total_iter:
      break
    with tf.GradientTape() as tape:
      loss_train, _, _ = cross_entropy_loss(model, (x, y), training=True)
    grads = tape.gradient(loss_train, model.variables)
    # Updating the model.
    optimizer.apply_gradients(
        list(zip(grads, model.variables)), global_step=global_step)


@gin.configurable('get_optimizer', blacklist=['epoch'])
def get_optimizer(epoch,
                  lr=0.01,
                  schedule=()):
  """Given the epoch number returns optimizer with the scaled learning rate.

  Ex. `schedule=((20, 0.1), (40, 0.01), (60, 0.001))` drop of learning rate
    starting with 20th epoch.
  Args:
    epoch: int, current epoch
    lr: float, initial/base learning rate.
    schedule: list of tuples, each element of the list is (epoch, multiplier).

  Returns:
    tf.train.GradientDescentOptimizer, with scaled learning rate.
  """
  prev_f = 1.0
  for e, f in schedule:
    if epoch >= e:
      prev_f = f
    else:
      break
  current_lr = lr * prev_f
  return tf.train.GradientDescentOptimizer(learning_rate=current_lr)


def log_loss_acc(model, subset_val, subset_test):
  """Evaluates the model on two subsets of data and logs results.

  Args:
    model: tf.keras.Model, model to be evaluated.
    subset_val: tf.data.Dataset or tuple, validation subset. If `tuple` it is
      a mini-batch in the form (input, target).
    subset_test: tf.data.Dataset or tuple, test subset. If `tuple` it is
      a mini-batch in the form (input, target).
  Returns:
   tuple: evaluation results.
  """
  test_loss, test_acc, n_samples = cross_entropy_loss(
      model, subset_test, calculate_accuracy=True)
  contrib_summary.scalar('test_loss', test_loss)
  contrib_summary.scalar('test_acc', test_acc)
  tf.logging.info('test_loss:%.4f, test_acc:%.4f, '
                  'n_samples:%d', test_loss, test_acc, n_samples)
  val_loss, val_acc, n_samples = cross_entropy_loss(
      model, subset_val, calculate_accuracy=True)
  contrib_summary.scalar('val_loss', val_loss)
  contrib_summary.scalar('val_acc', val_acc)
  tf.logging.info('val_loss:%.4f, val_acc:%.4f, '
                  'n_samples:%d', val_loss, val_acc, n_samples)
  return val_loss, val_acc, test_loss, test_acc


def log_sparsity(model):
  """Logs the sparsity of each layer.

  Args:
    model: deadunits.generic_convnet.GenericConvnet
  """
  for l_name in model.forward_chain:
    l = getattr(model, l_name)
    # We skip BatchNormalization, Dropout, TaylorScorer, etc...
    if isinstance(l, layers.MaskedLayer):
      # If there is no mask it is a vector with `ones`.
      if not hasattr(l, 'mask_bias'):
        b_mask = tf.ones_like(l.layer.weights[1])
      else:
        b_mask = l.mask_bias
      img = tf.expand_dims(tf.expand_dims(tf.expand_dims(b_mask, 0), 0), -1)
      contrib_summary.image(l_name + '_mask_bias', img)
      contrib_summary.scalar(l_name + '_sparsity', l.get_sparsity())
