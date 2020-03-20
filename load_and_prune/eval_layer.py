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

r"""Loads model and evaluates the pruning performance of a single layer.

This script loads a model and prunes a single layer with different pruning
fractions. Performance after pruning without fine-tuning saved as pickle file.

"""

from __future__ import division
import json
import os
from absl import app
from absl import flags
from absl import logging

from deadunits import data
from deadunits import model_load
from deadunits import pruner
from deadunits import train_utils
from deadunits import utils
import gin
import numpy as np
import tensorflow.compat.v2 as tf
FLAGS = flags.FLAGS

flags.DEFINE_string('outdir', '/tmp/dead_units/test',
                    'Path to directory where to store summaries.')
flags.DEFINE_boolean('override_existing_dir', False, 'remove if exists')
flags.DEFINE_multi_string(
    'gin_config',
    'configs/eval_layer_default.gin',
    'List of paths to the config files.')
flags.DEFINE_multi_string('gin_binding', None,
                          'Newline separated list of Gin parameter bindings.')


# in prune_all_layers.
@gin.configurable
def prune_layer_and_eval(dataset_name='cifar10',
                         pruning_methods=(('rand', True),),
                         model_dir=gin.REQUIRED,
                         l_name=gin.REQUIRED,
                         n_exps=gin.REQUIRED,
                         val_size=gin.REQUIRED,
                         max_pruning_fraction=gin.REQUIRED):
  """Loads and prunes a model with various sparsity targets.

  This function assumes that the `seed_dir` exists
  Args:
    dataset_name: str, either 'cifar10' or 'imagenet'.
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
    model_dir: str, Path to the checkpoint directory.
    l_name: str, a valid layer name from the model loaded.
    n_exps: int, number of pruning experiments to be made. This number is used
      generate pruning counts for different experiments.
    val_size: int, size for the first dataset, passed to the `get_datasets`.
    max_pruning_fraction: float, max sparsity for pruning. Multiplying this
      number with the total number of units, we would get the upper limit for
      the pruning_count.

  Raises:
    AssertionError: when no checkpoint is found.
    ValueError: when the scoring function key is not valid.
    OSError: when there is no checkpoint found.
  """
  logging.info('Looking checkpoint at: %s', model_dir)
  latest_cpkt = tf.train.latest_checkpoint(model_dir)
  if not latest_cpkt:
    raise OSError('No checkpoint found in %s' % model_dir)
  logging.info('Using latest checkpoint at %s', latest_cpkt)
  model = model_load.get_model(load_path=latest_cpkt,
                               dataset_name=dataset_name)
  datasets = data.get_datasets(dataset_name=dataset_name,
                               val_size=val_size)
  _, _, subset_val, subset_test, subset_val2 = datasets
  input_shapes = {
      l_name: getattr(model, l_name + '_ts').xshape
  }
  layers2prune = [l_name]
  measurements = pruner.get_pruning_measurements(model, subset_val,
                                                 layers2prune)
  (all_scores, mean_values, _) = measurements
  for scoring, is_bp in pruning_methods:
    if scoring not in pruner.ALL_SCORING_FUNCTIONS:
      raise ValueError(
          '%s is not one of %s' % (scoring,
                                   pruner.ALL_SCORING_FUNCTIONS))
    scores = all_scores[scoring]
    d_path = os.path.join(FLAGS.outdir, '%d-%s-%s-%s.pickle' % (
        val_size, l_name, scoring, str(is_bp)))
    logging.info(d_path)
    if tf.gfile.Exists(d_path):
      logging.warning('File %s exists, skipping.', d_path)
    else:
      ls_train_loss = []
      ls_train_acc = []
      ls_test_loss = []
      ls_test_acc = []

      n_units = input_shapes[l_name][-1].value
      n_unit_pruned_max = n_units*max_pruning_fraction
      c_slice = np.linspace(0,
                            n_unit_pruned_max,
                            n_exps, dtype=np.int32)
      logging.info('Layer:%s, n_units:%d, c_slice:%s', l_name, n_units,
                   str(c_slice))
      for pruning_count in c_slice:
        # Cast from np.int32 to int.
        pruning_count = int(pruning_count)
        copied_model = model.clone()
        pruning_factor = None
        pruner.prune_model_with_scores(
            copied_model, scores, is_bp, layers2prune, pruning_factor,
            pruning_count, mean_values, input_shapes)
        test_loss, test_acc, _ = train_utils.cross_entropy_loss(
            copied_model, subset_test, calculate_accuracy=True)
        train_loss, train_acc, _ = train_utils.cross_entropy_loss(
            copied_model, subset_val2, calculate_accuracy=True)
        logging.info('is_bp: %s, n: %d, test_loss%f, train_loss:%f', str(is_bp),
                     pruning_count, test_loss, train_loss)
        ls_train_loss.append(train_loss.numpy())
        ls_test_loss.append(test_loss.numpy())
        ls_test_acc.append(test_acc.numpy())
        ls_train_acc.append(train_acc.numpy())
      utils.pickle_object((ls_train_loss, ls_train_acc,
                           ls_test_loss, ls_test_acc), d_path)


def main(_):
  tf.enable_v2_behavior()
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_binding)
  if tf.io.gfile.exists(FLAGS.outdir):
    if FLAGS.override_existing_dir:
      tf.io.gfile.rmtree(FLAGS.outdir)
      tf.io.gfile.makedirs(FLAGS.outdir)
      logging.info('The folder:%s has been generated', FLAGS.outdir)
    else:
      logging.warning('The folder:%s exists', FLAGS.outdir)
  else:
    tf.io.gfile.makedirs(FLAGS.outdir)
    logging.info('The folder:%s has been generated', FLAGS.outdir)
  prune_layer_and_eval()

if __name__ == '__main__':
  app.run(main)
