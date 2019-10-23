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

r"""Loads and prunes model with global scoring and one unit at a time.

Follow's the strategy given at https://arxiv.org/abs/1611.06440.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
from absl import app
from absl import flags
from deadunits import data
from deadunits import model_load
from deadunits import pruner
from deadunits import train_utils
from deadunits import utils
from deadunits.train_utils import cross_entropy_loss
import gin
import tensorflow as tf
import tensorflow.contrib.eager as tfe
FLAGS = flags.FLAGS

flags.DEFINE_string('outdir', '/tmp/dead_units/test',
                    'Path to directory where to store summaries.')
flags.DEFINE_boolean('is_debug', False, 'sets the logging level to debug')
flags.DEFINE_boolean('override_existing_dir', False, 'remove if exists')
flags.DEFINE_multi_string(
    'gin_config',
    'configs/global_prune_default.gin',
    'List of paths to the config files.')
flags.DEFINE_multi_string('gin_binding', None,
                          'Newline separated list of Gin parameter bindings.')


_all_vgg_conv_layers = ['conv_%d' % i for i in range(1, 14)]
# Taken from Molchanov's `Pruning Convolutional Networks for...` paper.
_vgg16_flop_regularizition = [3.1, 57.8, 14.1, 28.9, 7.0, 14.5, 14.5,
                              3.5, 7.2, 7.2, 1.8, 1.8, 1.8, 1.8]


@gin.configurable
def prune_and_finetune_model(dataset_name='imagenet_vgg',
                             flop_regularizer=0,
                             n_units_target=4000,
                             checkpoint_interval=5,
                             log_interval=5,
                             n_finetune=100,
                             lr=1e-4,
                             momentum=0.9,
                             seed=8):
  """Trains the model with regular logging and checkpoints.

  Args:
    dataset_name: str, dataset to train on.
    flop_regularizer: float, multiplier for the flop regularization. If 0, no
      regularization is made during pruning.
    n_units_target: int, number of unit to prune.
    checkpoint_interval: int, number of epochs between two checkpoints.
    log_interval: int, number of steps between two logging events.
    n_finetune: int, number of steps between two pruning steps. Starting from
      the first iteration we prune 1 unit every `n_finetune` gradient update.
    lr: float, learning rate for the fine-tuning steps.
    momentum: float, momentum multiplier for the fine-tuning steps.
    seed: int, random seed to be set to produce reproducible experiments.
  Raises:
    ValueError: when the n_finetune is not positive.
  """
  if n_finetune <= 0:
    raise ValueError('n_finetune must be positive: given %d' % n_finetune)
  tf.set_random_seed(seed)
  optimizer = tf.train.MomentumOptimizer(lr, momentum)
  # imagenet_vgg->imagenet
  dataset_basename = dataset_name.split('_')[0]
  model = model_load.get_model(dataset_name=dataset_basename)
  # Uncomment following if your model is (defunable).
  # model.call = tf.contrib.eager.defun(model.call)
  datasets = data.get_datasets(dataset_name=dataset_name)
  (dataset_train, _, subset_val, subset_test, subset_val2) = datasets
  tf.logging.info('Model init-config: %s', model.init_config)
  tf.logging.info('Model forward chain: %s', str(model.forward_chain))

  unit_pruner = pruner.UnitPruner(model, subset_val)
  # We prune all conv layers.
  pruning_pool = _all_vgg_conv_layers
  baselines = {l_name: c_flop * flop_regularizer for l_name, c_flop
               in zip(pruning_pool, _vgg16_flop_regularizition)}

  step_counter = tf.train.get_or_create_global_step()
  c_pruning_step = tfe.Variable(1)
  # Create checkpoint object TODO check whether you need ckpt-prefix.
  checkpoint = tfe.Checkpoint(
      optimizer=optimizer, model=model, step_counter=step_counter,
      c_pruning_step=c_pruning_step)
  # No limits basically by setting to `n_units_target`.
  checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
      checkpoint, directory=FLAGS.outdir, max_to_keep=None)

  latest_cpkt = checkpoint_manager.latest_checkpoint
  if latest_cpkt:
    tf.logging.info('Using latest checkpoint at %s', latest_cpkt)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)
    tf.logging.info('Resuming with pruning step: %d', c_pruning_step.numpy())
  pruning_step = c_pruning_step.numpy()
  while pruning_step <= n_units_target:
    with tf.contrib.summary.record_summaries_every_n_global_steps(
        log_interval, global_step=step_counter):
      for (x, y) in dataset_train:
        # Every `n_finetune` step perform pruning.
        if tf.equal(step_counter % n_finetune, 0):
          if pruning_step > n_units_target:
            # This holds true when we prune last time and fine tune N many
            # iterations. We would break and the while loop above would break,
            # too.
            break
          tf.logging.info('Pruning Step:%d', pruning_step)
          start = time.time()
          unit_pruner.prune_one_unit(pruning_pool, baselines=baselines)
          end = time.time()
          tf.logging.info(
              '\nTrain time for Pruning Step #%d (step %d): %f',
              pruning_step,
              tf.train.get_or_create_global_step().numpy(),
              end - start)
          pruning_step += 1
          c_pruning_step.assign(pruning_step)
          if tf.equal((pruning_step - 1) % checkpoint_interval, 0):
            checkpoint_manager.save()
        if tf.contrib.summary.should_record_summaries():
          train_utils.log_loss_acc(model, subset_val2, subset_test)
          train_utils.log_sparsity(model)
        with tf.GradientTape() as tape:
          tf.contrib.summary.image('x', x, max_images=1)
          loss_train, _, _ = cross_entropy_loss(model, (x, y), training=True)
        grads = tape.gradient(loss_train, model.variables)
        # Updating the model.
        tf.contrib.summary.scalar('loss_train', loss_train)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=step_counter)


def main(_):
  tf.enable_eager_execution()
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_binding)
  if tf.gfile.Exists(FLAGS.outdir):
    if FLAGS.override_existing_dir:
      tf.gfile.DeleteRecursively(FLAGS.outdir)
      tf.gfile.MakeDirs(FLAGS.outdir)
      tf.logging.info('The folder:%s has been re-generated', FLAGS.outdir)
    else:
      tf.logging.warning('The folder:%s exists', FLAGS.outdir)
  else:
    tf.gfile.MakeDirs(FLAGS.outdir)
    tf.logging.info('The folder:%s has been generated', FLAGS.outdir)
  tb_path = os.path.join(FLAGS.outdir, 'tb')
  summary_writer = tf.contrib.summary.create_file_writer(
      tb_path, flush_millis=1000)
  with summary_writer.as_default():
    prune_and_finetune_model()
  logconfigfile_path = os.path.join(FLAGS.outdir, 'config.log')
  if tf.gfile.Exists(logconfigfile_path):
    # This should happen when the process is preempted and continued afterwards.
    tf.logging.warning('The log_file:%s exists', logconfigfile_path)
  else:
    with tf.gfile.GFile(logconfigfile_path, 'w') as f:
      f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  app.run(main)
