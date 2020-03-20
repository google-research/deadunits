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
from absl import logging

from deadunits import data
from deadunits import model_load
from deadunits import pruner
from deadunits import train_utils
from deadunits import utils
from deadunits.train_utils import cross_entropy_loss
import gin
import tensorflow.compat.v2 as tf
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
  tf.random.set_seed(seed)
  optimizer = tf.keras.optimizers.SGD(lr, momentum=momentum)
  # imagenet_vgg->imagenet
  dataset_basename = dataset_name.split('_')[0]
  model = model_load.get_model(dataset_name=dataset_basename)
  # Uncomment following if your model is (defunable).
  # model.call = tf.function(model.call)
  datasets = data.get_datasets(dataset_name=dataset_name)
  (dataset_train, _, subset_val, subset_test, subset_val2) = datasets
  logging.info('Model init-config: %s', model.init_config)
  logging.info('Model forward chain: %s', str(model.forward_chain))

  unit_pruner = pruner.UnitPruner(model, subset_val)
  # We prune all conv layers.
  pruning_pool = _all_vgg_conv_layers
  baselines = {l_name: c_flop * flop_regularizer for l_name, c_flop
               in zip(pruning_pool, _vgg16_flop_regularizition)}

  step_counter = optimizer.iterations
  tf.summary.experimental.set_step(step_counter)
  c_pruning_step = tf.Variable(1)
  # Create checkpoint object TODO check whether you need ckpt-prefix.
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer,
      model=model,
      c_pruning_step=c_pruning_step)
  # No limits basically by setting to `n_units_target`.
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=FLAGS.outdir, max_to_keep=None)

  latest_cpkt = checkpoint_manager.latest_checkpoint
  if latest_cpkt:
    logging.info('Using latest checkpoint at %s', latest_cpkt)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)
    logging.info('Resuming with pruning step: %d', c_pruning_step.numpy())
  pruning_step = c_pruning_step.numpy()
  while pruning_step <= n_units_target:
    for (x, y) in dataset_train:
      # Every `n_finetune` step perform pruning.
      if step_counter.numpy() % n_finetune == 0:
        if pruning_step > n_units_target:
          # This holds true when we prune last time and fine tune N many
          # iterations. We would break and the while loop above would break,
          # too.
          break
        tf.logging.info('Pruning Step:%d', pruning_step)
        start = time.time()
        unit_pruner.prune_one_unit(pruning_pool, baselines=baselines)
        end = time.time()
        tf.logging.info('\nTrain time for Pruning Step #%d (step %d): %f',
                        pruning_step,
                        tf.train.get_or_create_global_step().numpy(),
                        end - start)
        pruning_step += 1
        c_pruning_step.assign(pruning_step)
        if tf.equal((pruning_step - 1) % checkpoint_interval, 0):
          checkpoint_manager.save()
      if step_counter.numpy() % log_interval == 0:
        train_utils.log_loss_acc(model, subset_val2, subset_test)
        train_utils.log_sparsity(model)
      with tf.GradientTape() as tape:
        loss_train, _, _ = cross_entropy_loss(model, (x, y), training=True)
      grads = tape.gradient(loss_train, model.variables)
      # Updating the model.
      optimizer.apply_gradients(
          list(zip(grads, model.variables)), global_step=step_counter)
      if step_counter.numpy() % log_interval == 0:
        tf.summary.scalar('loss_train', loss_train)
        tf.summary.image('x', x, max_outputs=1)


def main(_):
  tf.enable_v2_behavior()
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_binding)
  if tf.io.gfile.exists(FLAGS.outdir):
    if FLAGS.override_existing_dir:
      tf.io.gfile.rmtree(FLAGS.outdir)
      tf.io.gfile.makedirs(FLAGS.outdir)
      logging.info('The folder:%s has been re-generated', FLAGS.outdir)
    else:
      logging.warning('The folder:%s exists', FLAGS.outdir)
  else:
    tf.io.gfile.makedirs(FLAGS.outdir)
    logging.info('The folder:%s has been generated', FLAGS.outdir)
  tb_path = os.path.join(FLAGS.outdir, 'tb')
  summary_writer = tf.summary.create_file_writer(tb_path, flush_millis=1000)
  with summary_writer.as_default():
    prune_and_finetune_model()
  logconfigfile_path = os.path.join(FLAGS.outdir, 'config.log')
  if tf.io.gfile.exists(logconfigfile_path):
    # This should happen when the process is preempted and continued afterwards.
    logging.warning('The log_file:%s exists', logconfigfile_path)
  else:
    with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
      f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  app.run(main)
