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

Follows the strategy given at https://arxiv.org/abs/1611.06440.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
    'configs/prune_all_layers_default.gin',
    'List of paths to the config files.')
flags.DEFINE_multi_string('gin_binding', None,
                          'Newline separated list of Gin parameter bindings.')


@gin.configurable
def prune_and_finetune_model(pruning_schedule=gin.REQUIRED,
                             model_dir=gin.REQUIRED,
                             dataset_name='cifar10',
                             checkpoint_interval=5,
                             log_interval=5,
                             n_finetune=100,
                             epochs=20,
                             lr=1e-4,
                             momentum=0.9):
  """Loads and prunes the model layer by layer according to the schedule.

  The model is finetuned between pruning tasks (as given in the schedule).

  Args:
    pruning_schedule: list<str, float>, where the str is a valid layer name and
      the float is the pruning fraction of that layer. Layers are pruned in
      the order they are given and `n_finetune` steps taken in between.
    model_dir: str, Path to the checkpoint directory.
    dataset_name: str, either 'cifar10' or 'imagenet'.
    checkpoint_interval: int, number of epochs between two checkpoints.
    log_interval: int, number of steps between two logging events.
    n_finetune: int, number of steps between two pruning steps.
    epochs: int, total number of epochs to run.
    lr: float, learning rate for the fine-tuning steps.
    momentum: float, momentum multiplier for the fine-tuning steps.
  Raises:
    ValueError: when the n_finetune is not positive.
    OSError: when there is no checkpoint found.
  """
  if n_finetune <= 0:
    raise ValueError('n_finetune must be positive: given %d' % n_finetune)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
  logging.info('Using outdir: %s', model_dir)
  latest_cpkt = tf.train.latest_checkpoint(model_dir)
  if not latest_cpkt:
    raise OSError('No checkpoint found in %s' % model_dir)
  logging.info('Using latest checkpoint at %s', latest_cpkt)
  model = model_load.get_model(load_path=latest_cpkt,
                               dataset_name=dataset_name)
  logging.info('Model init-config: %s', model.init_config)
  logging.info('Model forward chain: %s', str(model.forward_chain))
  datasets = data.get_datasets(dataset_name=dataset_name)
  dataset_train, dataset_test, subset_val, subset_test, subset_val2 = datasets

  unit_pruner = pruner.UnitPruner(model, subset_val)
  step_counter = optimizer.iteration
  tf.summary.experimental.set_step(step_counter)
  current_epoch = tf.Variable(1)
  current_layer_index = tf.Variable(0)
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer,
      model=model,
      current_epoch=current_epoch,
      current_layer_index=current_layer_index)
  latest_cpkt = tf.train.latest_checkpoint(FLAGS.outdir)
  if latest_cpkt:
    logging.info('Using latest checkpoint at %s', latest_cpkt)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)
    logging.info('Resuming with epoch: %d', current_epoch.numpy())
  c_epoch = current_epoch.numpy()
  c_layer_index = current_layer_index.numpy()
  # Starting from the first batch, we perform pruning every `n_finetune` step.
  # Layers pruned one by one according to the pruning schedule given.

  while c_epoch <= epochs:
    logging.info('Starting Epoch: %d', c_epoch)
    for (x, y) in dataset_train:
      # Every `n_finetune` step perform pruning.
      if (step_counter.numpy() % n_finetune == 0 and
          c_layer_index < len(pruning_schedule)):
        logging.info('Pruning at iteration: %d', step_counter.numpy())
        l_name, pruning_factor = pruning_schedule[c_layer_index]
        unit_pruner.prune_layer(l_name, pruning_factor=pruning_factor)

        train_utils.log_loss_acc(model, subset_val2, subset_test)
        train_utils.log_sparsity(model)
        # Re-init optimizer and therefore remove previous momentum.
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        c_layer_index += 1
        current_layer_index.assign(c_layer_index)
      else:
        if step_counter.numpy() % log_interval == 0:
          logging.info('Iteration: %d', step_counter.numpy())
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
    # End of an epoch.
    c_epoch += 1
    current_epoch.assign(c_epoch)
    # Save every n OR after last epoch.
    if (tf.equal((current_epoch - 1) % checkpoint_interval, 0) or
        c_epoch > epochs):
      # Re-init checkpoint to ensure the masks are captured. The reason for
      # this is that the masks are initially not generated.
      checkpoint = tf.train.Checkpoint(
          optimizer=optimizer,
          model=model,
          current_epoch=current_epoch,
          current_layer_index=current_layer_index)
      logging.info('Checkpoint after epoch: %d', c_epoch - 1)
      checkpoint.save(os.path.join(FLAGS.outdir, 'ckpt-%d' % (c_epoch - 1)))

  # Test model
  test_loss, test_acc, n_samples = cross_entropy_loss(
      model, dataset_test, calculate_accuracy=True)
  tf.summary.scalar('test_loss_all', test_loss)
  tf.summary.scalar('test_acc_all', test_acc)
  logging.info('Overall_test_loss: %.4f, Overall_test_acc: %.4f, n_samples: %d',
               test_loss, test_acc, n_samples)


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
    logging.warning('The log_file: %s exists', logconfigfile_path)
  else:
    with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
      f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  app.run(main)
