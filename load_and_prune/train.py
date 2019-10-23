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

r"""Main training loop for various architectures, datasets.

Supports loading pretrained models and finetuning.

Main script for running experiments.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from absl import app
from absl import flags
from deadunits import data
from deadunits import model_load
from deadunits import train_utils
from deadunits import utils
from deadunits.train_utils import cross_entropy_loss
import gin
import tensorflow as tf
import tensorflow.contrib.eager as tfe
FLAGS = flags.FLAGS

flags.DEFINE_string('outdir', '/tmp/dead_units/test',
                    'Path to directory where to store summaries and models.')
flags.DEFINE_boolean('is_debug', False, 'sets the logging level to debug')
flags.DEFINE_boolean('override_existing_dir', False, 'remove if exists')
flags.DEFINE_multi_string(
    'gin_config',
    'configs/train_default.gin',
    'List of paths to the config files.')
flags.DEFINE_multi_string('gin_binding', None,
                          'Newline separated list of Gin parameter bindings.')


@gin.configurable
def train_model(dataset_name='cifar10',
                checkpoint_every_n_epoch=5,
                log_interval=1000,
                epochs=10,
                lr=1e-2,
                lr_drop_iter=1500,
                lr_decay=0.5,
                momentum=0.9,
                seed=8):
  """Trains the model with regular logging and checkpoints.

  Args:
    dataset_name: str, either 'cifar10' or 'imagenet'.
    checkpoint_every_n_epoch: int, number of epochs between two checkpoints.
    log_interval: int, number of steps between two logging events.
    epochs: int, epoch to train with.
    lr: float, learning rate for the fine-tuning steps.
    lr_drop_iter: int, iteration between two consequtive lr drop.
    lr_decay: float: multiplier for step learning rate reduction.
    momentum: float, momentum multiplier for the fine-tuning steps.
    seed: int, random seed to be set to produce reproducible experiments.

  Raises:
    AssertionError: when the args doesn't match the specs.
  """
  assert dataset_name in ['cifar10', 'imagenet', 'cub200', 'imagenet_vgg']
  tf.set_random_seed(seed)
  # The model is configured through gin parameters.
  model = model_load.get_model(dataset_name=dataset_name)

  # model.call = tf.contrib.eager.defun(model.call)
  datasets = data.get_datasets(dataset_name=dataset_name)
  dataset_train, dataset_test, _, subset_test, subset_val = datasets
  step_counter = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
      lr,
      step_counter,
      lr_drop_iter,
      lr_decay,
      staircase=True)
  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  tf.logging.info('Model init-config: %s', model.init_config)
  tf.logging.info('Model forward chain: %s', str(model.forward_chain))

  current_epoch = tfe.Variable(1)
  # Create checkpoint object TODO check whether you need ckpt-prefix.
  checkpoint = tfe.Checkpoint(
      optimizer=optimizer, model=model, step_counter=step_counter,
      current_epoch=current_epoch)
  # No limits basically by setting to `n_units_target`.
  checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
      checkpoint, directory=FLAGS.outdir, max_to_keep=None)

  latest_cpkt = checkpoint_manager.latest_checkpoint
  if latest_cpkt:
    tf.logging.info('Using latest checkpoint at %s', latest_cpkt)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)
    tf.logging.info('Resuming with epoch: %d', current_epoch.numpy())
  c_epoch = current_epoch.numpy()
  with tf.contrib.summary.record_summaries_every_n_global_steps(
      log_interval, global_step=step_counter):
    while c_epoch <= epochs:
      tf.logging.info('Starting Epoch:%d', c_epoch)
      for (x, y) in dataset_train:
        if tf.contrib.summary.should_record_summaries():
          tf.logging.info('Iteration:%d', step_counter.numpy())
          train_utils.log_loss_acc(model, subset_val, subset_test)
        with tf.GradientTape() as tape:
          tf.contrib.summary.image('x', x, max_images=1)
          loss_train, _, _ = cross_entropy_loss(model, (x, y), training=True)
        grads = tape.gradient(loss_train, model.variables)
        # Updating the model.
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=step_counter)
        tf.contrib.summary.scalar('loss_train', loss_train)
        tf.contrib.summary.scalar('lr', learning_rate())
      # End of an epoch.
      c_epoch += 1
      current_epoch.assign(c_epoch)
      # Save every n OR after last epoch.
      if (tf.equal((current_epoch - 1) % checkpoint_every_n_epoch, 0)
          or c_epoch > epochs):
        tf.logging.info('Checkpoint after epoch: %d', c_epoch-1)
        checkpoint_manager.save(checkpoint_number=c_epoch-1)
  # Test model
  with tf.contrib.summary.always_record_summaries():
    test_loss, test_acc, n_samples = cross_entropy_loss(
        model, dataset_test, calculate_accuracy=True)
    tf.contrib.summary.scalar('test_loss_all', test_loss)
    tf.contrib.summary.scalar('test_acc_all', test_acc)
  tf.logging.info('Overall_test_loss:%.4f, Overall_test_acc:%.4f, '
                  'n_samples:%d', test_loss, test_acc, n_samples)


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
    train_model()

  logconfigfile_path = os.path.join(FLAGS.outdir, 'config.log')
  if tf.gfile.Exists(logconfigfile_path):
    # This should happen when the process is preempted and continued afterwards.
    tf.logging.warning('The log_file:%s exists', logconfigfile_path)
  else:
    with tf.gfile.GFile(logconfigfile_path, 'w') as f:
      f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  app.run(main)
