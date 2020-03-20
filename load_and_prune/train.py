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
from absl import logging

from deadunits import data
from deadunits import model_load
from deadunits import train_utils
from deadunits import utils
from deadunits.train_utils import cross_entropy_loss
import gin
import tensorflow.compat.v2 as tf
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
  tf.random.set_seed(seed)
  # The model is configured through gin parameters.
  model = model_load.get_model(dataset_name=dataset_name)

  # model.call = tf.contrib.eager.defun(model.call)
  datasets = data.get_datasets(dataset_name=dataset_name)
  dataset_train, dataset_test, _, subset_test, subset_val = datasets
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      lr, decay_steps=lr_drop_iter, decay_rate=lr_decay, staircase=True)
  optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=momentum)
  step_counter = optimizer.iterations
  tf.summary.experimental.set_step(step_counter)
  logging.info('Model init-config: %s', model.init_config)
  logging.info('Model forward chain: %s', str(model.forward_chain))

  current_epoch = tf.Variable(1)
  # Create checkpoint object TODO check whether you need ckpt-prefix.
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer,
      model=model,
      current_epoch=current_epoch)
  # No limits basically by setting to `n_units_target`.
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=FLAGS.outdir, max_to_keep=None)

  latest_cpkt = checkpoint_manager.latest_checkpoint
  if latest_cpkt:
    logging.info('Using latest checkpoint at %s', latest_cpkt)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)
    logging.info('Resuming with epoch: %d', current_epoch.numpy())
  c_epoch = current_epoch.numpy()
  while c_epoch <= epochs:
    logging.info('Starting Epoch:%d', c_epoch)
    for (x, y) in dataset_train:
      if step_counter % log_interval == 0:
        train_utils.log_loss_acc(model, subset_val, subset_test)
        tf.summary.image('x', x, max_outputs=1)
        logging.info('Iteration:%d', step_counter.numpy())
      with tf.GradientTape() as tape:
        loss_train, _, _ = cross_entropy_loss(model, (x, y), training=True)
      grads = tape.gradient(loss_train, model.variables)
      # Updating the model.
      optimizer.apply_gradients(zip(grads, model.variables))
      tf.summary.scalar('loss_train', loss_train)
      tf.summary.scalar('lr', optimizer.lr(step_counter))
    # End of an epoch.
    c_epoch += 1
    current_epoch.assign(c_epoch)
    # Save every n OR after last epoch.
    if (tf.equal((current_epoch - 1) % checkpoint_every_n_epoch, 0) or
        c_epoch > epochs):
      logging.info('Checkpoint after epoch: %d', c_epoch - 1)
      checkpoint_manager.save(checkpoint_number=c_epoch - 1)
  test_loss, test_acc, n_samples = cross_entropy_loss(
      model, dataset_test, calculate_accuracy=True)
  tf.summary.scalar('test_loss_all', test_loss)
  tf.summary.scalar('test_acc_all', test_acc)
  logging.info('Overall_test_loss:%.4f, Overall_test_acc:%.4f, n_samples:%d',
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
    train_model()

  logconfigfile_path = os.path.join(FLAGS.outdir, 'config.log')
  if tf.io.gfile.exists(logconfigfile_path):
    # This should happen when the process is preempted and continued afterwards.
    logging.warning('The log_file:%s exists', logconfigfile_path)
  else:
    with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
      f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  app.run(main)
