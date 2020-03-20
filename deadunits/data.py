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
"""Dataset operations for retrieving and preprocessing.

Supports 2 dataset for now: cifar10 and imagenet.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os
import gin
from tensor2tensor import problems
from tensor2tensor.data_generators.imagenet import imagenet_preprocess_example
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.estimator import ModeKeys

_t2t_problem_names = {
    'cifar10': 'image_cifar10',
    'imagenet': 'image_imagenet224',
    'imagenet_vgg': 'image_imagenet224_no_normalization',
    'cub200': None
}

N_CLASSES_BY_DATASET = {'cifar10': 10, 'imagenet': 1000, 'cub200': 200}


def get_partition(dataset, start, n_samples):
  """Returns a subset of a given dataset by skipping data.

    Be sure dataset has samples more than `start + n_samples`. Otherwise the
  returned dataset might be an empty one.
  Args:
    dataset: tf.data.Dataset
    start: int, starting index for data.
    n_samples: number_of samples to take from dataset.

  Returns:
    tf.data.Dataset, with `n_samples`.
  """
  return dataset.skip(start).take(n_samples)


def _keras_vgg16_preprocess(x):
  # RGB->BGR and mean substraction.
  return x[Ellipsis, ::-1] - [103.939, 116.779, 123.68]


def _do_scale(image, size):
  """Rescale the image by scaling the smaller spatial dimension to `size`."""
  shape = tf.cast(tf.shape(image), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(w_greater,
                  lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                  lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

  return tf.image.resize([image], shape, method='bicubic')[0]


def load_and_process_example(example_string, mode,
                             image_size=224, preprocess=True):
  """To process records read from tfRecords file.

  Args:
    example_string: str, serialized string record.
    mode: str, from tf.estimator.ModeKeys. Decides how the data is preprocessed.
    image_size: int, for resizing the image.
    preprocess: bool, if true resized to `image_size`.

  Returns:
    {'inputs': image, 'targets':label}
  """
  data = tf.io.parse_single_example(
      example_string,
      features={
          'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
          'image/class/label': tf.io.FixedLenFeature([], tf.int64)
      })
  image_string = data['image/encoded']
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  if preprocess:
    # The following does random crop/flip for training and center crop for test.
    image_decoded = _do_scale(image_decoded, image_size + 32)
    image_decoded = imagenet_preprocess_example({'inputs': image_decoded},
                                                mode,
                                                resize_size=(image_size,
                                                             image_size),
                                                normalize=False)['inputs']
    image_decoded = _keras_vgg16_preprocess(image_decoded)
  return {'inputs': image_decoded, 'targets': data['image/class/label']}


def cub200_loader(mode,
                  dataset_split=None,
                  preprocess=True,
                  shuffle_files=None,
                  shuffle_buffer_size=1024,
                  num_threads=None,
                  output_buffer_size=None,
                  data_dir=None):
  """Data Loader for cu-Birds 200 dataset.

  Loads data preprocessed using https://github.com/visipedia/tf_classification/
    wiki/CUB-200-Image-Classification
  Data loader mimics the API of `t2t.problems.Problem.dataset()`.

  Args:
   mode: str, from tf.estimator.ModeKeys. Decides whether the data is shuffled
   dataset_split: str, from tf.estimator.ModeKeys used to decide which data to
     load. If None, set to the `mode`.
   preprocess: bool, whether to process images while loading.
   shuffle_files: bool, whether to shuffle files during evaluation. If None and
     mode is train, than shuffled.
   shuffle_buffer_size: int, shuffle buffer size.
   num_threads: int, passed to the map operation.
   output_buffer_size: int, passed to the prefetch operator.
   data_dir: str, path for where data lies or will be downloaded.

  Returns:
    tf.data.Dataset
  """
  dataset_split = mode if dataset_split is None else dataset_split
  # Push this to t2t later.
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  is_training_data = dataset_split == tf.estimator.ModeKeys.TRAIN
  shuffle_files = shuffle_files or shuffle_files is None and is_training
  all_files = tf.io.gfile.listdir(data_dir)
  # This assumes that the TFRecords are prefixed with 'train', 'val' and 'test'.
  # We currently support two split: (1) Train: reads both `train` and `val`
  # into one Dataset. (2) Test: only test records are read.
  record_prefixes = ('train', 'val') if is_training_data else 'test'
  filtered_paths = [
      os.path.join(data_dir, s)
      for s in all_files
      if s.startswith(record_prefixes)
  ]
  dataset = tf.data.TFRecordDataset(filtered_paths)
  process_example = functools.partial(load_and_process_example,
                                      mode=mode,
                                      preprocess=preprocess)
  dataset = dataset.map(process_example, num_parallel_calls=num_threads)
  if shuffle_files:
    dataset = dataset.shuffle(shuffle_buffer_size)
  if output_buffer_size:
    dataset = dataset.prefetch(output_buffer_size)
  return dataset


@gin.configurable
def get_datasets(dataset_name='cifar10',
                 eval_size=1000,
                 val_size=1000,
                 batch_size=32,
                 shuffle_size=500,
                 num_parallel_calls=None,
                 prefetch_n=1,
                 chunk_size=-1,
                 data_dir=None,
                 val_disjoint=True):
  """Processes and returns the train and test sets of the dataset asked.

  Each dataset is shuffled independently.
  Returned elements from the iterator has the following form: [x, y]

  Args:
   dataset_name: str, dataset name from `data._t2t_problem_names.keys()`.
   eval_size: int, size for the second validation set and the test set.
   val_size: int, number of images from training set to calculate saliency
     scores and the training loss.
   batch_size: int, default batch size for not repating dataset versions.
   shuffle_size: int, shuffle buffer size.
   num_parallel_calls: int, passed to the map operation.
   prefetch_n: int, passed to the prefetch operator.
   chunk_size: int, batch size for the subsets (i.e. `subset_train`,
     `subset_test`). If <=0, not used.
   data_dir: str, path for where data lies or will be downloaded.
   val_disjoint: bool, if True val and val2 are disjoint meaning. We would take
     first `val_size` samples as the dataset as `subset_val` and next
     `eval_size` as `subset_val2`. Otherwise both of them are taken from the
     beginning, so they overlap.

  Returns:
    dataset_train: for sampling batches from training set 1 time only (1 epoch).
      batched with `batch_size`. Shuffled with buffer size
      min(n_training_samples, shuffle_size_max).
    dataset_test: for sampling batches from test set 1 time only (1 epoch).
      batched with `batch_size`. Shuffled with buffer size
      min(n_test_samples, shuffle_size_max).
    subset_val: `val_size` samples from training set batched as `val_size` if
     chunk_size=None `else` batch_size=chunk_size. Not shuffled.
    subset_test: `eval_size` samples from test set batched as `eval_size` if
     chunk_size=None `else` batch_size=chunk_size. Not shuffled.
    subset_val2: `eval_size` samples from training set batched as `eval_size` if
     chunk_size=None `else` batch_size=chunk_size. Not shuffled.

   Raises:
     AssertionError: if `dataset_name` is not valid.
  """
  assert dataset_name in _t2t_problem_names
  if dataset_name == 'cub200':
    dataset_fn = cub200_loader
  else:
    problem_name = _t2t_problem_names[dataset_name]
    problem = problems.problem(problem_name)
    dataset_fn = problem.dataset
  is_vgg = dataset_name == 'imagenet_vgg'
  to_float_fn = lambda x: tf.cast(x, dtype=tf.float32) / 255.
  # Vgg model has its colors in different ordering, so we need to reverse it.
  if is_vgg:
    input_fn = lambda x: _keras_vgg16_preprocess(to_float_fn(x))
  else:
    input_fn = to_float_fn
  # Imagenet label's start from 1. So we need to offset.
  is_imagenet = dataset_name.startswith('imagenet')
  label_fn = lambda a: tf.squeeze(a - 1 if is_imagenet else a)
  data_fn = lambda x: (input_fn(x['inputs']), label_fn(x['targets']))

  subset_val_bs = val_size if chunk_size <= 0 else chunk_size
  subset_eval_bs = eval_size if chunk_size <= 0 else chunk_size
  # Loading the dataset.
  dataset_train = dataset_fn(
      ModeKeys.TRAIN,
      preprocess=True,
      shuffle_buffer_size=shuffle_size,
      num_threads=num_parallel_calls,
      output_buffer_size=prefetch_n,
      data_dir=data_dir).map(data_fn).batch(batch_size)
  # ModeKeys.EVAL to disable shuffling and get same subset.
  subset_val = dataset_fn(
      ModeKeys.EVAL,
      dataset_split=ModeKeys.TRAIN,
      preprocess=True,
      shuffle_buffer_size=shuffle_size,
      num_threads=num_parallel_calls,
      output_buffer_size=prefetch_n,
      data_dir=data_dir)
  subset_val = get_partition(subset_val, 0,
                             val_size).map(data_fn).batch(subset_val_bs)
  subset_val2 = dataset_fn(
      ModeKeys.EVAL,
      dataset_split=ModeKeys.TRAIN,
      preprocess=True,
      shuffle_buffer_size=shuffle_size,
      num_threads=num_parallel_calls,
      output_buffer_size=prefetch_n,
      data_dir=data_dir)
  subset_val2 = get_partition(subset_val2, val_size if val_disjoint else 0,
                              eval_size).map(data_fn).batch(subset_eval_bs)

  dataset_test = dataset_fn(
      ModeKeys.EVAL,
      preprocess=True,
      shuffle_buffer_size=shuffle_size,
      num_threads=num_parallel_calls,
      output_buffer_size=prefetch_n,
      data_dir=data_dir).map(data_fn).batch(batch_size)
  subset_test = dataset_fn(
      ModeKeys.EVAL,
      preprocess=True,
      shuffle_buffer_size=shuffle_size,
      num_threads=num_parallel_calls,
      output_buffer_size=prefetch_n,
      data_dir=data_dir)
  subset_test = get_partition(subset_test, 0,
                              eval_size).map(data_fn).batch(subset_eval_bs)

  return (dataset_train, dataset_test, subset_val, subset_test, subset_val2)
