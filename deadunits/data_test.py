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
"""Tests for `deadunits.data`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deadunits import data
import mock
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()


class GetDatasetsTest(tf.test.TestCase):

  def testGetPartition(self):
    cases = [(0, 3),
             (3, 3),
             (1, 5),
             (15, 17)]
    dataset = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(20),
                                                                1))
    for start, n_samples in cases:
      partition = data.get_partition(dataset, start, n_samples)
      for i, t in zip(list(range(start, start + n_samples)), partition):
        self.assertEqual(i, int(t))

  def testDefaultArgs(self):
    bs = 32
    val_size = eval_size = 1000
    datasets = data.get_datasets(shuffle_size=1)
    (dataset_train, dataset_test,
     subset_val, subset_test, subset_val2) = datasets
    x, y = next(dataset_train.__iter__())
    self.assertEqual(x.shape, [bs, 32, 32, 3])
    self.assertEqual(y.shape, [bs])
    self.assertLessEqual(tf.reduce_max(x), 1.0)
    self.assertGreaterEqual(tf.reduce_min(x), 0)
    x, y = next(dataset_test.__iter__())
    self.assertLessEqual(tf.reduce_max(x), 1.0)
    self.assertGreaterEqual(tf.reduce_min(x), 0)
    self.assertEqual(x.shape, [bs, 32, 32, 3])
    self.assertEqual(y.shape, [bs])

    c_iterator = subset_val.__iter__()
    x, y = next(c_iterator)
    self.assertLessEqual(tf.reduce_max(x), 1.0)
    self.assertGreaterEqual(tf.reduce_min(x), 0)
    self.assertEqual(x.shape, [val_size, 32, 32, 3])
    self.assertEqual(y.shape, [val_size])
    # Since chunk_size=None, it should only have one batch.
    with self.assertRaises(StopIteration):
      next(c_iterator)

    c_iterator = subset_val2.__iter__()
    x2, y2 = next(c_iterator)
    self.assertLessEqual(tf.reduce_max(x2), 1.0)
    self.assertGreaterEqual(tf.reduce_min(x2), 0)
    self.assertEqual(x2.shape, [eval_size, 32, 32, 3])
    self.assertEqual(y2.shape, [eval_size])
    # Check that the subset's are disjoint.
    self.assertNotAllClose(x, x2)
    # Since chunk_size=None, it should only have one batch.
    with self.assertRaises(StopIteration):
      next(c_iterator)

    c_iterator = subset_test.__iter__()
    x, y = next(c_iterator)
    self.assertLessEqual(tf.reduce_max(x), 1.0)
    self.assertGreaterEqual(tf.reduce_min(x), 0)
    self.assertEqual(x.shape, [eval_size, 32, 32, 3])
    self.assertEqual(y.shape, [eval_size])
    with self.assertRaises(StopIteration):
      next(c_iterator)

  def testDatasetName(self):
    data.get_datasets(dataset_name='cifar10')
    data.get_datasets(dataset_name='imagenet')
    with self.assertRaises(AssertionError):
      data.get_datasets(dataset_name='cifar')
    with self.assertRaises(AssertionError):
      data.get_datasets(dataset_name='shvc')

  def testCustomImagenetArgs(self):
    bs = 4
    val_size = 15
    eval_size = 12
    chunk_size = 5
    datasets = data.get_datasets('imagenet',
                                 eval_size=eval_size,
                                 val_size=val_size,
                                 batch_size=bs,
                                 shuffle_size=1,
                                 chunk_size=chunk_size)
    (dataset_train, dataset_test, subset_val, subset_test, _) = datasets
    x, y = next(dataset_train.__iter__())
    self.assertEqual(x.shape, [bs, 224, 224, 3])
    self.assertEqual(y.shape, [bs])
    x, y = next(dataset_test.__iter__())
    self.assertEqual(x.shape, [bs, 224, 224, 3])
    self.assertEqual(y.shape, [bs])
    c_iterator = subset_val.__iter__()
    x, y = next(c_iterator)
    self.assertEqual(x.shape, [chunk_size, 224, 224, 3])
    self.assertEqual(y.shape, [chunk_size])
    # Let us consume all batches.
    for _ in range((val_size-1)//chunk_size):
      x, y = next(c_iterator)
    # Since we iterated over all batches, we should get an exception.
    with self.assertRaises(StopIteration):
      next(c_iterator)
    c_iterator = subset_test.__iter__()
    x, y = next(c_iterator)
    self.assertEqual(x.shape, [chunk_size, 224, 224, 3])
    self.assertEqual(y.shape, [chunk_size])
    for _ in range((eval_size-1)//chunk_size):
      x, y = next(c_iterator)
    last_batch_shape = eval_size%chunk_size
    self.assertEqual(x.shape, [last_batch_shape, 224, 224, 3])
    self.assertEqual(y.shape, [last_batch_shape])
    with self.assertRaises(StopIteration):
      next(c_iterator)

  @mock.patch('deadunits.data._keras_vgg16_preprocess')
  def testVgg(self, mock_pp_i):
    mock_pp_i.side_effect = lambda a: a
    bs = 4
    val_size = 15
    eval_size = 12
    datasets = data.get_datasets('imagenet_vgg',
                                 eval_size=eval_size,
                                 val_size=val_size,
                                 num_parallel_calls=1,
                                 shuffle_size=1,
                                 batch_size=bs)
    (dataset_train, _, _, _, _) = datasets
    x, y = next(dataset_train.__iter__())
    self.assertEqual(x.shape, [bs, 224, 224, 3])
    self.assertEqual(y.shape, [bs])
    # Due to data augmentation the max can be slightly bigger than 1.0.
    self.assertLessEqual(tf.reduce_max(x), 1.5)
    self.assertGreaterEqual(tf.reduce_min(x), -0.5)
    self.assertTrue(mock_pp_i.called)
    self.assertTrue(mock_pp_i.call_count, bs)

  def testCUB200(self):
    bs = 4
    val_size = 15
    eval_size = 12
    datasets = data.get_datasets('cub200',
                                 eval_size=eval_size,
                                 val_size=val_size,
                                 num_parallel_calls=1,
                                 shuffle_size=1,
                                 batch_size=bs)
    (dataset_train, _, _, _, _) = datasets
    x, y = next(dataset_train.__iter__())
    self.assertEqual(x.shape, [bs, 224, 224, 3])
    self.assertEqual(y.shape, [bs])

if __name__ == '__main__':
  tf.test.main()
