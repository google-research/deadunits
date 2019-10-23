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
"""Includes common model definitions.

They don't include the output layer, so one needs to add a ['O', n_classes]
element at the end of each list.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

small_conv = [['C', 32, 5, {}],
              ['MP', 2, 2],
              ['C', 64, 3, {}],
              ['MP', 2, 2],
              ['C', 128, 3, {}],
              ['MP', 2, 2],
              ['F'],
              ['D', 512],
              ['D', 128]]

medium_conv = [['C', 64, 5, {}],
               ['MP', 2, 2],
               ['C', 128, 3, {}],
               ['MP', 2, 2],
               ['C', 256, 3, {}],
               ['MP', 2, 2],
               ['F'],
               ['D', 1024],
               ['D', 256]]

morcos_conv = [['C', 64, 3, {'padding': 'same'}],
               ['C', 64, 3, {'padding': 'same'}],
               ['C', 128, 3, {'padding': 'same', 'strides': 2}],
               ['C', 128, 3, {'padding': 'same'}],
               ['C', 128, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same', 'strides': 2}],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same', 'strides': 2}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['F']]

morcos_convfc = [['C', 64, 3, {'padding': 'same'}],
                 ['C', 128, 3, {'padding': 'same', 'strides': 2}],
                 ['C', 128, 3, {'padding': 'same'}],
                 ['C', 256, 3, {'padding': 'same', 'strides': 2}],
                 ['C', 256, 3, {'padding': 'same'}],
                 ['C', 512, 3, {'padding': 'same', 'strides': 2}],
                 ['C', 512, 3, {'padding': 'same'}],
                 ['F'],
                 ['D', 1024],
                 ['D', 128]]

vgg_11_base = [['C', 64, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 128, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2]]

vgg_13_base = [['C', 64, 3, {'padding': 'same'}],
               ['C', 64, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 128, 3, {'padding': 'same'}],
               ['C', 128, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2]]

vgg_16_base = [['C', 64, 3, {'padding': 'same'}],
               ['C', 64, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 128, 3, {'padding': 'same'}],
               ['C', 128, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2]]

vgg_19_base = [['C', 64, 3, {'padding': 'same'}],
               ['C', 64, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 128, 3, {'padding': 'same'}],
               ['C', 128, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['C', 256, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['C', 512, 3, {'padding': 'same'}],
               ['MP', 2, 2]]

vgg_fc_end = [['F'],
              ['D', 4096],
              ['DO', 0.5],
              ['D', 4096],
              ['DO', 0.5]]

vgg_conv_end = [['C', 4096, 7, {'padding': 'valid'}],
                ['DO', 0.5],
                ['C', 4096, 1, {'padding': 'same'}],
                ['DO', 0.5],
                ['F']]

vgg_gap_end = [['GA']]

vgg_11 = vgg_11_base + vgg_fc_end
vgg_13 = vgg_13_base + vgg_fc_end
vgg_16 = vgg_16_base + vgg_fc_end
vgg_19 = vgg_19_base + vgg_fc_end

vgg_11_convend = vgg_11_base + vgg_conv_end
vgg_13_convend = vgg_13_base + vgg_conv_end
vgg_16_convend = vgg_16_base + vgg_conv_end
vgg_19_convend = vgg_19_base + vgg_conv_end

vgg_11_gapend = vgg_11_base + vgg_gap_end
vgg_13_gapend = vgg_13_base + vgg_gap_end
vgg_16_gapend = vgg_16_base + vgg_gap_end
vgg_19_gapend = vgg_19_base + vgg_gap_end


alexnet_imagenet = [['C', 96, [11, 11], {'strides': (4, 4)}],
                    ['MP', 3, 2],
                    ['C', 256, [5, 5], {}],
                    ['MP', 3, 2],
                    ['C', 384, [3, 3], {}],
                    ['C', 384, [3, 3], {}],
                    ['C', 256, [3, 3], {}],
                    ['F'],
                    ['D', 4096],
                    ['D', 4096]]

alexnet_cifar10 = [['C', 96, [5, 5], {'strides': (2, 2),
                                      'padding': 'same'}],
                   ['MP', 2, 2],
                   ['C', 256, [5, 5], {'padding': 'same'}],
                   ['MP', 2, 2],
                   ['C', 384, [3, 3], {'padding': 'same'}],
                   ['C', 384, [3, 3], {'padding': 'same'}],
                   ['C', 256, [3, 3], {'padding': 'same'}],
                   ['F'],
                   ['D', 256]]
