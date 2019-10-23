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

"""Setup file for lottery ticket hypothesis."""

from setuptools import setup

SHORT_DESCRIPTION = """
Pruning library for neural networks focused on structured/unit
pruning.""".strip()

DEPENDENCIES = [
    'six>=1.10',
    'gin-config>=0.1.2',
    'tensorflow==1.14.0',
    'tensor2tensor',
    'mock',
    'numpy>=1.13.3',
    'absl-py>=0.7',
]


VERSION = '1'
URL = 'https://github.com/google-research/deadunits'

setup(
    name='deadunits',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    url=URL,

    author='Utku Evci',
    author_email='evcu@google.com',
    license='Apache Software License',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
    ],

    keywords='deadunits structured pruning',

    packages=['deadunits'],

    install_requires=DEPENDENCIES,
)
