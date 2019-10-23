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

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r deadunits/requirements.txt
python setup.py install
python -m deadunits.generic_convnet_test
python -m deadunits.layers_test
python -m deadunits.pruner_test
python -m deadunits.model_defs_test
python -m deadunits.train_utils_test
python -m deadunits.unitscorers_test
python -m deadunits.utils_test
