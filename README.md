This is not an official Google product.

## Detecting Dead Units Project
This repo involves the code for running various structured pruning algorithms
and it is built on top of the codebase used in ICLR 2018 submission:
[Mean Replacement Pruning](https://openreview.net/forum?id=BJxRVnC5Fm).

This library uses [gin-config](https://github.com/google/gin-config),
[tensor2tensor](https://github.com/tensorflow/tensor2tensor),
[tf.keras](https://www.tensorflow.org/guide/keras) and
[tf.eager](https://www.tensorflow.org/guide/eager). It
might be useful to familiarize yourself with those libraries.

### Getting Started
- Installing the package:
```bash
pip install -r deadunits/requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD
python -m deadunits.pruner_test
```
- Downloading and preparing the data. We use `t2t-datagen`.
```bash
t2t-datagen --problem=image_cifar10 --data_dir~/t2t_data --tmp_dir=~/t2t_data/tmp
```
- Training models on cifar-10.
```bash
cd load_and_prune
export OUTDIR=/tmp/dead_units/test
python train.py --gin_binding=get_datasets.data_dir=\"${HOME}/t2t_data\" \
  --gin_binding=get_model.model_arch_name=\"small_conv\" --outdir=$OUTDIR
```
- Pruning a single layer.
```bash
python eval_layer.py --gin_binding=get_datasets.data_dir=\"${HOME}/t2t_data\" \
  --gin_binding=get_model.model_arch_name=\"small_conv\" \
  --gin_binding=prune_layer_and_eval.model_dir=\"$OUTDIR\" \
  --gin_binding=prune_layer_and_eval.l_name=\"conv_2\"
```
  Running this code would prune units of the second convolutional layer in
  increasing sparsities.

### Various Pruning Experiments
We implement various pruning techniques and strategies under load_and_prune:
- `eval_layer.py`: For pruning individual layers without fine-tuning for
  reproducing experiments in
  [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710).

- `global_prune.py`:   Loads a pretrained network and prunes one unit at a time
  following the recipe in
  [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440),
  except there is no weight decay implemented, since (1) it is not that straight
  forward to implement. (2) It would be biased towards norm based scoring
  functions.

- `prune_all_layers.py`: Pruning a network according to the method given in
  [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342).

### Deadunits Library Modules
- **pruner**: involves UnitPruner class and other pruning utilities.
  - `UnitPruner`: Pruner for keras models that supports a wide range of pruning
    methods.
  - `.probe_pruning()`: Prunes a copy of the network and calculates change in
    the loss.
  - `.prune_model_with_scores()`: Prunes the model with or without bias
    propagation.
  - `.process_layers2prune()`: Utility function for verifying pruning layers.
  - `.get_pruning_measurements()`: Function for calculating 6 different
    scoring functions efficiently along with some other metrics.
- **layers**: definition of custom tf.keras layers.
  - `TaylorScorer`: Layer that calculates the taylor approximation of unit
    saliencies efficiently.
  - `MeanReplacer`: A layer that replaces the activations with its  mean value.
  - `MaskedLayer`: Layer wrapper for masking any given layer.
- **generic_convnet**: definition of generic keras models.
  - `GenericConvnet`: Generic class used to generate various ConvNet
    architectures.
- **unitscorers**: has the scoring functions for unit pruning: `norm` and
  `random`.
- **utils**: some utilities for creating binary masks, saving objects and
  binding gin parameters.
- **train_utils**: Utilities for pruning experiments like loss function,
  pruning schedule, logging.
- **data**: Data loader for `Cifar-10`, `Imagenet-2012` and `CU-Birds200`.
- **model_defs**: Involves definition of common architectures used in
  experiments.
- **model_load**: Utilities for model creation that supports checkpoint loading
  and transfer learning.

### Disclaimer
This is not an official Google product.

