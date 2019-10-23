# Train, Load and Prune

Involves 3 types of scripts

- `train.py`: Trains and dumps network with fixed
intervals.

- `global_prune.py`: Loads a pretrained network and prunes one unit
at a time following the recipe in
[`Pruning Convolutional Neural Networks for Resource Efficient Inference`](https://arxiv.org/abs/1611.06440),
Molchanov et.al.\[2017\], except there is no weight decay implemented, since (1)
it is not that straight forward to implement. (2) It would be biased towards norm based scoring functions.

- `eval_layer.py`: Prunes a single layer at different sparsity fractions.
