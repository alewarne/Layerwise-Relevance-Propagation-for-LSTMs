# Layerwise Relevance Propagation for LSTMs

This repository contains an implementation of the Layerwise-Relevance-Propagation (LRP) algorithm for Long-Short-Term-Memory (LSTM) neural networks in tensorflow 2.1.

## What is LRP?

LRP is an algorithm that enables a better understanding of the decisions of neural networks. The core idea is to assign a relevance value to each input feature of the neural network in a way that conserves the output score that has been assigned to a desired class. More information on LRP can be found [here](http://www.heatmapping.org).

### Why a special implementation for LSTMs?

While the LRP algorithm has been proposed for classical feed-forward networks, LSTM networks operate on sequences of inputs in a special way. The LRP implementation here is based on the paper of [Arras et al.](https://www.aclweb.org/anthology/W17-5221/) and the corresponding [numpy implementation](https://github.com/ArrasL/LRP_for_LSTM).

### Why not using the original implementation?

Since this implementation is in tensorflow 2.1 it comes with automatic GPU compatability and also enables batch-processsing of examples. This means, multiple input samples can be processed at once (and on a GPU) wihch results in large performance speedups which is especially handy when a lot of explanations have to be generated.

## Ok, what can I use it for?

Currently the implementation is limited to a network consisting of one (bi-directional) LSTM layer followed by a Dense layer. However, an extension to models with more layers is straight-forward and planned.

### My model has been built with tensorflow/keras/pytorch/... Is it supported?

Yes! This implementation is independent of the underlying model implementation and only uses the parameters of the model as input.

### How can I make my model ready for use?

To use LRP you need to provide four parameters to the constructor

* n_hidden: The number of units in the LSTM layer.
* embedding_dim: The input dimension of each token.
* n_classes: The output dimension of the network
* weights: A list of numpy arrays corresponding to the parameters of the network.

The ordering of the list for the weights parameter is of great importance. The parameters of an LSTM layer include two matrices W_x and W_h which have the shapes (embedding_dim, 4\*n_hidden) and (n_hidden, 4\* n_hidden) respectively and a bias b of shape (n_hidden,). See [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a detailed description of these parameters. The quarters of the second dimension of W_x and W_h corresponds to the gates and thus their ordering is important aswell. The natural ordering used in this repository is given by the keras way of arranging them and is given by

* First quarter: Input Gate
* Second quarter: Forget Gate
* Third quarter: Cell Gate
* Fourth quarter: Output Gate

Thus, if you have a keras model you don't have to adjust the ordering. Else, you should check which ordering your model has and can either use [this](model_transformer.py) script to permute them or change lines 45-48 in the [class](lstm_network.py) to adjust the underlying implementation.

A Bi-directional LSTM network has another set of parameters (W_x_backward, W_h_backward, b_backward) which have the same dimensions as above but process the input from the last to the first token. To this end, the weights parameter must be a list of in the order (W_x, W_h, b, W_x_backward, W_h_backward, b_backward, W_dense, b_dense) where W_dense and b_dense are the weights and bias of the final fully connected layer. You can then create your LRP model with

```python
from lstm_network import LSTM_network
model = LSTM_network(n_hidden, embedding_dim, n_classes, weights)
```

There will be a check whether the weight parameters have the correct dimensions with respect to the parameters you set but the ordering of the gates is not checked. If you have a classic LSTM layer, you can set (W_x_backward, W_h_backward, b_backward) to zero to get the same output. Likewise, if you have two biases for each W_x and W_h you can simply add them to get one bias b.

### Example

Example usage is documented in an [jupyter-notebook](example.ipynb)

## Contributing

Feel free to contribute to this repository. There is a [test](lrp_tests.py) file that contains test for both, the forward and backward pass, to be sure that both work correctly.

## Authors

* **Alexander Warnecke** [TU Braunschweig](https://www.tu-braunschweig.de/sec/team/alex)

## License

This project is licensed under the MIT License.

## Acknowledgments

* Many thanks to Leila Arras for the [initial implementation](https://github.com/ArrasL/LRP_for_LSTM) and some helpful discussions on LRP for LSTMs.
