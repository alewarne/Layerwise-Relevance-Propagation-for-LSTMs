import numpy as np
import tensorflow as tf
from lstm_network import LSTM_network

if __name__ == '__main__':
    T = 5
    n_hidden = 3
    n_embedding = 2
    n_classes = 2
    batch_size = 4
    eps = 1e-3
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    x = tf.constant(np.random.randn(batch_size,n_embedding))
    y = np.zeros((batch_size,))
    input = tf.constant(np.random.randn(batch_size, T, n_embedding))
    #output = net.full_pass(input)
    net.lrp(input, y, eps)
