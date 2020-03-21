import numpy as np
import tensorflow as tf
from lstm_network import LSTM_network

if __name__ == '__main__':
    T = 20
    n_hidden = 30
    n_embedding = 10
    n_classes = 2
    batch_size = 3
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    x = tf.constant(np.random.randn(batch_size,n_embedding))
    input = tf.constant(np.random.randn(batch_size, T, n_embedding))
    f_b = net.one_step(x)
    output = net.forward_pass(input)
