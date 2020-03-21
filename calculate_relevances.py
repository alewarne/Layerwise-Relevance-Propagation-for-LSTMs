import numpy as np
import tensorflow as tf
from lstm_network import LSTM_network

if __name__ == '__main__':
    T = 5
    n_hidden = 3
    n_embedding = 10
    n_classes = 2
    batch_size = 2
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    print(net.c_forward)
    x = tf.constant(np.random.randn(batch_size,n_embedding))
    input = tf.constant(np.random.randn(batch_size, T, n_embedding))
    input_rev = tf.reverse(input, [1])
    f, b = net.one_step(x)
