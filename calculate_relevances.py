import numpy as np
import tensorflow as tf
import time
from lstm_network import LSTM_network
from tqdm import tqdm

if __name__ == '__main__':
    n_samples = 2500
    T = 50
    n_hidden = 300
    n_embedding = 200
    n_classes = 2
    batch_size = 25
    eps = 1e-3
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    input = tf.constant(np.random.randn(n_samples, T, n_embedding))
    start = time.time()
    for i in tqdm(range(0, n_samples, batch_size)):
        Rx, rest = net.lrp(input[i:i+batch_size], eps=eps)
    print(Rx.shape)
    print(rest.shape)
    end = time.time()
    total = end-start
    per_sample = total / n_samples
    print('Processing {} samples took {} s. {} seconds per sample'.format(n_samples, total, per_sample))
