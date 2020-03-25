import numpy as np
import tensorflow as tf
import time
from lstm_network import LSTM_network
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense


def get_dummy_model(units, embedding_dim, n_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(units), input_shape=(None, embedding_dim)))
    model.add(Dense(n_classes))
    return model


def test_equality():
    orig_model = get_dummy_model(n_hidden, n_embedding, n_classes)
    weights = orig_model.get_weights()
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size, weights)
    input_keras = np.random.randn(batch_size, T, n_embedding)
    input_tf = tf.constant(input_keras)
    net.full_pass(input_tf)
    model_output = orig_model.predict(input_keras)
    net_output = net.y_hat
    return np.allclose(net_output.numpy(), model_output, atol=1e-7)


def test_time():
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    input = tf.constant(np.random.randn(n_samples, T, n_embedding))
    start = time.time()
    for i in tqdm(range(0, n_samples, batch_size)):
        Rx, rest = net.lrp(input[i:i+batch_size], eps=eps)
    end = time.time()
    total = end-start
    per_sample = total / n_samples
    print('Processing {} samples took {} s. {} seconds per sample'.format(n_samples, total, per_sample))


if __name__ == '__main__':
    n_samples = 2500
    T = 50
    n_hidden = 300
    n_embedding = 200
    n_classes = 2
    batch_size = 25
    eps = 1e-3
    #path_to_model = '../LRPForSecurity/NetworkTraining/VulDeePecker/models/keras_model_wo_metrics_w_softmax.hdf5'
    if test_equality():
        print('Forward pass of model is correct.')
    test_time()
