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
    # just something
    T, n_hidden, n_embedding, n_classes, batch_size = 50, 300, 200, 2, 25
    orig_model = get_dummy_model(n_hidden, n_embedding, n_classes)
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size, orig_model.get_weights())
    input_keras = np.random.randn(batch_size, T, n_embedding)
    input_tf = tf.constant(input_keras)
    net.full_pass(input_tf)
    model_output = orig_model.predict(input_keras)
    net_output = net.y_hat
    res = np.allclose(net_output.numpy(), model_output, atol=1e-6)
    if res:
        print('Forward pass of model is correct.')
    else:
        diff = np.sum(np.abs(net_output.numpy()-model_output))
        print('Error in forward pass. Total abs difference : {}'.format(diff))


def test_runtime():
    T, n_hidden, n_embedding, n_classes, batch_size = 3, 300, 200, 2, 1
    n_samples = 1
    eps = 1e-3
    debug = True
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size, debug=debug)
    input = tf.constant(np.random.randn(n_samples, T, n_embedding))
    start = time.time()
    for i in tqdm(range(0, n_samples, batch_size)):
        Rx, rest = net.lrp(input[i:i+batch_size], eps=eps, bias_factor=1.0)
        print('Relevance in:', tf.reduce_sum(tf.reduce_max(net.y_hat, axis=1)).numpy())
        print('Relevance out:', tf.reduce_sum(Rx).numpy()+rest.numpy())
    end = time.time()
    total = end-start
    per_sample = total / n_samples
    print('Processing {} samples took {} s. {} seconds per sample'.format(n_samples, total, per_sample))


if __name__ == '__main__':
    #path_to_model = '../LRPForSecurity/NetworkTraining/VulDeePecker/models/keras_model_wo_metrics_w_softmax.hdf5'
    #test_equality()
    test_runtime()
