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


def test_forwrad_pass():
    # just something
    T, n_hidden, n_embedding, n_classes, batch_size, total = 50, 300, 200, 2, 20, 50
    orig_model = get_dummy_model(n_hidden, n_embedding, n_classes)
    net = LSTM_network(n_hidden, n_embedding, n_classes, orig_model.get_weights())
    input_keras = np.random.randn(total, T, n_embedding)
    net_output = np.vstack([net.full_pass(input_keras[i:i + batch_size])[0] for i in range(0, total, batch_size)])
    model_output = orig_model.predict(input_keras, batch_size=batch_size)
    res = np.allclose(net_output, model_output, atol=1e-6)
    np.set_printoptions(precision=5)
    if res:
        print('Forward pass of model is correct!')
    else:
        diff = np.sum(np.abs(net_output-model_output))
        print('Error in forward pass. Total abs difference : {}'.format(diff))


def test_lrp():
    T, n_hidden, n_embedding, n_classes, batch_size = 5, 300, 10, 2, 5
    eps = 0.
    bias_factor = 1.0
    debug = False
    np.random.seed(42)
    net = LSTM_network(n_hidden, n_embedding, n_classes, debug=debug)
    input = tf.constant(np.random.randn(batch_size, T, n_embedding))
    Rx, rest = net.lrp(input, eps=eps, bias_factor=bias_factor)
    R_in, R_out = (tf.reduce_sum(tf.reduce_max(net.y_hat, axis=1)).numpy(),
                  tf.reduce_sum(Rx).numpy() + tf.reduce_sum(rest).numpy())
    if np.isclose(R_in, R_out):
        print('LRP pass is correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))
    else:
        print('LRP pass is not correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))


def test_runtime():
    T, n_hidden, n_embedding, n_classes, batch_size = 25, 300, 200, 2, 10
    n_samples = 2500
    eps = 1e-3
    bias_factor = 0.
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    input = tf.constant(np.random.randn(n_samples, T, n_embedding))
    start = time.time()
    for i in tqdm(range(0, n_samples, batch_size)):
        net.lrp(input[i:i+batch_size], eps=eps, bias_factor=bias_factor)
    end = time.time()
    total = end-start
    per_sample = total / n_samples
    print('Processing {} samples took {} s. {} seconds per sample'.format(n_samples, total, per_sample))


if __name__ == '__main__':
    #path_to_model = '../LRPForSecurity/NetworkTraining/VulDeePecker/models/keras_model_wo_metrics_w_softmax.hdf5'
    test_forwrad_pass()
    test_lrp()
    #test_runtime()
