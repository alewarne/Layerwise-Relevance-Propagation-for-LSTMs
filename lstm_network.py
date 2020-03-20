import tensorflow as tf
import numpy as np


# currently 1 bi-directional lstm layer followed by a dense layer
class LSTM_network:

    def __init__(self, n_hidden, embedding_dim, n_classes, batch_size):
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

        self.W_x_forward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
        self.W_h_forward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
        self.b_forward = tf.constant(np.random.randn(4*self.n_hidden,))

        self.W_x_backward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
        self.W_h_backward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
        self.b_backward = tf.constant(np.random.randn(4 * self.n_hidden, ))

        self.W_hidden = tf.constant(np.random.randn(n_hidden, n_classes))

        self.h_forward = tf.Variable(np.zeros((batch_size, n_hidden)))
        self.c_forward = tf.Variable(np.zeros((batch_size, n_hidden)))
        self.h_forward = tf.Variable(np.zeros((batch_size, n_hidden)))
        self.c_backward = tf.Variable(np.zeros((batch_size, n_hidden)))

        self.idx_i = slice(0, self.n_hidden)
        self.idx_f = slice(self.n_hidden, 2 * self.n_hidden)
        self.idx_c = slice(2 * self.n_hidden, 3 * self.n_hidden)
        self.idx_o = slice(3 * self.n_hidden, 4 * self.n_hidden)

    # input_arr is a numpy array of shape (batch_size, embedding_dim)
    @tf.function
    def forward(self, input_arr):
        gate_forward_x = tf.matmul(input_arr, self.W_x_forward)
        gate_forward_h = tf.matmul(self.h_forward, self.W_h_forward)
        gate_forward_pre = gate_forward_x + gate_forward_h + self.b_forward
        print(gate_forward_pre.shape)
        gate_forward_post = tf.concat([
                            tf.sigmoid(gate_forward_pre[:,self.idx_i]), tf.sigmoid(gate_forward_pre[:, self.idx_f]),
                            tf.tanh(gate_forward_pre[:,self.idx_c]), tf.sigmoid(gate_forward_pre[:, self.idx_o]),
                            ], axis=1)
        print(gate_forward_post.shape)
        self.c_forward.assign(gate_forward_post[:,self.idx_f] * self.c_forward + \
                         gate_forward_post[:, self.idx_i] * gate_forward_post[:, self.idx_c])
