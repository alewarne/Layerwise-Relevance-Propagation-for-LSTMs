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
        self.h_backward = tf.Variable(np.zeros((batch_size, n_hidden)))
        self.c_backward = tf.Variable(np.zeros((batch_size, n_hidden)))

        self.idx_i = slice(0, self.n_hidden)
        self.idx_f = slice(self.n_hidden, 2 * self.n_hidden)
        self.idx_c = slice(2 * self.n_hidden, 3 * self.n_hidden)
        self.idx_o = slice(3 * self.n_hidden, 4 * self.n_hidden)

    # input_arr x is a numpy array of shape (batch_size, embedding_dim)
    @tf.function
    def cell_step(self, x, h, c, W_x, W_h, b):
        # forward pass
        gate_forward_x = tf.matmul(x, W_x)
        gate_forward_h = tf.matmul(h, W_h)
        gate_forward_pre = gate_forward_x + gate_forward_h + b
        print(gate_forward_pre.shape)
        gate_forward_post = tf.concat([
                            tf.sigmoid(gate_forward_pre[:,self.idx_i]), tf.sigmoid(gate_forward_pre[:, self.idx_f]),
                            tf.tanh(gate_forward_pre[:,self.idx_c]), tf.sigmoid(gate_forward_pre[:, self.idx_o]),
                            ], axis=1)
        print(gate_forward_post.shape)
        c.assign(gate_forward_post[:, self.idx_f]*c + gate_forward_post[:, self.idx_i]*gate_forward_post[:, self.idx_c])
        h.assign(gate_forward_post[:, self.idx_o] * tf.tanh(c))
        return gate_forward_pre, gate_forward_post, c, h


    @tf.function
    def one_step(self, x):
        forward = self.cell_step(x, self.h_forward, self.c_forward, self.W_x_forward, self.W_h_forward, self.b_forward)
        backward = self.cell_step(x, self.h_backward, self.c_backward, self.W_x_backward, self.W_h_backward, self.b_backward)
        return forward, backward
