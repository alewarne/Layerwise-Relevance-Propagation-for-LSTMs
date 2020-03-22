import tensorflow as tf
import numpy as np


# currently 1 bi-directional lstm layer followed by a dense layer
class LSTM_network:

    def __init__(self, n_hidden, embedding_dim, n_classes, batch_size):
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.W_x_fward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
        self.W_h_fward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
        self.b_fward = tf.constant(np.random.randn(4*self.n_hidden,))

        self.W_x_bward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
        self.W_h_bward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
        self.b_bward = tf.constant(np.random.randn(4 * self.n_hidden, ))

        self.W_dense_fw = tf.constant(np.random.randn(n_hidden, n_classes))
        self.W_dense_bw = tf.constant(np.random.randn(n_hidden, n_classes))

        self.h_fward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.c_fward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.h_bward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.c_bward = tf.Variable(np.zeros((self.batch_size, n_hidden)))

        self.output = tf.Variable(np.zeros((self.batch_size, n_classes)))

        self.idx_i = slice(0, self.n_hidden)
        self.idx_f = slice(self.n_hidden, 2 * self.n_hidden)
        self.idx_c = slice(2 * self.n_hidden, 3 * self.n_hidden)
        self.idx_o = slice(3 * self.n_hidden, 4 * self.n_hidden)

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def cell_step(self, x, h, c, W_x, W_h, b):
        # fward pass
        gate_x = tf.matmul(x, W_x)
        gate_h = tf.matmul(h, W_h)
        gate_pre = gate_x + gate_h + b
        gate_post = tf.concat([
                            tf.sigmoid(gate_pre[:, self.idx_i]), tf.sigmoid(gate_pre[:, self.idx_f]),
                            tf.tanh(gate_pre[:, self.idx_c]), tf.sigmoid(gate_pre[:, self.idx_o]),
                            ], axis=1)
        c.assign(gate_post[:, self.idx_f] * c + gate_post[:, self.idx_i] * gate_post[:, self.idx_c])
        h.assign(gate_post[:, self.idx_o] * tf.tanh(c))
        return gate_pre, gate_post, c, h

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def one_step_fward(self, x):
        fward = self.cell_step(x, self.h_fward, self.c_fward, self.W_x_fward, self.W_h_fward, self.b_fward)
        return fward

    # x_rev is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def one_step_bward(self, x_rev):
        bward = self.cell_step(x_rev, self.h_bward, self.c_bward, self.W_x_bward, self.W_h_bward, self.b_bward)
        return bward

    # input is full batch (batch_size, T, embedding_dim)
    @tf.function
    def full_pass(self, x):
        # we have to reshape the input since tf.scans scans the input along the first axis
        elems = tf.reshape(x, (tf.shape(x)[1], tf.shape(x)[0], tf.shape(x)[2]))
        initializer = (tf.zeros((self.batch_size, 4 * self.n_hidden), dtype=tf.float64),
                  tf.zeros((self.batch_size, 4 * self.n_hidden), dtype=tf.float64),
                  tf.zeros((self.batch_size, self.n_hidden), dtype=tf.float64),
                  tf.zeros((self.batch_size, self.n_hidden), dtype=tf.float64))
        fn_fward = lambda a, x: self.one_step_fward(x)
        fn_bward = lambda a, x: self.one_step_bward(x)
        # outputs contain tesnors with (T, gates_pre, gates_post, c,h)
        o_fward = tf.scan(fn_fward, elems, initializer=initializer)
        o_bward = tf.scan(fn_bward, elems, initializer=initializer, reverse=True)
        # final prediction scores
        y_fward = tf.matmul(self.h_fward, self.W_dense_fw)
        y_bward = tf.matmul(self.h_bward, self.W_dense_bw)
        self.output.assign(y_fward + y_bward)
        return o_fward, o_bward
