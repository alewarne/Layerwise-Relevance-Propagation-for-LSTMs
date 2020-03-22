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

        self.y_hat = tf.Variable(np.zeros((self.batch_size, n_classes)))

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
        initializer = (tf.constant(np.zeros((self.batch_size, 4 * self.n_hidden))),
                  tf.constant(np.zeros((self.batch_size, 4 * self.n_hidden))),
                  tf.constant(np.zeros((self.batch_size, self.n_hidden))),
                  tf.constant(np.zeros((self.batch_size, self.n_hidden))))
        fn_fward = lambda a, x: self.one_step_fward(x)
        fn_bward = lambda a, x: self.one_step_bward(x)
        # outputs contain tesnors with (T, gates_pre, gates_post, c,h)
        o_fward = tf.scan(fn_fward, elems, initializer=initializer)
        o_bward = tf.scan(fn_bward, elems, initializer=initializer, reverse=True)
        # final prediction scores
        y_fward = tf.matmul(self.h_fward, self.W_dense_fw)
        y_bward = tf.matmul(self.h_bward, self.W_dense_bw)
        self.y_hat.assign(y_fward + y_bward)
        return o_fward, o_bward

    @tf.function
    def lrp_linear_layer(self, h_in, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - hin:            forward pass input, of shape (batch_size, D)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - hout:           forward pass output, of shape (batch_size, M) (unequal to np.dot(w.T,hin)+b if more than
                          one incoming layer!)
        - Rout:           relevance at layer output, of shape (batch_size, M)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution
                          is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore
                          bias/stabilizer redistribution (recommended)
        Returns:
        - Rin:            relevance at layer input, of shape (batch_size, D)
        """
        bias_factor_t = tf.constant(bias_factor, dtype=tf.float64)
        eps_t = tf.constant(eps, dtype=tf.float64)
        sign_out = tf.where(hout >= 0, 1., -1.)   # shape (batch_size, M)
        sign_out = tf.cast(sign_out, tf.float64)
        numerator_1 = tf.expand_dims(h_in, axis=2) * w
        numerator_2 = (bias_factor_t * (tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units)
        numerator = numerator_1 + tf.expand_dims(numerator_2, 1)
        denom = hout + (eps*sign_out)
        message = numerator / tf.expand_dims(denom, 1) * tf.expand_dims(Rout, 1)
        R_in = tf.reduce_sum(message, axis=2)
        return R_in

    @tf.function
    def lrp(self, x, y=None, eps=1e-3, bias_factor=0.0):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - x:              input array. dim = (batch_size, T, embedding_dim)
        - y:              desired output_class to explain. dim = (batch_size,)
        - eps:            eps value for lrp-eps
        - bias_factor:    bias factor for lrp
        Returns:
        - Relevances:     relevances of each input dimension. dim = (batch_size, T, embedding_dim
        """
        output_fw, output_bw = self.full_pass(x)
        print(self.y_hat)


