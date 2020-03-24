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
        elems = tf.reshape(x, (x.shape[1], x.shape[0], x.shape[2]))
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
        self.T = x.shape[1]
        self.Rx = tf.Variable(np.zeros(x.shape))
        self.Rx_rev = tf.Variable(np.zeros(x.shape))

        self.lrp_lstm(x,y,eps, bias_factor)

    @tf.function
    def lrp_lstm(self, x, y=None, eps=1e-3, bias_factor=0.0):
        # update inner states
        output_fw, output_bw = self.full_pass(x)
        # if classes are given, use them. Else choose prediction of the network
        if y is not None:
            if not y.dtype is tf.int64:
                y = tf.cast(y, tf.int64)
            R_out_mask = tf.one_hot(y, depth=self.n_classes, dtype=tf.float64)
        else:
            R_out_mask = tf.one_hot(tf.argmax(self.y_hat, axis=1), depth=self.n_classes, dtype=tf.float64)
        R_T = self.y_hat * R_out_mask
        gates_pre_fw, gates_post_fw, c_fw, h_fw = output_fw
        gates_pre_bw, gates_post_bw, c_bw, h_bw = output_bw
        # first calculate relevaces from final linear layer
        Rh_fw_T = self.lrp_linear_layer(h_fw[self.T - 1], self.W_dense_fw, tf.constant(np.zeros(self.n_classes)),
                                       self.y_hat, R_T, 2 * self.n_hidden, eps, bias_factor)
        # Rh_bw_T = self.lrp_linear_layer(h_bw[self.T - 1], self.W_dense_bw, tf.constant(np.zeros(self.n_classes)),
        #                                self.y_hat, R_T, 2 * self.n_hidden, eps, bias_factor)
        elems = np.arange(self.T-1, -1, -1)
        initializer = (
                       Rh_fw_T,                # R_h_fw
                       Rh_fw_T,                # R_c_fw
                       tf.constant(np.zeros((self.batch_size, self.embedding_dim))),     # R_x_fw
                       # tf.constant(Rh_bw_T),                # R_h_bw
                       # tf.constant(Rh_bw_T),                # R_c_bw
                       # tf.constant(np.zersos(x.shape))      # R_x_fw
                       )

        # t = 0
        # input_tuple = initializer
        @tf.function
        def update(input_tuple, t):
            # t starts with T-1 ; the values we want to update are essentially Rh, Rc and Rx
            # input_tuple is (R_h_fw_t+1, R_c_fw_t+1, R_x_fw_t+1, R_h_bw_t+1, R_h_bw_t+1, R_x_bw_t+1)
            Rc_fw_t = self.lrp_linear_layer(gates_post_fw[t, :, self.idx_f] * c_fw[t - 1, :],
                                               tf.eye(self.n_hidden, dtype=tf.float64), tf.constant(np.zeros((self.n_hidden))),
                                               c_fw[t, :],  input_tuple[1], 2 * self.n_hidden, eps, bias_factor)
            R_g_fw = self.lrp_linear_layer(gates_post_fw[t, :, self.idx_i] * gates_post_fw[t, :, self.idx_c],
                                           tf.eye(self.n_hidden, dtype=tf.float64), tf.constant(np.zeros((self.n_hidden))),
                                             c_fw[t, :], input_tuple[1], 2 * self.n_hidden, eps, bias_factor)
            Rx_t = self.lrp_linear_layer(x[:,t], self.W_x_fward[:, self.idx_c], self.b_fward[self.idx_c],
                                         gates_pre_fw[t, :, self.idx_c], R_g_fw, self.n_hidden+self.embedding_dim,
                                         eps, bias_factor)
            Rh_fw_t = self.lrp_linear_layer(h_fw[t-1, :, :], self.W_h_fward[:, self.idx_c], self.b_fward[self.idx_c],
                                            gates_pre_fw[t, :, self.idx_c], R_g_fw, self.n_hidden + self.embedding_dim,
                                            eps, bias_factor
                                            )
            Rc_fw_t += Rh_fw_T
        # print(Rh_fw_t.shape)
        # print(Rc_fw_t.shape)
        # print(Rx_t.shape)
            return (Rh_fw_t, Rc_fw_t, Rx_t)

        lrp_pass = tf.scan(update, elems, initializer)
