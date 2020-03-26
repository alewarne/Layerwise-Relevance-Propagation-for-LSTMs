import tensorflow as tf
import numpy as np


# currently 1 bi-directional lstm layer followed by a dense layer
class LSTM_network:

    def __init__(self, n_hidden, embedding_dim, n_classes, batch_size, weights=None, debug=False):
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.debug = debug

        # model parameters
        if weights is not None:
            self.check_weights(weights)
            self.W_x_fward = tf.constant(weights[0], dtype=tf.float64)
            self.W_h_fward = tf.constant(weights[1], dtype=tf.float64)
            self.b_fward = tf.constant(weights[2], dtype=tf.float64)

            self.W_x_bward = tf.constant(weights[3], dtype=tf.float64)
            self.W_h_bward = tf.constant(weights[4], dtype=tf.float64)
            self.b_bward = tf.constant(weights[5], dtype=tf.float64)

            self.W_dense_fw = tf.constant(weights[6][:self.n_hidden], dtype=tf.float64)
            self.W_dense_bw = tf.constant(weights[6][self.n_hidden:], dtype=tf.float64)
            self.b_dense = tf.constant(np.zeros(n_classes))
        else:
            self.W_x_fward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
            self.W_h_fward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
            self.b_fward = tf.constant(np.random.randn(4*self.n_hidden,))

            self.W_x_bward = tf.constant(np.random.randn(self.embedding_dim, 4 * self.n_hidden))
            self.W_h_bward = tf.constant(np.random.randn(self.n_hidden, 4 * self.n_hidden))
            self.b_bward = tf.constant(np.random.randn(4 * self.n_hidden, ))

            self.W_dense_fw = tf.constant(np.random.randn(n_hidden, n_classes))
            self.W_dense_bw = tf.constant(np.random.randn(n_hidden, n_classes))
            self.b_dense = tf.constant(np.random.randn(n_classes))

        # the intermediate states we have to remember in order to use LRP
        self.h_fward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.c_fward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.h_bward = tf.Variable(np.zeros((self.batch_size, n_hidden)))
        self.c_bward = tf.Variable(np.zeros((self.batch_size, n_hidden)))

        # prediction of the net
        self.y_hat = tf.Variable(np.zeros((self.batch_size, n_classes)))

        # the following order is from keras. You might have to adjust it if you use different frameworks
        self.idx_i = slice(0, self.n_hidden)
        self.idx_f = slice(self.n_hidden, 2 * self.n_hidden)
        self.idx_c = slice(2 * self.n_hidden, 3 * self.n_hidden)
        self.idx_o = slice(3 * self.n_hidden, 4 * self.n_hidden)

    def check_weights(self, weights):
        assert weights[0].shape == weights[3].shape == (self.embedding_dim, 4 * self.n_hidden)
        assert weights[1].shape == weights[4].shape == (self.n_hidden, 4 * self.n_hidden)
        assert weights[2].shape == weights[5].shape == (4 * self.n_hidden, )
        assert weights[6].shape == (2 * self.n_hidden, self.n_classes)

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def cell_step(self, x, h_old, c_old, W_x, W_h, b):
        # fward pass
        gate_x = tf.matmul(x, W_x)
        gate_h = tf.matmul(h_old, W_h)
        gate_pre = gate_x + gate_h + b
        gate_post = tf.concat([
                            tf.sigmoid(gate_pre[:, self.idx_i]), tf.sigmoid(gate_pre[:, self.idx_f]),
                            tf.tanh(gate_pre[:, self.idx_c]), tf.sigmoid(gate_pre[:, self.idx_o]),
                            ], axis=1)
        c_new = gate_post[:, self.idx_f] * c_old + gate_post[:, self.idx_i] * gate_post[:, self.idx_c]
        h_new = gate_post[:, self.idx_o] * tf.tanh(c_new)
        return gate_pre, gate_post, c_new, h_new

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def one_step_fward(self, x, h_old_fw, c_old_fw):
        fward = self.cell_step(x, h_old_fw, c_old_fw, self.W_x_fward, self.W_h_fward, self.b_fward)
        return fward

    # x_rev is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def one_step_bward(self, x_rev, h_old_bw, c_old_bw):
        bward = self.cell_step(x_rev, h_old_bw, c_old_bw, self.W_x_bward, self.W_h_bward, self.b_bward)
        return bward

    # input is full batch (batch_size, T, embedding_dim)
    @tf.function
    def full_pass(self, x):
        # we have to reorder the input since tf.scan scans the input along the first axis
        elems = tf.transpose(x, perm=[1,0,2])
        initializer = (tf.constant(np.zeros((self.batch_size, 4 * self.n_hidden))),  # gates_pre
                       tf.constant(np.zeros((self.batch_size, 4 * self.n_hidden))),  # gates_post
                       tf.constant(np.zeros((self.batch_size, self.n_hidden))),      # c_t
                       tf.constant(np.zeros((self.batch_size, self.n_hidden))))      # h_t
        fn_fward = lambda a, x: self.one_step_fward(x, a[3], a[2])
        fn_bward = lambda a, x: self.one_step_bward(x, a[3], a[2])
        # outputs contain tesnors with (T, gates_pre, gates_post, c,h)
        o_fward = tf.scan(fn_fward, elems, initializer=initializer)
        o_bward = tf.scan(fn_bward, elems, initializer=initializer, reverse=True)
        # update cell state and h
        self.c_fward.assign(o_fward[2][-1])
        self.h_fward.assign(o_fward[3][-1])
        # careful, when calling tf.scan with 'reverse=True', the last computation result is stored at the first index
        self.c_bward.assign(o_bward[2][0])
        self.h_bward.assign(o_bward[3][0])
        # final prediction scores
        y_fward = tf.matmul(self.h_fward, self.W_dense_fw)
        y_bward = tf.matmul(self.h_bward, self.W_dense_bw)
        self.y_hat.assign(y_fward + y_bward + self.b_dense)
        return o_fward, o_bward

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
        sign_out = tf.cast(tf.where(hout >= 0, 1., -1.), tf.float64)   # shape (batch_size, M)
        numerator_1 = tf.expand_dims(h_in, axis=2) * w
        numerator_2 = (bias_factor_t * tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        numerator = numerator_1 + tf.expand_dims(numerator_2, 1)
        denom = hout + (eps*sign_out)
        message = numerator / tf.expand_dims(denom, 1) * tf.expand_dims(Rout, 1)
        R_in = tf.reduce_sum(message, axis=2)
        return R_in

    def lrp(self, x, y=None, eps=1e-3, bias_factor=0.0):
        """
        LRP for a batch of samples x.
        Args:
        - x:              input array. dim = (batch_size, T, embedding_dim)
        - y:              desired output_class to explain. dim = (batch_size,)
        - eps:            eps value for lrp-eps
        - bias_factor:    bias factor for lrp
        Returns:
        - Relevances:     relevances of each input dimension. dim = (batch_size, T, embedding_dim
        """
        self.T = x.shape[1]

        lrp_pass = self.lrp_lstm(x,y,eps, bias_factor)
        # add forward and backward relevances of x (revert x_rev)
        Rx_ = lrp_pass[2] + tf.reverse(lrp_pass[5], axis=[0])
        Rx = tf.transpose(Rx_, perm=(1,0,2))  # put batch dimension to first dim again
        # remaining relevance is sum of last entry of Rh and Rc
        rest = tf.reduce_sum(lrp_pass[0][-1] + lrp_pass[1][-1] + lrp_pass[3][-1] + lrp_pass[4][-1], axis=1)
        return Rx, rest

    @tf.function
    def lrp_lstm(self, x, y=None, eps=1e-3, bias_factor=0.0):
        x_rev = tf.reverse(x, axis=[1])
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
        # c and h have one timestep more than x (the initial one, we have to add these zeros manually)
        zero_block = tf.constant(np.zeros((1, self.batch_size, self.n_hidden)))
        c_fw = tf.concat([c_fw, zero_block], axis=0)
        h_fw = tf.concat([h_fw, zero_block], axis=0)
        gates_pre_bw = tf.reverse(gates_pre_bw, [0])
        gates_post_bw = tf.reverse(gates_post_bw, [0])
        c_bw = tf.reverse(c_bw, [0])
        h_bw = tf.reverse(h_bw, [0])
        c_bw = tf.concat([c_bw, zero_block], axis=0)
        h_bw = tf.concat([h_bw, zero_block], axis=0)

        # first calculate relevaces from final linear layer
        Rh_fw_T = self.lrp_linear_layer(h_fw[self.T - 1], self.W_dense_fw, 0.5 * self.b_dense,
                                       self.y_hat, R_T, self.n_hidden, eps, bias_factor)
        Rh_bw_T = self.lrp_linear_layer(h_bw[self.T -1 ], self.W_dense_bw, 0.5 * self.b_dense,
                                       self.y_hat, R_T, self.n_hidden, eps, bias_factor)
        if self.debug:
            tf.print('Dense: Input relevance', tf.reduce_sum(R_T, axis=1))
            tf.print('Dense: Output relevance', tf.reduce_sum(Rh_fw_T+Rh_bw_T, axis=1))
        elems = np.arange(self.T-1, -1, -1)
        initializer = (
                       Rh_fw_T,                                                             # R_h_fw
                       Rh_fw_T,                                                             # R_c_fw
                       tf.constant(np.zeros((self.batch_size, self.embedding_dim))),        # R_x_fw
                       Rh_bw_T,                                                             # R_h_bw
                       Rh_bw_T,                                                             # R_c_bw
                       tf.constant(np.zeros((self.batch_size, self.embedding_dim)))         # R_x_bw
                       )
        eye = tf.eye(self.n_hidden, dtype=tf.float64)
        zeros_hidden = tf.constant(np.zeros((self.n_hidden)))

        @tf.function
        def update(input_tuple, t):
            # t starts with T-1 ; the values we want to update are essentially Rh, Rc and Rx
            # input_tuple is (R_h_fw_t+1, R_c_fw_t+1, R_x_fw_t+1, R_h_bw_t+1, R_h_bw_t+1, R_x_bw_t+1)
            #forward
            Rc_fw_t = self.lrp_linear_layer(gates_post_fw[t, :, self.idx_f] * c_fw[t-1, :], eye, zeros_hidden,
                                               c_fw[t, :],  input_tuple[1], self.n_hidden, eps, bias_factor)
            R_g_fw = self.lrp_linear_layer(gates_post_fw[t,:,self.idx_i] * gates_post_fw[t,:,self.idx_c], eye,
                                        zeros_hidden, c_fw[t, :], input_tuple[1], self.n_hidden, eps, bias_factor)
            if self.debug:
                tf.print('Fw1: Input relevance', tf.reduce_sum(input_tuple[1], axis=1))
                tf.print('Fw1: Output relevance', tf.reduce_sum(Rc_fw_t + R_g_fw, axis=1))
            Rx_t = self.lrp_linear_layer(x[:,t], self.W_x_fward[:, self.idx_c], 0.5 * self.b_fward[self.idx_c],
                                         gates_pre_fw[t, :, self.idx_c], R_g_fw, self.embedding_dim,
                                         eps, bias_factor)
            Rh_fw_t = self.lrp_linear_layer(h_fw[t-1, :], self.W_h_fward[:, self.idx_c], 0.5 * self.b_fward[self.idx_c],
                                            gates_pre_fw[t, :, self.idx_c], R_g_fw, self.n_hidden,
                                            eps, bias_factor
                                            )
            if self.debug:
                tf.print('Fw2: Input relevance', tf.reduce_sum(R_g_fw, axis=1))
                tf.print('Fw2: Output relevance', tf.reduce_sum(Rx_t,axis=1)+tf.reduce_sum(Rh_fw_t, axis=1))
            if t != 0:
                Rc_fw_t += Rh_fw_t
            #backward
            Rc_bw_t = self.lrp_linear_layer(gates_post_bw[t, :, self.idx_f] * c_bw[t-1, :], eye, zeros_hidden,
                                            c_bw[t, :], input_tuple[4], self.n_hidden, eps, bias_factor)
            R_g_bw = self.lrp_linear_layer(gates_post_bw[t, :, self.idx_i] * gates_post_bw[t, :, self.idx_c], eye,
                                           zeros_hidden, c_bw[t,:], input_tuple[4], self.n_hidden, eps, bias_factor)
            if self.debug:
                tf.print('Bw1: Input relevance', tf.reduce_sum(input_tuple[4], axis=1))
                tf.print('Bw1: Output relevance', tf.reduce_sum(Rc_bw_t + R_g_bw, axis=1))
            Rx_rev_t = self.lrp_linear_layer(x_rev[:, t], self.W_x_bward[:, self.idx_c], 0.5 * self.b_bward[self.idx_c],
                                            gates_pre_bw[t, :, self.idx_c], R_g_bw, self.embedding_dim,
                                            eps, bias_factor)
            Rh_bw_t = self.lrp_linear_layer(h_bw[t-1, :], self.W_h_bward[:, self.idx_c], 0.5 * self.b_bward[self.idx_c],
                                            gates_pre_bw[t, :, self.idx_c], R_g_bw, self.n_hidden,
                                            eps, bias_factor
                                            )
            if self.debug:
                tf.print('Bw2: Input relevance', tf.reduce_sum(R_g_bw, axis=1))
                tf.print('Bw2: Output relevance', tf.reduce_sum(Rx_rev_t,axis=1)+tf.reduce_sum(Rh_bw_t, axis=1))
            if t != 0:
                Rc_bw_t += Rh_bw_t
            return Rh_fw_t, Rc_fw_t, Rx_t, Rh_bw_t, Rc_bw_t, Rx_rev_t

        lrp_pass = tf.scan(update, elems, initializer)
        return lrp_pass
