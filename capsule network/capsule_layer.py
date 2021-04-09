import numpy as np
import tensorflow as tf
from config import cfg
epsilon = 1e-9

class CapsLayer(object):

    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):

        capsules = []
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                capsules = tf.contrib.layers.conv1d(input, self.num_outputs * self.vec_len, self.kernel_size, self.stride, padding="VALID",
                 activation_fn=tf.nn.relu)
                print(capsules.get_shape())
                capsules = tf.reshape(capsules, (capsules.shape[0], -1, self.vec_len,1))
                capsules = squash(capsules)

                # assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return (capsules)

        # the PrimaryCaps layer, a convolutional layer
        if self.layer_type == 'FC':
            if self.with_routing:
                print("FC.input: ",input)
                self.input = tf.reshape(input, shape=(input.shape[0], -1, 1, input.shape[-2].value, 1))
                print("FC_input dimention：",self.input.shape)
                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(
                        np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    print("b_ij:",b_IJ.shape)
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ):
    print("dynamic routing!!!!!!")
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    ###############################################
    W = tf.get_variable('Weight', shape=(1, input.shape[1], 5, 8, 8), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    # Eq.2, calc u_hat
    input = tf.tile(input, [1, 1, 5, 1, 1])
    a_i = tf.sqrt(tf.reduce_sum(tf.square(input), axis=-2, keep_dims=True))
    a_i = tf.nn.softmax(a_i, axis=2)
    print("a_i: ", a_i.shape)

    W = tf.tile(W, [cfg.batch_size,1, 1, 1, 1])

    u_hat = tf.matmul(W, input, transpose_a=True)
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    ##########################################
    # s_J = tf.reduce_mean(u_hat_stopped, axis=2, keep_dims=True)
    # b_IJ = tf.reduce_sum(tf.square(tf.subtract(u_hat_stopped, s_J)), axis=1, keep_dims=True)
    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            # #########################method 1 :leaky_softmax######################
            # leak = tf.zeros_like(b_IJ, optimize=True)
            # print("leak1",leak.shape)
            # leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
            # print("leak2",leak.shape)
            #
            # leaky_logits = tf.concat([leak, b_IJ], axis=2)
            # print("leaky_logits",leaky_logits.shape)
            #
            # leaky_routing = tf.nn.softmax(leaky_logits, axis=2)
            # print("leak_routing",leaky_routing.shape)
            # #####################
            # c_IJ = tf.split(leaky_routing, [1, 5], axis=2)[1]
            # c_IJ = c_IJ*a_i
            ##########################METHOD 2#####################################################
            # b_IJ = tf.multiply(b_IJ,a_i)
            # a_i = tf.nn.softmax(a_i,axis=2)
            # print("softmax_ai:",a_i.shape)
            c_IJ = tf.nn.softmax(b_IJ, dim=2)*a_i
            # print("C_IJ:", c_IJ.shape)
            # c_IJ = tf.multiply(c_IJ,a_i)
            # print("C_ij: ",c_IJ.shape)
            #########################METHOD3 KMEANS_ROUTING#########################################
            # b_IJ = tf.reduce_sum(tf.square(tf.subtract(u_hat_stopped, s_J)), axis=1, keep_dims=True)
            # print("循环b_IJ: ", b_IJ.shape)
            # c_IJ = tf.nn.sigmoid(b_IJ)
            # print("sigmoid_C_ij: ", c_IJ.shape)
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J, [1, input.shape[1], 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    ###########method1 squansh function
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element wise
    #############method2 strong squansh function#############
    # vec_squared_norm_a = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    # vec_squared_norm_b = tf.sqrt(vec_squared_norm_a + epsilon)
    # scalar_factor = vec_squared_norm_a/vec_squared_norm_b
    # vec_squashed = scalar_factor*vector


    return(vec_squashed)
