import tensorflow as tf
import numpy as np
from config import cfg
from Makeing_dataset import load_dataset
# from utils import get_batch_data
from capsule_layer import CapsLayer
from Makeing_dataset import get_batch_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
from spactial import get_spacial_batch_data
from one import get_one_batch_data
epsilon = 1e-9


class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                ###############################
                self.X, self.labels = get_batch_data(cfg.batch_size, cfg.num_threads)
                # self.X, self.labels = get_spacial_batch_data(cfg.batch_size, cfg.num_threads)
                # self.X, self.labels = get_one_batch_data(cfg.batch_size, cfg.num_threads)

                self.Y_pre = []
                ##################################
                self.Y = tf.one_hot(self.labels, depth=5, axis=1, dtype=tf.float32)
                # self.Y = tf.one_hot(self.labels, depth=2, axis=1, dtype=tf.float32)
                self.build_arch()
                self.loss()
                self._summary()
                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            else:
                ###########################input nodes###########
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 300, 1))#
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 5, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):

        # 第一层1D卷积
        with tf.variable_scope('Conv1_layer'):


            conv1 = tf.contrib.layers.conv1d(self.X, num_outputs=32,
                                             kernel_size=21, stride=1,
                                             padding='VALID',activation_fn=tf.nn.relu)

            # building lstm
            ###################input nodes#################################
            cells = tf.contrib.rnn.LSTMCell(32, input_shape=(5,300))#
            lstm, state = tf.nn.dynamic_rnn(cells, self.X, dtype=tf.float32)

            fe_input = tf.concat([conv1, lstm], 1)
            # method 2
            # cells = tf.contrib.rnn.LSTMCell(16, input_shape=(5, 300), forget_bias=1.0)
            # (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cells, cells, time_major=False, inputs=self.X, dtype=tf.float32)
            # lstm_output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

            # fe_input = tf.concat([conv1, lstm_output], 1)
            # tf.random.shuffle(fe_input)

        # Primary Capsules layer,
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=8, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(fe_input, kernel_size=21, stride=2)

        with tf.variable_scope('DigitCaps_layer'):
            #############################
            digitCaps = CapsLayer(num_outputs=5, vec_len=8, with_routing=True, layer_type='FC')
            # digitCaps = CapsLayer(num_outputs=2, vec_len=8, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)
            print("digicCaps:", self.caps2.get_shape())

        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            print("do masking")
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True) + epsilon)
            print("v_length", self.v_length.get_shape()) #[128,5,1,1]
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            print("softmax_v shape: ", self.softmax_v.get_shape()) #[128,5,1,1]
            # assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 5 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            print("self.argmax_idx shape",self.argmax_idx.get_shape()) #[128,1,1]
            # assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))
            print("self.argmax_idx shape", self.argmax_idx.get_shape())
            # tf.print(self.argmax_idx)

            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                print("method 1")
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 8, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                print("self.masked_v shape:",self.masked_v.get_shape())
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 8, 1]
            # Method 2. masking with true label, default mode
            else:
                print("method 2")
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                ###########################################
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 5, 1)))
                # self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 2, 1)))

                print("masked_v shape:",self.masked_v.get_shape)
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)
                print("v_length shape:",self.v_length.get_shape)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            print("Reconstructe")
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            print("fc1 shape: ", fc1.get_shape())
            # assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            print("fc2 shape: ", fc2.get_shape())
            # assert fc2.get_shape() == [cfg.batch_size, 1024]
            #############################input nodes#######################
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=300, activation_fn=tf.sigmoid)#
            print("self.decoded shape: ", self.decoded.get_shape())
    def loss(self):
        # 1. The margin loss
        # max_l = max(0, m_plus-||v_c||)^2
        print("loss")

        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        print("max_l shape: ", max_l.get_shape())
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))

        print("max_r shape: ", max_r.get_shape())


        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))


        T_c = self.Y

        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        print("L_c shape", L_c.shape)
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        print("margin_loss",self.margin_loss)
        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        print("origin shape: ", orgin.get_shape())
        squared = tf.square(self.decoded - orgin)
        print("squared: ", squared.get_shape())
        self.reconstruction_err = tf.reduce_mean(squared)
        print("reconstruction loss:",self.reconstruction_err)
        #
        # # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*300
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err
        print("loss ending")


    # Summary
    def _summary(self):
        print("summary")
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_losspl', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)
        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        print(self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

