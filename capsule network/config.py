import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 25, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.225, 'regularization coefficient for reconstruction loss, default to 0.0005*450=0.392')



flags.DEFINE_boolean('is_training',False, 'train or predict phase')
flags.DEFINE_integer('num_threads', 1, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('cnnlstmdir', 'cnnlstmdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 50, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')



cfg = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
