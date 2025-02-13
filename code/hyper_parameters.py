'''
This file defines all hyper-parameters regarding training
'''
import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS
# Hyper-parameters about saving path and data loading path
tf.compat.v1.app.flags.DEFINE_string('version', 'exp1', '''version number of this experiment''')

tf.compat.v1.app.flags.DEFINE_string('train_path', '../data/train_modified2.csv', '''path to the train image
list csv''')
tf.compat.v1.app.flags.DEFINE_string('vali_path', '../data/vali_modified2.csv', '''path to the validation
image list csv''')
tf.compat.v1.app.flags.DEFINE_string('test_path', '../data/test_modified2.csv', '''path to the test image list
csv''')
tf.compat.v1.app.flags.DEFINE_string('fc_path', '../data/downloaded_test_fc.csv', '''path to save the feature
layer values of the test data''')
# tf.compat.v1.app.flags.DEFINE_string('test_ckpt_path', 'cache/logs_v3_9/min_model.ckpt-27280',
#                            '''checkpoint to load when testing''')
# tf.compat.v1.app.flags.DEFINE_string('ckpt_path', 'logs_v3_10/model.ckpt-59999',
#                            '''checkpoint to load when continue training''')

tf.compat.v1.app.flags.DEFINE_string('test_ckpt_path', 'logs_exp1/min_model.ckpt-21050',
                           '''checkpoint to load when testing''')
tf.compat.v1.app.flags.DEFINE_string('ckpt_path', 'logs_exp1/model.ckpt-44999',
                           '''checkpoint to load when continue training''')

## Hyper-paramters about training
tf.compat.v1.app.flags.DEFINE_float('weight_decay', 0.00025, '''scale for l2 regularization''')
tf.compat.v1.app.flags.DEFINE_float('fc_weight_decay', 0.00025, '''scale for fully connected layer's l2
regularization''')
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.01, '''Learning rate''')
tf.compat.v1.app.flags.DEFINE_boolean('continue_train_ckpt', False, '''Whether to continue training from a
checkpoint''')

## Hyper-parameters about the model
tf.compat.v1.app.flags.DEFINE_integer('num_residual_blocks', 2, '''number of residual blocks in ResNet''')
tf.compat.v1.app.flags.DEFINE_boolean('is_localization', True, '''Add localization task or not''')
