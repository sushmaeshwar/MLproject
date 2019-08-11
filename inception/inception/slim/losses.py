
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# In order to gather all losses in a network, the user should use this
# key for get_collection, i.e:
#   losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
LOSSES_COLLECTION = '_losses'


def l1_regularizer(weight=1.0, scope=None):
  
  def regularizer(tensor):
    with tf.name_scope(scope, 'L1Regularizer', [tensor]):
      l1_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l1_weight, tf.reduce_sum(tf.abs(tensor)), name='value')
  return regularizer


def l2_regularizer(weight=1.0, scope=None):
  
  def regularizer(tensor):
    with tf.name_scope(scope, 'L2Regularizer', [tensor]):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer


def l1_l2_regularizer(weight_l1=1.0, weight_l2=1.0, scope=None):
  
  def regularizer(tensor):
    with tf.name_scope(scope, 'L1L2Regularizer', [tensor]):
      weight_l1_t = tf.convert_to_tensor(weight_l1,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l1')
      weight_l2_t = tf.convert_to_tensor(weight_l2,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l2')
      reg_l1 = tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(tensor)),
                      name='value_l1')
      reg_l2 = tf.multiply(weight_l2_t, tf.nn.l2_loss(tensor),
                      name='value_l2')
      return tf.add(reg_l1, reg_l2, name='value')
  return regularizer


def l1_loss(tensor, weight=1.0, scope=None):
  
  with tf.name_scope(scope, 'L1Loss', [tensor]):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def l2_loss(tensor, weight=1.0, scope=None):
  
  with tf.name_scope(scope, 'L2Loss', [tensor]):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits(
        logits, one_hot_labels, name='xentropy')

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss
