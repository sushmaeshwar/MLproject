
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.python.training import moving_averages

from inception.slim import losses
from inception.slim import scopes
from inception.slim import variables

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'


@scopes.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation=None,
               is_training=True,
               trainable=True,
               restore=True,
               scope=None,
               reuse=None):
  
  inputs_shape = inputs.get_shape()
  with tf.variable_scope(scope, 'BatchNorm', [inputs], reuse=reuse):
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = variables.variable('beta',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=trainable,
                                restore=restore)
    if scale:
      gamma = variables.variable('gamma',
                                 params_shape,
                                 initializer=tf.ones_initializer(),
                                 trainable=trainable,
                                 restore=restore)
    # Create moving_mean and moving_variance add them to
    # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
    moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    moving_mean = variables.variable('moving_mean',
                                     params_shape,
                                     initializer=tf.zeros_initializer(),
                                     trainable=False,
                                     restore=restore,
                                     collections=moving_collections)
    moving_variance = variables.variable('moving_variance',
                                         params_shape,
                                         initializer=tf.ones_initializer(),
                                         trainable=False,
                                         restore=restore,
                                         collections=moving_collections)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)

      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance
    # Normalize the activations.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    if activation:
      outputs = activation(outputs)
    return outputs


def _two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')


@scopes.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
  
  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    if batch_norm_params is not None:
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [num_filters_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      outputs = activation(outputs)
    return outputs


@scopes.add_arg_scope
def fc(inputs,
       num_units_out,
       activation=tf.nn.relu,
       stddev=0.01,
       bias=0.0,
       weight_decay=0,
       batch_norm_params=None,
       is_training=True,
       trainable=True,
       restore=True,
       scope=None,
       reuse=None):
  
  with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    num_units_in = inputs.get_shape()[1]
    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    if batch_norm_params is not None:
      outputs = tf.matmul(inputs, weights)
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(outputs, **batch_norm_params)
    else:
      bias_shape = [num_units_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    if activation:
      outputs = activation(outputs)
    return outputs


def one_hot_encoding(labels, num_classes, scope=None):
  
  with tf.name_scope(scope, 'OneHotEncoding', [labels]):
    batch_size = labels.get_shape()[0]
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
    concated = tf.concat(axis=1, values=[indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, num_classes]), 1.0, 0.0)
    onehot_labels.set_shape([batch_size, num_classes])
    return onehot_labels


@scopes.add_arg_scope
def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  
  with tf.name_scope(scope, 'MaxPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)


@scopes.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  
  with tf.name_scope(scope, 'AvgPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.avg_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)


@scopes.add_arg_scope
def dropout(inputs, keep_prob=0.5, is_training=True, scope=None):
  
  if is_training and keep_prob > 0:
    with tf.name_scope(scope, 'Dropout', [inputs]):
      return tf.nn.dropout(inputs, keep_prob)
  else:
    return inputs


def flatten(inputs, scope=None):
  
  if len(inputs.get_shape()) < 2:
    raise ValueError('Inputs must be have a least 2 dimensions')
  dims = inputs.get_shape()[1:]
  k = dims.num_elements()
  with tf.name_scope(scope, 'Flatten', [inputs]):
    return tf.reshape(inputs, [-1, k])


def repeat_op(repetitions, inputs, op, *args, **kwargs):
  
  scope = kwargs.pop('scope', None)
  with tf.variable_scope(scope, 'RepeatOp', [inputs]):
    tower = inputs
    for _ in range(repetitions):
      tower = op(tower, *args, **kwargs)
    return tower
