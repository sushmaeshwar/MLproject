import tensorflow as tf

from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers


def init_state(inputs,
               state_shape,
               state_initializer=tf.zeros_initializer(),
               dtype=tf.float32):
  
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
  initial_state = state_initializer(
      [inferred_batch_size] + state_shape, dtype=dtype)
  return initial_state


@add_arg_scope
def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=5,
                         forget_bias=1.0,
                         scope=None,
                         reuse=None):
  
  spatial_size = inputs.get_shape()[1:3]
  if state is None:
    state = init_state(inputs, list(spatial_size) + [2 * num_channels])
  with tf.variable_scope(scope,
                         'BasicConvLstmCell',
                         [inputs, state],
                         reuse=reuse):
    inputs.get_shape().assert_has_rank(4)
    state.get_shape().assert_has_rank(4)
    c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
    inputs_h = tf.concat(axis=3, values=[inputs, h])
    # Parameters of gates are concatenated into one conv for efficiency.
    i_j_f_o = layers.conv2d(inputs_h,
                            4 * num_channels, [filter_size, filter_size],
                            stride=1,
                            activation_fn=None,
                            scope='Gates')

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

    new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(axis=3, values=[new_c, new_h])



