

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow as tf

layers = tf.contrib.layers


def pix2pix_arg_scope():
  
  # These parameters come from the online port, which don't necessarily match
  # those in the paper.
  # TODO(nsilberman): confirm these values with Philip.
  instance_norm_params = {
      'center': True,
      'scale': True,
      'epsilon': 0.00001,
  }

  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.conv2d_transpose],
      normalizer_fn=layers.instance_norm,
      normalizer_params=instance_norm_params,
      weights_initializer=tf.random_normal_initializer(0, 0.02)) as sc:
    return sc


def upsample(net, num_outputs, kernel_size, method='nn_upsample_conv'):
  
  net_shape = tf.shape(net)
  height = net_shape[1]
  width = net_shape[2]

  if method == 'nn_upsample_conv':
    net = tf.image.resize_nearest_neighbor(
        net, [kernel_size[0] * height, kernel_size[1] * width])
    net = layers.conv2d(net, num_outputs, [4, 4], activation_fn=None)
  elif method == 'conv2d_transpose':
    net = layers.conv2d_transpose(
        net, num_outputs, [4, 4], stride=kernel_size, activation_fn=None)
  else:
    raise ValueError('Unknown method: [%s]' % method)

  return net


class Block(
    collections.namedtuple('Block', ['num_filters', 'decoder_keep_prob'])):
  
  pass


def _default_generator_blocks():
  
  return [
      Block(64, 0.5),
      Block(128, 0.5),
      Block(256, 0.5),
      Block(512, 0),
      Block(512, 0),
      Block(512, 0),
      Block(512, 0),
  ]


def pix2pix_generator(net,
                      num_outputs,
                      blocks=None,
                      upsample_method='nn_upsample_conv',
                      is_training=False):  # pylint: disable=unused-argument
  
  end_points = {}

  blocks = blocks or _default_generator_blocks()

  input_size = net.get_shape().as_list()

  input_size[3] = num_outputs

  upsample_fn = functools.partial(upsample, method=upsample_method)

  encoder_activations = []

  ###########
  # Encoder #
  ###########
  with tf.variable_scope('encoder'):
    with tf.contrib.framework.arg_scope(
        [layers.conv2d],
        kernel_size=[4, 4],
        stride=2,
        activation_fn=tf.nn.leaky_relu):

      for block_id, block in enumerate(blocks):
        # No normalizer for the first encoder layers as per 'Image-to-Image',
        # Section 5.1.1
        if block_id == 0:
          # First layer doesn't use normalizer_fn
          net = layers.conv2d(net, block.num_filters, normalizer_fn=None)
        elif block_id < len(blocks) - 1:
          net = layers.conv2d(net, block.num_filters)
        else:
          # Last layer doesn't use activation_fn nor normalizer_fn
          net = layers.conv2d(
              net, block.num_filters, activation_fn=None, normalizer_fn=None)

        encoder_activations.append(net)
        end_points['encoder%d' % block_id] = net

  ###########
  # Decoder #
  ###########
  reversed_blocks = list(blocks)
  reversed_blocks.reverse()

  with tf.variable_scope('decoder'):
    # Dropout is used at both train and test time as per 'Image-to-Image',
    # Section 2.1 (last paragraph).
    with tf.contrib.framework.arg_scope([layers.dropout], is_training=True):

      for block_id, block in enumerate(reversed_blocks):
        if block_id > 0:
          net = tf.concat([net, encoder_activations[-block_id - 1]], axis=3)

        # The Relu comes BEFORE the upsample op:
        net = tf.nn.relu(net)
        net = upsample_fn(net, block.num_filters, [2, 2])
        if block.decoder_keep_prob > 0:
          net = layers.dropout(net, keep_prob=block.decoder_keep_prob)
        end_points['decoder%d' % block_id] = net

  with tf.variable_scope('output'):
    # Explicitly set the normalizer_fn to None to override any default value
    # that may come from an arg_scope, such as pix2pix_arg_scope.
    logits = layers.conv2d(
        net, num_outputs, [4, 4], activation_fn=None, normalizer_fn=None)
    logits = tf.reshape(logits, input_size)

    end_points['logits'] = logits
    end_points['predictions'] = tf.tanh(logits)

  return logits, end_points


def pix2pix_discriminator(net, num_filters, padding=2, pad_mode='REFLECT',
                          activation_fn=tf.nn.leaky_relu, is_training=False):
  
  del is_training
  end_points = {}

  num_layers = len(num_filters)

  def padded(net, scope):
    if padding:
      with tf.variable_scope(scope):
        spatial_pad = tf.constant(
            [[0, 0], [padding, padding], [padding, padding], [0, 0]],
            dtype=tf.int32)
        return tf.pad(net, spatial_pad, pad_mode)
    else:
      return net

  with tf.contrib.framework.arg_scope(
      [layers.conv2d],
      kernel_size=[4, 4],
      stride=2,
      padding='valid',
      activation_fn=activation_fn):

    # No normalization on the input layer.
    net = layers.conv2d(
        padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')

    end_points['conv0'] = net

    for i in range(1, num_layers - 1):
      net = layers.conv2d(
          padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)
      end_points['conv%d' % i] = net

    # Stride 1 on the last layer.
    net = layers.conv2d(
        padded(net, 'conv%d' % (num_layers - 1)),
        num_filters[-1],
        stride=1,
        scope='conv%d' % (num_layers - 1))
    end_points['conv%d' % (num_layers - 1)] = net

    # 1-dim logits, stride 1, no activation, no normalization.
    logits = layers.conv2d(
        padded(net, 'conv%d' % num_layers),
        1,
        stride=1,
        activation_fn=None,
        normalizer_fn=None,
        scope='conv%d' % num_layers)
    end_points['logits'] = logits
    end_points['predictions'] = tf.sigmoid(logits)
  return logits, end_points
