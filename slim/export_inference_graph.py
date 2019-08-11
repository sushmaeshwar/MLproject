

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool(
    'is_video_model', False, 'whether to use 5-D inputs for video model.')

tf.app.flags.DEFINE_integer(
    'num_frames', None,
    'The number of frames to use. Only used if is_video_model is True.')

tf.app.flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  if FLAGS.is_video_model and not FLAGS.num_frames:
    raise ValueError(
        'Number of frames must be specified for video models with --num_frames')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    if FLAGS.is_video_model:
      input_shape = [FLAGS.batch_size, FLAGS.num_frames,
                     image_size, image_size, 3]
    else:
      input_shape = [FLAGS.batch_size, image_size, image_size, 3]
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=input_shape)
    network_fn(placeholder)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
      tf.io.write_graph(
          graph_def,
          os.path.dirname(FLAGS.output_file),
          os.path.basename(FLAGS.output_file),
          as_text=True)
    else:
      with gfile.GFile(FLAGS.output_file, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
