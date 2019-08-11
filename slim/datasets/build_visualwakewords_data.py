

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets import build_visualwakewords_data_lib

flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_float(
    'small_object_area_threshold', 0.005,
    'Threshold of fraction of image area below which small'
    'objects are filtered')
tf.flags.DEFINE_string(
    'foreground_class_of_interest', 'person',
    'Build a binary classifier based on the presence or absence'
    'of this object in the scene (default is person/not-person)')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # Path to COCO dataset images and annotations
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  visualwakewords_annotations_train = os.path.join(
      FLAGS.output_dir, 'instances_visualwakewords_train2014.json')
  visualwakewords_annotations_val = os.path.join(
      FLAGS.output_dir, 'instances_visualwakewords_val2014.json')
  visualwakewords_labels_filename = os.path.join(FLAGS.output_dir,
                                                 'labels.txt')
  small_object_area_threshold = FLAGS.small_object_area_threshold
  foreground_class_of_interest = FLAGS.foreground_class_of_interest
  # Create the Visual WakeWords annotations from COCO annotations
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  build_visualwakewords_data_lib.create_visual_wakeword_annotations(
      FLAGS.train_annotations_file, visualwakewords_annotations_train,
      small_object_area_threshold, foreground_class_of_interest,
      visualwakewords_labels_filename)
  build_visualwakewords_data_lib.create_visual_wakeword_annotations(
      FLAGS.val_annotations_file, visualwakewords_annotations_val,
      small_object_area_threshold, foreground_class_of_interest,
      visualwakewords_labels_filename)

  # Create the TF Records for Visual WakeWords Dataset
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
  build_visualwakewords_data_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_train,
      FLAGS.train_image_dir,
      train_output_path,
      num_shards=100)
  build_visualwakewords_data_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_val,
      FLAGS.val_image_dir,
      val_output_path,
      num_shards=10)


if __name__ == '__main__':
  tf.app.run()
