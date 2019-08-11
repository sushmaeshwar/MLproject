
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils


slim = tf.contrib.slim

_FILE_PATTERN = '%s.record-*'

_SPLITS_TO_SIZES = {
    'train': 82783,
    'validation': 40504,
}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, an integer in {0, 1}',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, all objects belong to the same class.',
}

_NUM_CLASSES = 2

# labels file
LABELS_FILENAME = 'labels.txt'


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  items_to_handlers = {
      'image':
          slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label':
          slim.tfexample_decoder.Tensor('image/class/label'),
      'object/bbox':
          slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/'),
      'object/label':
          slim.tfexample_decoder.Tensor('image/object/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)

  labels_to_names = None
  labels_file = os.path.join(dataset_dir, LABELS_FILENAME)
  if tf.gfile.Exists(labels_file):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
