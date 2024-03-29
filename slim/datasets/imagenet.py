
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

_NUM_CLASSES = 1001

# If set to false, will not try to set label_to_names in dataset
# by reading them from labels.txt or github.
LOAD_READABLE_NAMES = True


def create_readable_names_for_imagenet_labels():
  

  # pylint: disable=g-line-too-long
  base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
  synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
  synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

  filename, _ = urllib.request.urlretrieve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  filename, _ = urllib.request.urlretrieve(synset_to_human_url)
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 21842

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names


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
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature(
          [], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/ymin': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/xmax': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/ymax': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/class/label': tf.VarLenFeature(
          dtype=tf.int64),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
      'object/bbox': slim.tfexample_decoder.BoundingBox(
          ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
      'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if LOAD_READABLE_NAMES:
    if dataset_utils.has_labels(dataset_dir):
      labels_to_names = dataset_utils.read_label_file(dataset_dir)
    else:
      labels_to_names = create_readable_names_for_imagenet_labels()
      dataset_utils.write_label_file(labels_to_names, dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
