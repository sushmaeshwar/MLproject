
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Invalid usage\n'
          'usage: preprocess_imagenet_validation_data.py '
          '<validation data dir> <validation labels file>')
    sys.exit(-1)
  data_dir = sys.argv[1]
  validation_labels_file = sys.argv[2]

  # Read in the 50000 synsets associated with the validation data set.
  labels = [l.strip() for l in open(validation_labels_file).readlines()]
  unique_labels = set(labels)

  # Make all sub-directories in the validation data dir.
  for label in unique_labels:
    labeled_data_dir = os.path.join(data_dir, label)
    os.makedirs(labeled_data_dir)

  # Move all of the image to the appropriate sub-directory.
  for i in xrange(len(labels)):
    basename = 'ILSVRC2012_val_000%.5d.JPEG' % (i + 1)
    original_filename = os.path.join(data_dir, basename)
    if not os.path.exists(original_filename):
      print('Failed to find: ', original_filename)
      sys.exit(-1)
    new_filename = os.path.join(data_dir, labels[i], basename)
    os.rename(original_filename, new_filename)
