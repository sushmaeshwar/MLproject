

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import visualwakewords

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'visualwakewords': visualwakewords,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
