from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_hparams(hparams_overrides=None):
  
  hparams = tf.contrib.training.HParams(
      # Whether a fine tuning checkpoint (provided in the pipeline config)
      # should be loaded for training.
      load_pretrained=True)
  # Override any of the preceding hyperparameter values.
  if hparams_overrides:
    hparams = hparams.parse(hparams_overrides)
  return hparams
