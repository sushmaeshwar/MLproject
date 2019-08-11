import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops


class MultipleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               box_specs_list,
               base_anchor_size=None,
               anchor_strides=None,
               anchor_offsets=None,
               clip_window=None):
   
    if isinstance(box_specs_list, list) and all(
        [isinstance(list_item, list) for list_item in box_specs_list]):
      self._box_specs = box_specs_list
    else:
      raise ValueError('box_specs_list is expected to be a '
                       'list of lists of pairs')
    if base_anchor_size is None:
      base_anchor_size = [256, 256]
    self._base_anchor_size = base_anchor_size
    self._anchor_strides = anchor_strides
    self._anchor_offsets = anchor_offsets
    if clip_window is not None and clip_window.get_shape().as_list() != [4]:
      raise ValueError('clip_window must either be None or a shape [4] tensor')
    self._clip_window = clip_window
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      if not all([isinstance(entry, tuple) and len(entry) == 2
                  for entry in box_spec]):
        raise ValueError('box_specs_list is expected to be a '
                         'list of lists of pairs')
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

    for arg, arg_name in zip([self._anchor_strides, self._anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if arg and not (isinstance(arg, list) and
                      len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if arg and not all([
          isinstance(list_item, tuple) and len(list_item) == 2
          for list_item in arg
      ]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

  def name_scope(self):
    return 'MultipleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return [len(box_specs) for box_specs in self._box_specs]

  def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
    
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == len(self._box_specs)):
      raise ValueError('feature_map_shape_list must be a list with the same '
                       'length as self._box_specs')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')

    im_height = tf.cast(im_height, dtype=tf.float32)
    im_width = tf.cast(im_width, dtype=tf.float32)

    if not self._anchor_strides:
      anchor_strides = [(1.0 / tf.cast(pair[0], dtype=tf.float32),
                         1.0 / tf.cast(pair[1], dtype=tf.float32))
                        for pair in feature_map_shape_list]
    else:
      anchor_strides = [(tf.cast(stride[0], dtype=tf.float32) / im_height,
                         tf.cast(stride[1], dtype=tf.float32) / im_width)
                        for stride in self._anchor_strides]
    if not self._anchor_offsets:
      anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                        for stride in anchor_strides]
    else:
      anchor_offsets = [(tf.cast(offset[0], dtype=tf.float32) / im_height,
                         tf.cast(offset[1], dtype=tf.float32) / im_width)
                        for offset in self._anchor_offsets]

    for arg, arg_name in zip([anchor_strides, anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if not all([isinstance(list_item, tuple) and len(list_item) == 2
                  for list_item in arg]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

    anchor_grid_list = []
    min_im_shape = tf.minimum(im_height, im_width)
    scale_height = min_im_shape / im_height
    scale_width = min_im_shape / im_width
    if not tf.contrib.framework.is_tensor(self._base_anchor_size):
      base_anchor_size = [
          scale_height * tf.constant(self._base_anchor_size[0],
                                     dtype=tf.float32),
          scale_width * tf.constant(self._base_anchor_size[1],
                                    dtype=tf.float32)
      ]
    else:
      base_anchor_size = [
          scale_height * self._base_anchor_size[0],
          scale_width * self._base_anchor_size[1]
      ]
    for feature_map_index, (grid_size, scales, aspect_ratios, stride,
                            offset) in enumerate(
                                zip(feature_map_shape_list, self._scales,
                                    self._aspect_ratios, anchor_strides,
                                    anchor_offsets)):
      tiled_anchors = grid_anchor_generator.tile_anchors(
          grid_height=grid_size[0],
          grid_width=grid_size[1],
          scales=scales,
          aspect_ratios=aspect_ratios,
          base_anchor_size=base_anchor_size,
          anchor_stride=stride,
          anchor_offset=offset)
      if self._clip_window is not None:
        tiled_anchors = box_list_ops.clip_to_window(
            tiled_anchors, self._clip_window, filter_nonoverlapping=False)
      num_anchors_in_layer = tiled_anchors.num_boxes_static()
      if num_anchors_in_layer is None:
        num_anchors_in_layer = tiled_anchors.num_boxes()
      anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
      tiled_anchors.add_field('feature_map_index', anchor_indices)
      anchor_grid_list.append(tiled_anchors)

    return anchor_grid_list


def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       base_anchor_size=None,
                       anchor_strides=None,
                       anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):

  if base_anchor_size is None:
    base_anchor_size = [1.0, 1.0]
  box_specs_list = []
  if scales is None or not scales:
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
  else:
    # Add 1.0 to the end, which will only be used in scale_next below and used
    # for computing an interpolated scale for the largest scale in the list.
    scales += [1.0]

  for layer, scale, scale_next in zip(
      range(num_layers), scales[:-1], scales[1:]):
    layer_box_specs = []
    if layer == 0 and reduce_boxes_in_lowest_layer:
      layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
      for aspect_ratio in aspect_ratios:
        layer_box_specs.append((scale, aspect_ratio))
      # Add one more anchor, with a scale between the current scale, and the
      # scale for the next layer, with a specified aspect ratio (1.0 by
      # default).
      if interpolated_scale_aspect_ratio > 0.0:
        layer_box_specs.append((np.sqrt(scale*scale_next),
                                interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

  return MultipleGridAnchorGenerator(box_specs_list, base_anchor_size,
                                     anchor_strides, anchor_offsets)
