import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops


class FlexibleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers of different scale."""

  def __init__(self, base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
               normalize_coordinates=True):
    
    self._base_sizes = base_sizes
    self._aspect_ratios = aspect_ratios
    self._anchor_strides = anchor_strides
    self._anchor_offsets = anchor_offsets
    self._normalize_coordinates = normalize_coordinates

  def name_scope(self):
    return 'FlexibleGridAnchorGenerator'

  def num_anchors_per_location(self):
    
    return [len(size) for size in self._base_sizes]

  def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
    
    anchor_grid_list = []
    for (feat_shape, base_sizes, aspect_ratios, anchor_stride, anchor_offset
        ) in zip(feature_map_shape_list, self._base_sizes, self._aspect_ratios,
                 self._anchor_strides, self._anchor_offsets):
      anchor_grid = grid_anchor_generator.tile_anchors(
          feat_shape[0],
          feat_shape[1],
          tf.cast(tf.convert_to_tensor(base_sizes), dtype=tf.float32),
          tf.cast(tf.convert_to_tensor(aspect_ratios), dtype=tf.float32),
          tf.constant([1.0, 1.0]),
          tf.cast(tf.convert_to_tensor(anchor_stride), dtype=tf.float32),
          tf.cast(tf.convert_to_tensor(anchor_offset), dtype=tf.float32))
      num_anchors = anchor_grid.num_boxes_static()
      if num_anchors is None:
        num_anchors = anchor_grid.num_boxes()
      anchor_indices = tf.zeros([num_anchors])
      anchor_grid.add_field('feature_map_index', anchor_indices)
      if self._normalize_coordinates:
        if im_height == 1 or im_width == 1:
          raise ValueError(
              'Normalized coordinates were requested upon construction of the '
              'FlexibleGridAnchorGenerator, but a subsequent call to '
              'generate did not supply dimension information.')
        anchor_grid = box_list_ops.to_normalized_coordinates(
            anchor_grid, im_height, im_width, check_range=False)
      anchor_grid_list.append(anchor_grid)

    return anchor_grid_list
