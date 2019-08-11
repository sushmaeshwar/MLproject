import tensorflow as tf

from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.utils import ops


class GridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generates a grid of anchors at given scales and aspect ratios."""

  def __init__(self,
               scales=(0.5, 1.0, 2.0),
               aspect_ratios=(0.5, 1.0, 2.0),
               base_anchor_size=None,
               anchor_stride=None,
               anchor_offset=None):
    
    # Handle argument defaults
    if base_anchor_size is None:
      base_anchor_size = [256, 256]
    if anchor_stride is None:
      anchor_stride = [16, 16]
    if anchor_offset is None:
      anchor_offset = [0, 0]

    self._scales = scales
    self._aspect_ratios = aspect_ratios
    self._base_anchor_size = base_anchor_size
    self._anchor_stride = anchor_stride
    self._anchor_offset = anchor_offset

  def name_scope(self):
    return 'GridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the `generate` function.
    """
    return [len(self._scales) * len(self._aspect_ratios)]

  def _generate(self, feature_map_shape_list):
    
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == 1):
      raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')

    # Create constants in init_scope so they can be created in tf.functions
    # and accessed from outside of the function.
    with tf.init_scope():
      self._base_anchor_size = tf.cast(tf.convert_to_tensor(
          self._base_anchor_size), dtype=tf.float32)
      self._anchor_stride = tf.cast(tf.convert_to_tensor(
          self._anchor_stride), dtype=tf.float32)
      self._anchor_offset = tf.cast(tf.convert_to_tensor(
          self._anchor_offset), dtype=tf.float32)

    grid_height, grid_width = feature_map_shape_list[0]
    scales_grid, aspect_ratios_grid = ops.meshgrid(self._scales,
                                                   self._aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(grid_height,
                           grid_width,
                           scales_grid,
                           aspect_ratios_grid,
                           self._base_anchor_size,
                           self._anchor_stride,
                           self._anchor_offset)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is None:
      num_anchors = anchors.num_boxes()
    anchor_indices = tf.zeros([num_anchors])
    anchors.add_field('feature_map_index', anchor_indices)
    return [anchors]


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
 
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  # Get a grid of box centers
  y_centers = tf.cast(tf.range(grid_height), dtype=tf.float32)
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = ops.meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  return box_list.BoxList(bbox_corners)


def _center_size_bbox_to_corners_bbox(centers, sizes):
  
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
