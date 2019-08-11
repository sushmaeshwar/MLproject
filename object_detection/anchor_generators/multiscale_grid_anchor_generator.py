from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops


class MultiscaleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers of different scale."""

  def __init__(self, min_level, max_level, anchor_scale, aspect_ratios,
               scales_per_octave, normalize_coordinates=True):
    
    self._anchor_grid_info = []
    self._aspect_ratios = aspect_ratios
    self._scales_per_octave = scales_per_octave
    self._normalize_coordinates = normalize_coordinates

    scales = [2**(float(scale) / scales_per_octave)
              for scale in range(scales_per_octave)]
    aspects = list(aspect_ratios)

    for level in range(min_level, max_level + 1):
      anchor_stride = [2**level, 2**level]
      base_anchor_size = [2**level * anchor_scale, 2**level * anchor_scale]
      self._anchor_grid_info.append({
          'level': level,
          'info': [scales, aspects, base_anchor_size, anchor_stride]
      })

  def name_scope(self):
    return 'MultiscaleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return len(self._anchor_grid_info) * [
        len(self._aspect_ratios) * self._scales_per_octave]

  def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
    """Generates a collection of bounding boxes to be used as anchors.

    Currently we require the input image shape to be statically defined.  That
    is, im_height and im_width should be integers rather than tensors.

    Args:
      feature_map_shape_list: list of pairs of convnet layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      im_height: the height of the image to generate the grid for. If both
        im_height and im_width are 1, anchors can only be generated in
        absolute coordinates.
      im_width: the width of the image to generate the grid for. If both
        im_height and im_width are 1, anchors can only be generated in
        absolute coordinates.

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.
    Raises:
      ValueError: if im_height and im_width are not integers.
      ValueError: if im_height and im_width are 1, but normalized coordinates
        were requested.
    """
    anchor_grid_list = []
    for feat_shape, grid_info in zip(feature_map_shape_list,
                                     self._anchor_grid_info):
      # TODO(rathodv) check the feature_map_shape_list is consistent with
      # self._anchor_grid_info
      level = grid_info['level']
      stride = 2**level
      scales, aspect_ratios, base_anchor_size, anchor_stride = grid_info['info']
      feat_h = feat_shape[0]
      feat_w = feat_shape[1]
      anchor_offset = [0, 0]
      if isinstance(im_height, int) and isinstance(im_width, int):
        if im_height % 2.0**level == 0 or im_height == 1:
          anchor_offset[0] = stride / 2.0
        if im_width % 2.0**level == 0 or im_width == 1:
          anchor_offset[1] = stride / 2.0
      ag = grid_anchor_generator.GridAnchorGenerator(
          scales,
          aspect_ratios,
          base_anchor_size=base_anchor_size,
          anchor_stride=anchor_stride,
          anchor_offset=anchor_offset)
      (anchor_grid,) = ag.generate(feature_map_shape_list=[(feat_h, feat_w)])

      if self._normalize_coordinates:
        if im_height == 1 or im_width == 1:
          raise ValueError(
              'Normalized coordinates were requested upon construction of the '
              'MultiscaleGridAnchorGenerator, but a subsequent call to '
              'generate did not supply dimension information.')
        anchor_grid = box_list_ops.to_normalized_coordinates(
            anchor_grid, im_height, im_width, check_range=False)
      anchor_grid_list.append(anchor_grid)

    return anchor_grid_list
