from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from object_detection.builders import dataset_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils

HASH_KEY = 'hash'
HASH_BINS = 1 << 31
SERVING_FED_EXAMPLE_KEY = 'serialized_example'

# A map of names to methods that help build the input pipeline.
INPUT_BUILDER_UTIL_MAP = {
    'dataset_build': dataset_builder.build,
    'model_build': model_builder.build,
}


def _multiclass_scores_or_one_hot_labels(multiclass_scores,
                                         groundtruth_boxes,
                                         groundtruth_classes, num_classes):
  """Returns one-hot encoding of classes when multiclass_scores is empty."""
  # Replace groundtruth_classes tensor with multiclass_scores tensor when its
  # non-empty. If multiclass_scores is empty fall back on groundtruth_classes
  # tensor.
  def true_fn():
    return tf.reshape(multiclass_scores,
                      [tf.shape(groundtruth_boxes)[0], num_classes])
  def false_fn():
    return tf.one_hot(groundtruth_classes, num_classes)

  return tf.cond(tf.size(multiclass_scores) > 0, true_fn, false_fn)


def transform_input_data(tensor_dict,
                         model_preprocess_fn,
                         image_resizer_fn,
                         num_classes,
                         data_augmentation_fn=None,
                         merge_multiple_boxes=False,
                         retain_original_image=False,
                         use_multiclass_scores=False,
                         use_bfloat16=False):
  
  out_tensor_dict = tensor_dict.copy()
  if fields.InputDataFields.multiclass_scores in out_tensor_dict:
    out_tensor_dict[
        fields.InputDataFields
        .multiclass_scores] = _multiclass_scores_or_one_hot_labels(
            out_tensor_dict[fields.InputDataFields.multiclass_scores],
            out_tensor_dict[fields.InputDataFields.groundtruth_boxes],
            out_tensor_dict[fields.InputDataFields.groundtruth_classes],
            num_classes)

  if fields.InputDataFields.groundtruth_boxes in out_tensor_dict:
    out_tensor_dict = util_ops.filter_groundtruth_with_nan_box_coordinates(
        out_tensor_dict)
    out_tensor_dict = util_ops.filter_unrecognized_classes(out_tensor_dict)

  if retain_original_image:
    out_tensor_dict[fields.InputDataFields.original_image] = tf.cast(
        image_resizer_fn(out_tensor_dict[fields.InputDataFields.image],
                         None)[0], tf.uint8)

  if fields.InputDataFields.image_additional_channels in out_tensor_dict:
    channels = out_tensor_dict[fields.InputDataFields.image_additional_channels]
    out_tensor_dict[fields.InputDataFields.image] = tf.concat(
        [out_tensor_dict[fields.InputDataFields.image], channels], axis=2)

  # Apply data augmentation ops.
  if data_augmentation_fn is not None:
    out_tensor_dict = data_augmentation_fn(out_tensor_dict)

  # Apply model preprocessing ops and resize instance masks.
  image = out_tensor_dict[fields.InputDataFields.image]
  preprocessed_resized_image, true_image_shape = model_preprocess_fn(
      tf.expand_dims(tf.cast(image, dtype=tf.float32), axis=0))
  if use_bfloat16:
    preprocessed_resized_image = tf.cast(
        preprocessed_resized_image, tf.bfloat16)
  out_tensor_dict[fields.InputDataFields.image] = tf.squeeze(
      preprocessed_resized_image, axis=0)
  out_tensor_dict[fields.InputDataFields.true_image_shape] = tf.squeeze(
      true_image_shape, axis=0)
  if fields.InputDataFields.groundtruth_instance_masks in out_tensor_dict:
    masks = out_tensor_dict[fields.InputDataFields.groundtruth_instance_masks]
    _, resized_masks, _ = image_resizer_fn(image, masks)
    if use_bfloat16:
      resized_masks = tf.cast(resized_masks, tf.bfloat16)
    out_tensor_dict[
        fields.InputDataFields.groundtruth_instance_masks] = resized_masks

  label_offset = 1
  zero_indexed_groundtruth_classes = out_tensor_dict[
      fields.InputDataFields.groundtruth_classes] - label_offset
  if use_multiclass_scores:
    out_tensor_dict[
        fields.InputDataFields.groundtruth_classes] = out_tensor_dict[
            fields.InputDataFields.multiclass_scores]
  else:
    out_tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(
        zero_indexed_groundtruth_classes, num_classes)
  out_tensor_dict.pop(fields.InputDataFields.multiclass_scores, None)

  if fields.InputDataFields.groundtruth_confidences in out_tensor_dict:
    groundtruth_confidences = out_tensor_dict[
        fields.InputDataFields.groundtruth_confidences]
    # Map the confidences to the one-hot encoding of classes
    out_tensor_dict[fields.InputDataFields.groundtruth_confidences] = (
        tf.reshape(groundtruth_confidences, [-1, 1]) *
        out_tensor_dict[fields.InputDataFields.groundtruth_classes])
  else:
    groundtruth_confidences = tf.ones_like(
        zero_indexed_groundtruth_classes, dtype=tf.float32)
    out_tensor_dict[fields.InputDataFields.groundtruth_confidences] = (
        out_tensor_dict[fields.InputDataFields.groundtruth_classes])

  if merge_multiple_boxes:
    merged_boxes, merged_classes, merged_confidences, _ = (
        util_ops.merge_boxes_with_multiple_labels(
            out_tensor_dict[fields.InputDataFields.groundtruth_boxes],
            zero_indexed_groundtruth_classes,
            groundtruth_confidences,
            num_classes))
    merged_classes = tf.cast(merged_classes, tf.float32)
    out_tensor_dict[fields.InputDataFields.groundtruth_boxes] = merged_boxes
    out_tensor_dict[fields.InputDataFields.groundtruth_classes] = merged_classes
    out_tensor_dict[fields.InputDataFields.groundtruth_confidences] = (
        merged_confidences)
  if fields.InputDataFields.groundtruth_boxes in out_tensor_dict:
    out_tensor_dict[fields.InputDataFields.num_groundtruth_boxes] = tf.shape(
        out_tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]

  return out_tensor_dict


def pad_input_data_to_static_shapes(tensor_dict, max_num_boxes, num_classes,
                                    spatial_image_shape=None):
  

  if not spatial_image_shape or spatial_image_shape == [-1, -1]:
    height, width = None, None
  else:
    height, width = spatial_image_shape  # pylint: disable=unpacking-non-sequence

  num_additional_channels = 0
  if fields.InputDataFields.image_additional_channels in tensor_dict:
    num_additional_channels = shape_utils.get_dim_as_int(tensor_dict[
        fields.InputDataFields.image_additional_channels].shape[2])

  # We assume that if num_additional_channels > 0, then it has already been
  # concatenated to the base image (but not the ground truth).
  num_channels = 3
  if fields.InputDataFields.image in tensor_dict:
    num_channels = shape_utils.get_dim_as_int(
        tensor_dict[fields.InputDataFields.image].shape[2])

  if num_additional_channels:
    if num_additional_channels >= num_channels:
      raise ValueError(
          'Image must be already concatenated with additional channels.')

    if (fields.InputDataFields.original_image in tensor_dict and
        shape_utils.get_dim_as_int(
            tensor_dict[fields.InputDataFields.original_image].shape[2]) ==
        num_channels):
      raise ValueError(
          'Image must be already concatenated with additional channels.')

  padding_shapes = {
      fields.InputDataFields.image: [
          height, width, num_channels
      ],
      fields.InputDataFields.original_image_spatial_shape: [2],
      fields.InputDataFields.image_additional_channels: [
          height, width, num_additional_channels
      ],
      fields.InputDataFields.source_id: [],
      fields.InputDataFields.filename: [],
      fields.InputDataFields.key: [],
      fields.InputDataFields.groundtruth_difficult: [max_num_boxes],
      fields.InputDataFields.groundtruth_boxes: [max_num_boxes, 4],
      fields.InputDataFields.groundtruth_classes: [max_num_boxes, num_classes],
      fields.InputDataFields.groundtruth_instance_masks: [
          max_num_boxes, height, width
      ],
      fields.InputDataFields.groundtruth_is_crowd: [max_num_boxes],
      fields.InputDataFields.groundtruth_group_of: [max_num_boxes],
      fields.InputDataFields.groundtruth_area: [max_num_boxes],
      fields.InputDataFields.groundtruth_weights: [max_num_boxes],
      fields.InputDataFields.groundtruth_confidences: [
          max_num_boxes, num_classes
      ],
      fields.InputDataFields.num_groundtruth_boxes: [],
      fields.InputDataFields.groundtruth_label_types: [max_num_boxes],
      fields.InputDataFields.groundtruth_label_weights: [max_num_boxes],
      fields.InputDataFields.true_image_shape: [3],
      fields.InputDataFields.groundtruth_image_classes: [num_classes],
      fields.InputDataFields.groundtruth_image_confidences: [num_classes],
  }

  if fields.InputDataFields.original_image in tensor_dict:
    padding_shapes[fields.InputDataFields.original_image] = [
        height, width,
        shape_utils.get_dim_as_int(tensor_dict[fields.InputDataFields.
                                               original_image].shape[2])
    ]
  if fields.InputDataFields.groundtruth_keypoints in tensor_dict:
    tensor_shape = (
        tensor_dict[fields.InputDataFields.groundtruth_keypoints].shape)
    padding_shape = [max_num_boxes,
                     shape_utils.get_dim_as_int(tensor_shape[1]),
                     shape_utils.get_dim_as_int(tensor_shape[2])]
    padding_shapes[fields.InputDataFields.groundtruth_keypoints] = padding_shape
  if fields.InputDataFields.groundtruth_keypoint_visibilities in tensor_dict:
    tensor_shape = tensor_dict[fields.InputDataFields.
                               groundtruth_keypoint_visibilities].shape
    padding_shape = [max_num_boxes, shape_utils.get_dim_as_int(tensor_shape[1])]
    padding_shapes[fields.InputDataFields.
                   groundtruth_keypoint_visibilities] = padding_shape

  padded_tensor_dict = {}
  for tensor_name in tensor_dict:
    padded_tensor_dict[tensor_name] = shape_utils.pad_or_clip_nd(
        tensor_dict[tensor_name], padding_shapes[tensor_name])

  # Make sure that the number of groundtruth boxes now reflects the
  # padded/clipped tensors.
  if fields.InputDataFields.num_groundtruth_boxes in padded_tensor_dict:
    padded_tensor_dict[fields.InputDataFields.num_groundtruth_boxes] = (
        tf.minimum(
            padded_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
            max_num_boxes))
  return padded_tensor_dict


def augment_input_data(tensor_dict, data_augmentation_options):
  
  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tf.cast(tensor_dict[fields.InputDataFields.image], dtype=tf.float32), 0)

  include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                            in tensor_dict)
  include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                       in tensor_dict)
  include_label_weights = (fields.InputDataFields.groundtruth_weights
                           in tensor_dict)
  include_label_confidences = (fields.InputDataFields.groundtruth_confidences
                               in tensor_dict)
  include_multiclass_scores = (fields.InputDataFields.multiclass_scores in
                               tensor_dict)
  tensor_dict = preprocessor.preprocess(
      tensor_dict, data_augmentation_options,
      func_arg_map=preprocessor.get_default_func_arg_map(
          include_label_weights=include_label_weights,
          include_label_confidences=include_label_confidences,
          include_multiclass_scores=include_multiclass_scores,
          include_instance_masks=include_instance_masks,
          include_keypoints=include_keypoints))
  tensor_dict[fields.InputDataFields.image] = tf.squeeze(
      tensor_dict[fields.InputDataFields.image], axis=0)
  return tensor_dict


def _get_labels_dict(input_dict):
  """Extracts labels dict from input dict."""
  required_label_keys = [
      fields.InputDataFields.num_groundtruth_boxes,
      fields.InputDataFields.groundtruth_boxes,
      fields.InputDataFields.groundtruth_classes,
      fields.InputDataFields.groundtruth_weights,
  ]
  labels_dict = {}
  for key in required_label_keys:
    labels_dict[key] = input_dict[key]

  optional_label_keys = [
      fields.InputDataFields.groundtruth_confidences,
      fields.InputDataFields.groundtruth_keypoints,
      fields.InputDataFields.groundtruth_instance_masks,
      fields.InputDataFields.groundtruth_area,
      fields.InputDataFields.groundtruth_is_crowd,
      fields.InputDataFields.groundtruth_difficult
  ]

  for key in optional_label_keys:
    if key in input_dict:
      labels_dict[key] = input_dict[key]
  if fields.InputDataFields.groundtruth_difficult in labels_dict:
    labels_dict[fields.InputDataFields.groundtruth_difficult] = tf.cast(
        labels_dict[fields.InputDataFields.groundtruth_difficult], tf.int32)
  return labels_dict


def _replace_empty_string_with_random_number(string_tensor):
  

  empty_string = tf.constant('', dtype=tf.string, name='EmptyString')

  random_source_id = tf.as_string(
      tf.random_uniform(shape=[], maxval=2**63 - 1, dtype=tf.int64))

  out_string = tf.cond(
      tf.equal(string_tensor, empty_string),
      true_fn=lambda: random_source_id,
      false_fn=lambda: string_tensor)

  return out_string


def _get_features_dict(input_dict):
  """Extracts features dict from input dict."""

  source_id = _replace_empty_string_with_random_number(
      input_dict[fields.InputDataFields.source_id])

  hash_from_source_id = tf.string_to_hash_bucket_fast(source_id, HASH_BINS)
  features = {
      fields.InputDataFields.image:
          input_dict[fields.InputDataFields.image],
      HASH_KEY: tf.cast(hash_from_source_id, tf.int32),
      fields.InputDataFields.true_image_shape:
          input_dict[fields.InputDataFields.true_image_shape],
      fields.InputDataFields.original_image_spatial_shape:
          input_dict[fields.InputDataFields.original_image_spatial_shape]
  }
  if fields.InputDataFields.original_image in input_dict:
    features[fields.InputDataFields.original_image] = input_dict[
        fields.InputDataFields.original_image]
  return features


def create_train_input_fn(train_config, train_input_config,
                          model_config):
  """Creates a train `input` function for `Estimator`.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in TRAIN mode.
  """

  def _train_input_fn(params=None):
    return train_input(train_config, train_input_config, model_config,
                       params=params)

  return _train_input_fn


def train_input(train_config, train_input_config,
                model_config, model=None, params=None):
  

  if not isinstance(train_config, train_pb2.TrainConfig):
    raise TypeError('For training mode, the `train_config` must be a '
                    'train_pb2.TrainConfig.')
  if not isinstance(train_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `train_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=True).preprocess
  else:
    model_preprocess_fn = model.preprocess

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]
    data_augmentation_fn = functools.partial(
        augment_input_data,
        data_augmentation_options=data_augmentation_options)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=config_util.get_number_of_classes(model_config),
        data_augmentation_fn=data_augmentation_fn,
        merge_multiple_boxes=train_config.merge_multiple_label_boxes,
        retain_original_image=train_config.retain_original_images,
        use_multiclass_scores=train_config.use_multiclass_scores,
        use_bfloat16=train_config.use_bfloat16)

    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=train_input_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config))
    return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict))

  dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
      train_input_config,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      batch_size=params['batch_size'] if params else train_config.batch_size)
  return dataset


def create_eval_input_fn(eval_config, eval_input_config, model_config):
  

  def _eval_input_fn(params=None):
    return eval_input(eval_config, eval_input_config, model_config,
                      params=params)

  return _eval_input_fn


def eval_input(eval_config, eval_input_config, model_config,
               model=None, params=None):
  
  params = params or {}
  if not isinstance(eval_config, eval_pb2.EvalConfig):
    raise TypeError('For eval mode, the `eval_config` must be a '
                    'train_pb2.EvalConfig.')
  if not isinstance(eval_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `eval_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=False).preprocess
  else:
    model_preprocess_fn = model.preprocess

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    num_classes = config_util.get_number_of_classes(model_config)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None,
        retain_original_image=eval_config.retain_original_images)
    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=eval_input_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config))
    return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict))
  dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
      eval_input_config,
      batch_size=params['batch_size'] if params else eval_config.batch_size,
      transform_input_data_fn=transform_and_pad_input_data_fn)
  return dataset


def create_predict_input_fn(model_config, predict_input_config):
  

  def _predict_input_fn(params=None):
    
    del params
    example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')

    num_classes = config_util.get_number_of_classes(model_config)
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=False).preprocess

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None)

    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=False,
        num_additional_channels=predict_input_config.num_additional_channels)
    input_dict = transform_fn(decoder.decode(example))
    images = tf.cast(input_dict[fields.InputDataFields.image], dtype=tf.float32)
    images = tf.expand_dims(images, axis=0)
    true_image_shape = tf.expand_dims(
        input_dict[fields.InputDataFields.true_image_shape], axis=0)

    return tf.estimator.export.ServingInputReceiver(
        features={
            fields.InputDataFields.image: images,
            fields.InputDataFields.true_image_shape: true_image_shape},
        receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})

  return _predict_input_fn
