
import functools

import tensorflow as tf
from object_detection.builders import calibration_builder
from object_detection.core import post_processing
from object_detection.protos import post_processing_pb2


def build(post_processing_config):
  
  if not isinstance(post_processing_config, post_processing_pb2.PostProcessing):
    raise ValueError('post_processing_config not of type '
                     'post_processing_pb2.Postprocessing.')
  non_max_suppressor_fn = _build_non_max_suppressor(
      post_processing_config.batch_non_max_suppression)
  score_converter_fn = _build_score_converter(
      post_processing_config.score_converter,
      post_processing_config.logit_scale)
  if post_processing_config.HasField('calibration_config'):
    score_converter_fn = _build_calibrated_score_converter(
        score_converter_fn,
        post_processing_config.calibration_config)
  return non_max_suppressor_fn, score_converter_fn


def _build_non_max_suppressor(nms_config):
  
  if nms_config.iou_threshold < 0 or nms_config.iou_threshold > 1.0:
    raise ValueError('iou_threshold not in [0, 1.0].')
  if nms_config.max_detections_per_class > nms_config.max_total_detections:
    raise ValueError('max_detections_per_class should be no greater than '
                     'max_total_detections.')
  if nms_config.soft_nms_sigma < 0.0:
    raise ValueError('soft_nms_sigma should be non-negative.')
  non_max_suppressor_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=nms_config.score_threshold,
      iou_thresh=nms_config.iou_threshold,
      max_size_per_class=nms_config.max_detections_per_class,
      max_total_size=nms_config.max_total_detections,
      use_static_shapes=nms_config.use_static_shapes,
      use_class_agnostic_nms=nms_config.use_class_agnostic_nms,
      max_classes_per_detection=nms_config.max_classes_per_detection,
      soft_nms_sigma=nms_config.soft_nms_sigma)
  return non_max_suppressor_fn


def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
  """Create a function to scale logits then apply a Tensorflow function."""
  def score_converter_fn(logits):
    scaled_logits = tf.divide(logits, logit_scale, name='scale_logits')
    return tf_score_converter_fn(scaled_logits, name='convert_scores')
  score_converter_fn.__name__ = '%s_with_logit_scale' % (
      tf_score_converter_fn.__name__)
  return score_converter_fn


def _build_score_converter(score_converter_config, logit_scale):
  
  if score_converter_config == post_processing_pb2.PostProcessing.IDENTITY:
    return _score_converter_fn_with_logit_scale(tf.identity, logit_scale)
  if score_converter_config == post_processing_pb2.PostProcessing.SIGMOID:
    return _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale)
  if score_converter_config == post_processing_pb2.PostProcessing.SOFTMAX:
    return _score_converter_fn_with_logit_scale(tf.nn.softmax, logit_scale)
  raise ValueError('Unknown score converter.')


def _build_calibrated_score_converter(score_converter_fn, calibration_config):
  
  calibration_fn = calibration_builder.build(calibration_config)
  def calibrated_score_converter_fn(logits):
    converted_logits = score_converter_fn(logits)
    return calibration_fn(converted_logits)
  calibrated_score_converter_fn.__name__ = (
      'calibrate_with_%s' % calibration_config.WhichOneof('calibrator'))
  return calibrated_score_converter_fn
