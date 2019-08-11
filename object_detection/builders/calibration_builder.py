

import tensorflow as tf
from object_detection.utils import shape_utils


def _find_interval_containing_new_value(x, new_value):
  """Find the index of x (ascending-ordered) after which new_value occurs."""
  new_value_shape = shape_utils.combined_static_and_dynamic_shape(new_value)[0]
  x_shape = shape_utils.combined_static_and_dynamic_shape(x)[0]
  compare = tf.cast(tf.reshape(new_value, shape=(new_value_shape, 1)) >=
                    tf.reshape(x, shape=(1, x_shape)),
                    dtype=tf.int32)
  diff = compare[:, 1:] - compare[:, :-1]
  interval_idx = tf.argmin(diff, axis=1)
  return interval_idx


def _tf_linear_interp1d(x_to_interpolate, fn_x, fn_y):
  
  x_pad = tf.concat([fn_x[:1] - 1, fn_x, fn_x[-1:] + 1], axis=0)
  y_pad = tf.concat([fn_y[:1], fn_y, fn_y[-1:]], axis=0)
  interval_idx = _find_interval_containing_new_value(x_pad, x_to_interpolate)

  # Interpolate
  alpha = (
      (x_to_interpolate - tf.gather(x_pad, interval_idx)) /
      (tf.gather(x_pad, interval_idx + 1) - tf.gather(x_pad, interval_idx)))
  interpolation = ((1 - alpha) * tf.gather(y_pad, interval_idx) +
                   alpha * tf.gather(y_pad, interval_idx + 1))

  return interpolation


def _function_approximation_proto_to_tf_tensors(x_y_pairs_message):
  """Extracts (x,y) pairs from a XYPairs message.

  Args:
    x_y_pairs_message: calibration_pb2..XYPairs proto
  Returns:
    tf_x: tf.float32 tensor of shape (number_xy_pairs,) for function domain.
    tf_y: tf.float32 tensor of shape (number_xy_pairs,) for function range.
  """
  tf_x = tf.convert_to_tensor([x_y_pair.x
                               for x_y_pair
                               in x_y_pairs_message.x_y_pair],
                              dtype=tf.float32)
  tf_y = tf.convert_to_tensor([x_y_pair.y
                               for x_y_pair
                               in x_y_pairs_message.x_y_pair],
                              dtype=tf.float32)
  return tf_x, tf_y


def _get_class_id_function_dict(calibration_config):
  
  class_id_function_dict = {}
  class_id_xy_pairs_map = (
      calibration_config.class_id_function_approximations.class_id_xy_pairs_map)
  for class_id in class_id_xy_pairs_map:
    class_id_function_dict[class_id] = (
        _function_approximation_proto_to_tf_tensors(
            class_id_xy_pairs_map[class_id]))

  return class_id_function_dict


def build(calibration_config):
  
  # Linear Interpolation (usually used as a result of calibration via
  # isotonic regression).
  if calibration_config.WhichOneof('calibrator') == 'function_approximation':

    def calibration_fn(class_predictions_with_background):
      
      # Flattening Tensors and then reshaping at the end.
      flat_class_predictions_with_background = tf.reshape(
          class_predictions_with_background, shape=[-1])
      fn_x, fn_y = _function_approximation_proto_to_tf_tensors(
          calibration_config.function_approximation.x_y_pairs)
      updated_scores = _tf_linear_interp1d(
          flat_class_predictions_with_background, fn_x, fn_y)

      # Un-flatten the scores
      original_detections_shape = shape_utils.combined_static_and_dynamic_shape(
          class_predictions_with_background)
      calibrated_class_predictions_with_background = tf.reshape(
          updated_scores,
          shape=original_detections_shape,
          name='calibrate_scores')
      return calibrated_class_predictions_with_background

  elif (calibration_config.WhichOneof('calibrator') ==
        'class_id_function_approximations'):

    def calibration_fn(class_predictions_with_background):
      
      class_id_function_dict = _get_class_id_function_dict(calibration_config)

      # Tensors are split by class and then recombined at the end to recover
      # the input's original shape. If a class id does not have calibration
      # parameters, it is left unchanged.
      class_tensors = tf.unstack(class_predictions_with_background, axis=-1)
      calibrated_class_tensors = []
      for class_id, class_tensor in enumerate(class_tensors):
        flat_class_tensor = tf.reshape(class_tensor, shape=[-1])
        if class_id in class_id_function_dict:
          output_tensor = _tf_linear_interp1d(
              x_to_interpolate=flat_class_tensor,
              fn_x=class_id_function_dict[class_id][0],
              fn_y=class_id_function_dict[class_id][1])
        else:
          tf.logging.info(
              'Calibration parameters for class id `%d` not not found',
              class_id)
          output_tensor = flat_class_tensor
        calibrated_class_tensors.append(output_tensor)

      combined_calibrated_tensor = tf.stack(calibrated_class_tensors, axis=1)
      input_shape = shape_utils.combined_static_and_dynamic_shape(
          class_predictions_with_background)
      calibrated_class_predictions_with_background = tf.reshape(
          combined_calibrated_tensor,
          shape=input_shape,
          name='calibrate_scores')
      return calibrated_class_predictions_with_background

  # TODO(zbeaver): Add sigmoid calibration.
  else:
    raise ValueError('No calibration builder defined for "Oneof" in '
                     'calibration_config.')

  return calibration_fn
