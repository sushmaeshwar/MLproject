import numpy
import tensorflow as tf

from global_objectives import util


def precision_recall_auc_loss(
    labels,
    logits,
    precision_range=(0.0, 1.0),
    num_anchors=20,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):

  with tf.variable_scope(scope,
                         'precision_recall_auc',
                         [labels, logits, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create Tensor of anchor points and distance between anchors.
    precision_values, delta = _range_to_anchors_and_delta(
        precision_range, num_anchors, logits.dtype)
    # Create lambdas with shape [1, num_labels, num_anchors].
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Create biases with shape [1, num_labels, num_anchors].
    biases = tf.contrib.framework.model_variable(
        name='biases',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=tf.zeros_initializer(),
        collections=variables_collections,
        trainable=trainable)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)
    label_priors = tf.reshape(label_priors, [1, num_labels, 1])

    # Expand logits, labels, and weights to shape [batch_size, num_labels, 1].
    logits = tf.expand_dims(logits, 2)
    labels = tf.expand_dims(labels, 2)
    weights = tf.expand_dims(weights, 2)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    loss = weights * util.weighted_surrogate_loss(
        labels,
        logits + biases,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - precision_values),
        negative_weights=lambdas * precision_values)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
    per_anchor_loss = loss - lambda_term
    per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
    # Normalize the AUC such that a perfect score function will have AUC 1.0.
    # Because precision_range is discretized into num_anchors + 1 intervals
    # but only num_anchors terms are included in the Riemann sum, the
    # effective length of the integration interval is `delta` less than the
    # length of precision_range.
    scaled_loss = tf.div(per_label_loss,
                         precision_range[1] - precision_range[0] - delta,
                         name='AUC_Normalize')
    scaled_loss = tf.reshape(scaled_loss, original_shape)

    other_outputs = {
        'lambdas': lambdas_variable,
        'biases': biases,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return scaled_loss, other_outputs


def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
  
  with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
    # Convert inputs to tensors and standardize dtypes.
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = util.weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, original_shape)
    return loss, {}


def recall_at_precision_loss(
    labels,
    logits,
    target_precision,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):

  with tf.variable_scope(scope,
                         'recall_at_precision',
                         [logits, labels, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_precision = util.convert_and_cast(
        target_precision, 'target_precision', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - target_precision),
        negative_weights=lambdas * target_precision)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - target_precision) * label_priors * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return loss, other_outputs


def precision_at_recall_loss(
    labels,
    logits,
    target_recall,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
 
  with tf.variable_scope(scope,
                         'precision_at_recall',
                         [logits, labels, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_recall = util.convert_and_cast(
        target_recall, 'target_recall', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type,
        positive_weights=lambdas,
        negative_weights=1.0)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * label_priors * (target_recall - 1.0) * maybe_log2
    loss = tf.reshape(weighted_loss + lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return loss, other_outputs


def false_positive_rate_at_true_positive_rate_loss(
    labels,
    logits,
    target_rate,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):

  return precision_at_recall_loss(labels=labels,
                                  logits=logits,
                                  target_recall=target_rate,
                                  weights=weights,
                                  dual_rate_factor=dual_rate_factor,
                                  label_priors=label_priors,
                                  surrogate_type=surrogate_type,
                                  lambdas_initializer=lambdas_initializer,
                                  reuse=reuse,
                                  variables_collections=variables_collections,
                                  trainable=trainable,
                                  scope=scope)


def true_positive_rate_at_false_positive_rate_loss(
    labels,
    logits,
    target_rate,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):

  with tf.variable_scope(scope,
                         'tpr_at_fpr',
                         [labels, logits, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_rate = util.convert_and_cast(
        target_rate, 'target_rate', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Loss op and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0,
        negative_weights=lambdas)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * target_rate * (1.0 - label_priors) * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

  return loss, other_outputs


def _prepare_labels_logits_weights(labels, logits, weights):
  
  # Convert `labels` and `logits` to Tensors and standardize dtypes.
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = util.convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
  weights = util.convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

  try:
    labels.get_shape().merge_with(logits.get_shape())
  except ValueError:
    raise ValueError('logits and labels must have the same shape (%s vs %s)' %
                     (logits.get_shape(), labels.get_shape()))

  original_shape = labels.get_shape().as_list()
  if labels.get_shape().ndims > 0:
    original_shape[0] = -1
  if labels.get_shape().ndims <= 1:
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])

  if weights.get_shape().ndims == 1:
    # Weights has shape [batch_size]. Reshape to [batch_size, 1].
    weights = tf.reshape(weights, [-1, 1])
  if weights.get_shape().ndims == 0:
    # Weights is a scalar. Change shape of weights to match logits.
    weights *= tf.ones_like(logits)

  return labels, logits, weights, original_shape


def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
  
  # Validate precision_range.
  if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
    raise ValueError('precision values must obey 0 <= %f <= %f <= 1' %
                     (precision_range[0], precision_range[-1]))
  if not 0 < len(precision_range) < 3:
    raise ValueError('length of precision_range (%d) must be 1 or 2' %
                     len(precision_range))

  # Sets precision_values uniformly between min_precision and max_precision.
  values = numpy.linspace(start=precision_range[0],
                          stop=precision_range[1],
                          num=num_anchors+2)[1:-1]
  precision_values = util.convert_and_cast(
      values, 'precision_values', dtype)
  delta = util.convert_and_cast(
      values[0] - precision_range[0], 'delta', dtype)
  # Makes precision_values [1, 1, num_anchors].
  precision_values = util.expand_outer(precision_values, 3)
  return precision_values, delta


def _create_dual_variable(name, shape, dtype, initializer, collections,
                          trainable, dual_rate_factor):
  
  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    dual_variable = tf.contrib.framework.model_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        collections=collections,
        trainable=trainable)
  finally:
    tf.get_variable_scope().set_partitioner(partitioner)
  # Using the absolute value enforces nonnegativity.
  dual_value = tf.abs(dual_variable)

  if trainable:
    # To reverse the gradient on the dual variable, multiply the gradient by
    # -dual_rate_factor
    dual_value = (tf.stop_gradient((1.0 + dual_rate_factor) * dual_value)
                  - dual_rate_factor * dual_value)
  return dual_value, dual_variable


def maybe_create_label_priors(label_priors,
                              labels,
                              weights,
                              variables_collections):
  
  if label_priors is not None:
    label_priors = util.convert_and_cast(
        label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
    return tf.squeeze(label_priors)

  label_priors = util.build_label_priors(
      labels,
      weights,
      variables_collections=variables_collections)
  return label_priors


def true_positives_lower_bound(labels, logits, weights, surrogate_type):
  
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
    labels = tf.expand_dims(labels, 2)
  loss_on_positives = util.weighted_surrogate_loss(
      labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights * (labels - loss_on_positives), 0)


def false_positives_upper_bound(labels, logits, weights, surrogate_type):
  
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  loss_on_negatives = util.weighted_surrogate_loss(
      labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights *  loss_on_negatives, 0)
