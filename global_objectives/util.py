import tensorflow as tf


def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):

  with tf.name_scope(
      name,
      'weighted_logistic_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    softplus_term = tf.add(tf.maximum(-logits, 0.0),
                           tf.log(1.0 + tf.exp(-tf.abs(logits))))
    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
  
  with tf.name_scope(
      name, 'weighted_hinge_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    positives_term = positive_weights * labels * tf.maximum(1.0 - logits, 0)
    negatives_term = (negative_weights * (1.0 - labels)
                      * tf.maximum(1.0 + logits, 0))
    return positives_term + negatives_term


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0,
                            name=None):
  
  with tf.name_scope(
      name, 'weighted_loss',
      [logits, labels, surrogate_type, positive_weights,
       negative_weights]) as name:
    if surrogate_type == 'xent':
      return weighted_sigmoid_cross_entropy_with_logits(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    elif surrogate_type == 'hinge':
      return weighted_hinge_loss(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def expand_outer(tensor, rank):
  
  if tensor.get_shape().ndims is None:
    raise ValueError('tensor dimension must be known.')
  if len(tensor.get_shape()) > rank:
    raise ValueError(
        '`rank` must be at least the current tensor dimension: (%s vs %s).' %
        (rank, len(tensor.get_shape())))
  while len(tensor.get_shape()) < rank:
    tensor = tf.expand_dims(tensor, 0)
  return tensor


def build_label_priors(labels,
                       weights=None,
                       positive_pseudocount=1.0,
                       negative_pseudocount=1.0,
                       variables_collections=None):
  
  dtype = labels.dtype.base_dtype
  num_labels = get_num_labels(labels)

  if weights is None:
    weights = tf.ones_like(labels)

  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    # Create variable and update op for weighted label counts.
    weighted_label_counts = tf.contrib.framework.model_variable(
        name='weighted_label_counts',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount] * num_labels, dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weighted_label_counts_update = weighted_label_counts.assign_add(
        tf.reduce_sum(weights * labels, 0))

    # Create variable and update op for the sum of the weights.
    weight_sum = tf.contrib.framework.model_variable(
        name='weight_sum',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount + negative_pseudocount] * num_labels,
            dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weight_sum_update = weight_sum.assign_add(tf.reduce_sum(weights, 0))

  finally:
    tf.get_variable_scope().set_partitioner(partitioner)

  label_priors = tf.div(
      weighted_label_counts_update,
      weight_sum_update)
  return label_priors


def convert_and_cast(value, name, dtype):
  
  return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def prepare_loss_args(labels, logits, positive_weights, negative_weights):
  
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = convert_and_cast(labels, 'labels', logits.dtype)
  if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
    labels = tf.expand_dims(labels, [2])

  positive_weights = convert_and_cast(positive_weights, 'positive_weights',
                                      logits.dtype)
  positive_weights = expand_outer(positive_weights, logits.get_shape().ndims)
  negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                      logits.dtype)
  negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
  return labels, logits, positive_weights, negative_weights


def get_num_labels(labels_or_logits):
  """Returns the number of labels inferred from labels_or_logits."""
  if labels_or_logits.get_shape().ndims <= 1:
    return 1
  return labels_or_logits.get_shape()[1].value
