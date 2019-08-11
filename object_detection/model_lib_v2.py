from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

import tensorflow as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import variables_helper

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP


def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True,
    use_tpu=False,
    use_bfloat16=False):

  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  # TODO(kaftan): Check how we're supposed to do this mixed precision stuff
  ## in TF2 TPUStrategy + Keras
  if use_tpu and use_bfloat16:
    with tf.contrib.tpu.bfloat16_scope():
      prediction_dict = model.predict(
          preprocessed_images,
          features[fields.InputDataFields.true_image_shape])
      prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)
  else:
    prediction_dict = model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape])

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict


# TODO(kaftan): Explore removing learning_rate from this method & returning
## The full losses dict instead of just total_loss, then doing all summaries
## saving in a utility method called by the outer training loop.
# TODO(kaftan): Explore adding gradient summaries
def eager_train_step(detection_model,
                     features,
                     labels,
                     unpad_groundtruth_tensors,
                     optimizer,
                     learning_rate,
                     add_regularization_loss=True,
                     clip_gradients_value=None,
                     use_tpu=False,
                     use_bfloat16=False,
                     global_step=None,
                     num_replicas=1.0):

  # """Execute a single training step in the TF v2 style loop."""
  is_training = True

  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  labels = model_lib.unstack_batch(
      labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

  with tf.GradientTape() as tape:
    losses_dict, _ = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss, use_tpu,
        use_bfloat16)

    total_loss = losses_dict['Loss/total_loss']

    # Normalize loss for num replicas
    total_loss = tf.math.divide(total_loss,
                                tf.constant(num_replicas, dtype=tf.float32))
    losses_dict['Loss/normalized_total_loss'] = total_loss

  for loss_type in losses_dict:
    tf.compat.v2.summary.scalar(
        loss_type, losses_dict[loss_type], step=global_step)

  trainable_variables = detection_model.trainable_variables

  gradients = tape.gradient(total_loss, trainable_variables)

  if clip_gradients_value:
    gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  if not use_tpu:
    tf.compat.v2.summary.scalar('learning_rate', learning_rate,
                                step=global_step)

  return total_loss


def load_fine_tune_checkpoint(
    model, checkpoint_path, checkpoint_type,
    load_all_detection_checkpoint_vars, input_dataset,
    unpad_groundtruth_tensors, use_tpu, use_bfloat16):
  
  features, labels = iter(input_dataset).next()

  def _dummy_computation_fn(features, labels):
    model._is_training = False  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(False)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    return _compute_losses_and_predictions_dicts(
        model,
        features,
        labels,
        use_tpu=use_tpu,
        use_bfloat16=use_bfloat16)

  strategy = tf.compat.v2.distribute.get_strategy()
  strategy.experimental_run_v2(
      _dummy_computation_fn, args=(
          features,
          labels,
      ))
  var_map = model.restore_map(
      fine_tune_checkpoint_type=checkpoint_type,
      load_all_detection_checkpoint_vars=(
          load_all_detection_checkpoint_vars))
  available_var_map = (
      variables_helper.get_variables_available_in_checkpoint(
          var_map,
          checkpoint_path,
          include_global_step=False))
  tf.train.init_from_checkpoint(checkpoint_path,
                                available_var_map)


def train_loop(
    hparams,
    pipeline_config_path,
    model_dir,
    config_override=None,
    train_steps=None,
    use_tpu=False,
    save_final_config=False,
    export_to_tpu=None,
    checkpoint_every_n=1000, **kwargs):
  
  ## Parse the configs
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'train_steps': train_steps,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  configs = merge_external_params_with_configs(
      configs, hparams, kwargs_dict=kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
  use_bfloat16 = train_config.use_bfloat16
  add_regularization_loss = train_config.add_regularization_loss
  clip_gradients_value = None
  if train_config.gradient_clipping_by_norm > 0:
    clip_gradients_value = train_config.gradient_clipping_by_norm

  # update train_steps from config but only when non-zero value is provided
  if train_steps is None and train_config.num_steps != 0:
    train_steps = train_config.num_steps

  # Read export_to_tpu from hparams if not passed.
  if export_to_tpu is None:
    export_to_tpu = hparams.get('export_to_tpu', False)
  tf.logging.info(
      'train_loop: use_tpu %s, export_to_tpu %s', use_tpu,
      export_to_tpu)

  # Parse the checkpoint fine tuning configs
  if hparams.load_pretrained:
    fine_tune_checkpoint_path = train_config.fine_tune_checkpoint
  else:
    fine_tune_checkpoint_path = None
  load_all_detection_checkpoint_vars = (
      train_config.load_all_detection_checkpoint_vars)
  # TODO(kaftan) (or anyone else): move this piece of config munging to
  ## utils/config_util.py
  if not train_config.fine_tune_checkpoint_type:
    # train_config.from_detection_checkpoint field is deprecated. For
    # backward compatibility, set train_config.fine_tune_checkpoint_type
    # based on train_config.from_detection_checkpoint.
    if train_config.from_detection_checkpoint:
      train_config.fine_tune_checkpoint_type = 'detection'
    else:
      train_config.fine_tune_checkpoint_type = 'classification'
  fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type

  # Write the as-run pipeline config to disk.
  if save_final_config:
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, model_dir)

  # TODO(kaftan): Either make strategy a parameter of this method, or
  ## grab it w/  Distribution strategy's get_scope
  # Build the model, optimizer, and training input
  strategy = tf.compat.v2.distribute.MirroredStrategy()
  with strategy.scope():
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # Create the inputs.
    train_input = inputs.train_input(
        train_config=train_config,
        train_input_config=train_input_config,
        model_config=model_config,
        model=detection_model)

    train_input = strategy.experimental_distribute_dataset(
        train_input.repeat())

    global_step = tf.compat.v2.Variable(
        0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
    optimizer, (learning_rate,) = optimizer_builder.build(
        train_config.optimizer, global_step=global_step)

    if callable(learning_rate):
      learning_rate_fn = learning_rate
    else:
      learning_rate_fn = lambda: learning_rate

  ## Train the model
  summary_writer = tf.compat.v2.summary.create_file_writer(model_dir + '/train')
  with summary_writer.as_default():
    with strategy.scope():
      # Load a fine-tuning checkpoint.
      if fine_tune_checkpoint_path:
        load_fine_tune_checkpoint(detection_model, fine_tune_checkpoint_path,
                                  fine_tune_checkpoint_type,
                                  load_all_detection_checkpoint_vars,
                                  train_input,
                                  unpad_groundtruth_tensors, use_tpu,
                                  use_bfloat16)

      ckpt = tf.compat.v2.train.Checkpoint(
          step=global_step, model=detection_model)
      manager = tf.compat.v2.train.CheckpointManager(
          ckpt, model_dir, max_to_keep=7)
      ## Maybe re-enable checkpoint restoration depending on how it works:
      # ckpt.restore(manager.latest_checkpoint)

      def train_step_fn(features, labels):
        return eager_train_step(
            detection_model,
            features,
            labels,
            unpad_groundtruth_tensors,
            optimizer,
            learning_rate=learning_rate_fn(),
            use_bfloat16=use_bfloat16,
            add_regularization_loss=add_regularization_loss,
            clip_gradients_value=clip_gradients_value,
            use_tpu=use_tpu,
            global_step=global_step,
            num_replicas=strategy.num_replicas_in_sync)

      @tf.function
      def _dist_train_step(data_iterator):
        """A distributed train step."""
        features, labels = data_iterator.next()
        per_replica_losses = strategy.experimental_run_v2(
            train_step_fn, args=(
                features,
                labels,
            ))
        # TODO(anjalisridhar): explore if it is safe to remove the
        ## num_replicas scaling of the loss and switch this to a ReduceOp.Mean
        mean_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return mean_loss

      train_input_iter = iter(train_input)
      for _ in range(train_steps):
        start_time = time.time()

        loss = _dist_train_step(train_input_iter)
        global_step.assign_add(1)
        end_time = time.time()
        tf.compat.v2.summary.scalar(
            'steps_per_sec', 1.0 / (end_time - start_time), step=global_step)
        # TODO(kaftan): Remove this print after it is no longer helpful for
        ## debugging.
        tf.print('Finished step', global_step, end_time, loss)
        if int(global_step.value().numpy()) % checkpoint_every_n == 0:
          manager.save()


def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    use_tpu=False,
    postprocess_on_cpu=False,
    global_step=None):
  
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']
  use_bfloat16 = train_config.use_bfloat16
  add_regularization_loss = train_config.add_regularization_loss

  is_training = False
  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  evaluator_options = eval_util.evaluator_options_from_eval_config(
      eval_config)

  class_agnostic_category_index = (
      label_map_util.create_class_agnostic_category_index())
  class_agnostic_evaluators = eval_util.get_evaluators(
      eval_config,
      list(class_agnostic_category_index.values()),
      evaluator_options)

  class_aware_evaluators = None
  if eval_input_config.label_map_path:
    class_aware_category_index = (
        label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path))
    class_aware_evaluators = eval_util.get_evaluators(
        eval_config,
        list(class_aware_category_index.values()),
        evaluator_options)

  evaluators = None
  loss_metrics = {}

  @tf.function
  def compute_eval_dict(features, labels):
    """Compute the evaluation result on an image."""
    # For evaling on train data, it is necessary to check whether groundtruth
    # must be unpadded.
    boxes_shape = (
        labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list())
    unpad_groundtruth_tensors = boxes_shape[1] is not None and not use_tpu
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss, use_tpu,
        use_bfloat16)

    def postprocess_wrapper(args):
      return detection_model.postprocess(args[0], args[1])

    # TODO(kaftan): Depending on how postprocessing will work for TPUS w/
    ## TPUStrategy, may be good to move wrapping to a utility method
    if use_tpu and postprocess_on_cpu:
      detections = tf.contrib.tpu.outside_compilation(
          postprocess_wrapper,
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))
    else:
      detections = postprocess_wrapper(
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))

    class_agnostic = (
        fields.DetectionResultFields.detection_classes not in detections)
    # TODO(kaftan) (or anyone): move `_prepare_groundtruth_for_eval to eval_util
    ## and call this from there.
    groundtruth = model_lib._prepare_groundtruth_for_eval(  # pylint: disable=protected-access
        detection_model, class_agnostic, eval_input_config.max_number_of_boxes)
    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images:
      eval_images = features[fields.InputDataFields.original_image]
      true_image_shapes = tf.slice(
          features[fields.InputDataFields.true_image_shape], [0, 0], [-1, 3])
      original_image_spatial_shapes = features[
          fields.InputDataFields.original_image_spatial_shape]
    else:
      eval_images = features[fields.InputDataFields.image]
      true_image_shapes = None
      original_image_spatial_shapes = None

    eval_dict = eval_util.result_dict_for_batched_example(
        eval_images,
        features[inputs.HASH_KEY],
        detections,
        groundtruth,
        class_agnostic=class_agnostic,
        scale_to_absolute=True,
        original_image_spatial_shapes=original_image_spatial_shapes,
        true_image_shapes=true_image_shapes)

    return eval_dict, losses_dict, class_agnostic

  i = 0
  for features, labels in eval_dataset:
    eval_dict, losses_dict, class_agnostic = compute_eval_dict(features, labels)
    end_time = time.time()
    # TODO(kaftan): Remove this print after it is no longer helpful for
    ## debugging.
    tf.print('Finished eval dict computation', i, end_time)
    i += 1

    if evaluators is None:
      if class_agnostic:
        evaluators = class_agnostic_evaluators
      else:
        evaluators = class_aware_evaluators

    for evaluator in evaluators:
      evaluator.add_eval_dict(eval_dict)

    for loss_key, loss_tensor in iter(losses_dict.items()):
      if loss_key not in loss_metrics:
        loss_metrics[loss_key] = tf.keras.metrics.Mean()
      loss_metrics[loss_key].update_state(loss_tensor)

  eval_metrics = {}

  for evaluator in evaluators:
    eval_metrics.update(evaluator.evaluate())
  for loss_key in loss_metrics:
    eval_metrics[loss_key] = loss_metrics[loss_key].result()

  eval_metrics = {str(k): v for k, v in eval_metrics.items()}
  for k in eval_metrics:
    tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)

  return eval_metrics


def eval_continuously(
    hparams,
    pipeline_config_path,
    config_override=None,
    train_steps=None,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=1,
    use_tpu=False,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    export_to_tpu=None,
    model_dir=None,
    checkpoint_dir=None,
    wait_interval=180,
    **kwargs):
  
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if train_steps is not None:
    kwargs['train_steps'] = train_steps
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, hparams, kwargs_dict=kwargs)
  model_config = configs['model']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  detection_model = model_builder.build(
      model_config=model_config, is_training=True)

  # Create the inputs.
  eval_inputs = []
  for eval_input_config in eval_input_configs:
    next_eval_input = inputs.eval_input(
        eval_config=eval_config,
        eval_input_config=eval_input_config,
        model_config=model_config,
        model=detection_model)
    eval_inputs.append((eval_input_config.name, next_eval_input))

  # Read export_to_tpu from hparams if not passed.
  if export_to_tpu is None:
    export_to_tpu = hparams.get('export_to_tpu', False)
  tf.logging.info('eval_continuously: use_tpu %s, export_to_tpu %s',
                  use_tpu, export_to_tpu)

  global_step = tf.compat.v2.Variable(
      0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

  prev_checkpoint = None
  waiting = False
  while True:
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model)
    manager = tf.compat.v2.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=3)

    latest_checkpoint = manager.latest_checkpoint
    if prev_checkpoint == latest_checkpoint:
      if prev_checkpoint is None:
        tf.logging.info('No checkpoints found yet. Trying again in %s seconds.'
                        % wait_interval)
        time.sleep(wait_interval)
      else:
        if waiting:
          tf.logging.info('Terminating eval after %s seconds of no new '
                          'checkpoints.' % wait_interval)
          break
        else:
          tf.logging.info('No new checkpoint found. Will try again '
                          'in %s seconds and terminate if no checkpoint '
                          'appears.' % wait_interval)
          waiting = True
          time.sleep(wait_interval)
    else:
      tf.logging.info('New checkpoint found. Starting evaluation.')
      waiting = False
      prev_checkpoint = latest_checkpoint
      ckpt.restore(latest_checkpoint)

      for eval_name, eval_input in eval_inputs:
        summary_writer = tf.compat.v2.summary.create_file_writer(
            model_dir + '/eval' + eval_name)
        with summary_writer.as_default():
          eager_eval_loop(
              detection_model,
              configs,
              eval_input,
              use_tpu=use_tpu,
              postprocess_on_cpu=postprocess_on_cpu,
              global_step=global_step)
