

import itertools
import tensorflow as tf
from object_detection.inference import detection_inference

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  with tf.Session() as sess:
    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    with tf.python_io.TFRecordWriter(
        FLAGS.output_tfrecord_path) as tf_record_writer:
      try:
        for counter in itertools.count():
          tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                 counter)
          tf_example = detection_inference.infer_detections_and_add_to_example(
              serialized_example_tensor, detected_boxes_tensor,
              detected_scores_tensor, detected_labels_tensor,
              FLAGS.discard_image_pixels)
          tf_record_writer.write(tf_example.SerializeToString())
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing records')


if __name__ == '__main__':
  tf.app.run()
