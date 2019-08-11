

from six.moves import xrange
import tensorflow as tf


def SequenceToImageAndDiff(images):
  
  image_diff_list = []
  image_seq = tf.unstack(images, axis=1)
  for size in [32, 64, 128, 256]:
    resized_images = [
        tf.image.resize_images(i, [size, size]) for i in image_seq]
    diffs = []
    for i in xrange(0, len(resized_images)-1):
      diffs.append(resized_images[i+1] - resized_images[i])
    image_diff_list.append(
        (tf.concat(axis=0, values=resized_images[:-1]), tf.concat(axis=0, values=diffs)))
  return image_diff_list


def ReadInput(data_filepattern, shuffle, params):
  
  image_size = params['image_size']
  filenames = tf.gfile.Glob(data_filepattern)
  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
  reader = tf.TFRecordReader()
  _, example = reader.read(filename_queue)
  feature_sepc = {
      'moving_objs': tf.FixedLenSequenceFeature(
          shape=[image_size * image_size * 3], dtype=tf.float32)}
  _, features = tf.parse_single_sequence_example(
      example, sequence_features=feature_sepc)
  moving_objs = tf.reshape(
      features['moving_objs'], [params['seq_len'], image_size, image_size, 3])
  if shuffle:
    examples = tf.train.shuffle_batch(
        [moving_objs],
        batch_size=params['batch_size'],
        num_threads=64,
        capacity=params['batch_size'] * 100,
        min_after_dequeue=params['batch_size'] * 4)
  else:
    examples = tf.train.batch([moving_objs],
                              batch_size=params['batch_size'],
                              num_threads=16,
                              capacity=params['batch_size'])
  examples /= params['norm_scale']
  return examples
