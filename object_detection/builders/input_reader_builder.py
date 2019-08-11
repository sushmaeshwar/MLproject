

import tensorflow as tf

from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2

parallel_reader = tf.contrib.slim.parallel_reader


def build(input_reader_config):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')
    _, string_tensor = parallel_reader.parallel_read(
        config.input_path[:],  # Convert `RepeatedScalarContainer` to list.
        reader_class=tf.TFRecordReader,
        num_epochs=(input_reader_config.num_epochs
                    if input_reader_config.num_epochs else None),
        num_readers=input_reader_config.num_readers,
        shuffle=input_reader_config.shuffle,
        dtypes=[tf.string, tf.string],
        capacity=input_reader_config.queue_capacity,
        min_after_dequeue=input_reader_config.min_after_dequeue)

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file)
    return decoder.decode(string_tensor)

  raise ValueError('Unsupported input_reader_config.')
