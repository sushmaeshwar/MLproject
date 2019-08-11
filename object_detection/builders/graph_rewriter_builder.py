

import tensorflow as tf


def build(graph_rewriter_config, is_training):
  
  def graph_rewrite_fn():
    """Function to quantize weights and activation of the default graph."""
    if (graph_rewriter_config.quantization.weight_bits != 8 or
        graph_rewriter_config.quantization.activation_bits != 8):
      raise ValueError('Only 8bit quantization is supported')

    # Quantize the graph by inserting quantize ops for weights and activations
    if is_training:
      tf.contrib.quantize.experimental_create_training_graph(
          input_graph=tf.get_default_graph(),
          quant_delay=graph_rewriter_config.quantization.delay
      )
    else:
      tf.contrib.quantize.experimental_create_eval_graph(
          input_graph=tf.get_default_graph()
      )

    tf.contrib.layers.summarize_collection('quant_vars')
  return graph_rewrite_fn
