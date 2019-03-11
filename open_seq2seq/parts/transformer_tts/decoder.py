from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from open_seq2seq.parts.transformer import attention_layer, ffn_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, LayerNormalization


class TransformerDecoder:
  def __init__(self, params, training):
    self.layers = []

    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
        params["hidden_size"], params["num_heads"],
        params["attention_dropout"], training
      )
      enc_dec_attention_layer = attention_layer.Attention(
        params["hidden_size"], params["num_heads"],
        params["attention_dropout"], training
      )
      feed_forward_network = ffn_layer.FeedFowardNetwork(
        params["hidden_size"], params["filter_size"],
        params["relu_dropout"], training
      )

      self.layers.append([
        PrePostProcessingWrapper(self_attention_layer, params, training),
        PrePostProcessingWrapper(enc_dec_attention_layer, params, training, pass_value=not training),
        PrePostProcessingWrapper(feed_forward_network, params, training)
      ])

    self.output_normalization = LayerNormalization(params["hidden_size"])

  def __call__(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias,
               cache=None, last_positions=None, window_size=None):
    monotonic_attention = last_positions is not None
    new_last_positions = []

    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          decoder_inputs = self_attention_layer(decoder_inputs, decoder_self_attention_bias, cache=layer_cache)

        with tf.variable_scope("encdec_attention"):
          if monotonic_attention:
            decoder_inputs, new_last_position = enc_dec_attention_layer(
              decoder_inputs,
              encoder_outputs,
              attention_bias,
              last_positions=last_positions[n, :, :],
              window_size=window_size
            )
            new_last_positions.append(new_last_position)
          else:
            decoder_inputs = enc_dec_attention_layer(decoder_inputs, encoder_outputs, attention_bias)

        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    if monotonic_attention:
      new_last_positions = tf.stack(new_last_positions)
    else:
      new_last_positions = None

    return self.output_normalization(decoder_inputs), new_last_positions
