# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow
# /transformer
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range
from open_seq2seq.parts.transformer_tts.utils import get_window_attention_bias
from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv
from open_seq2seq.encoders import Encoder
from open_seq2seq.parts.transformer import attention_layer, ffn_layer, utils, \
                                           embedding_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                                  LayerNormalization


class Prenet:
  """
  Encoder Pre-net for TTS Transformer.

  Consists of convolution layers and does not contain an embedding layer.
  """

  def __init__(self, conv_layers, activation_fn, cnn_dropout_prob, regularizer, training,
               data_format, bn_momentum, bn_epsilon):
    self.conv_layers = conv_layers
    self.activation_fn = activation_fn
    self.cnn_dropout_prob = cnn_dropout_prob
    self.regularizer = regularizer
    self.training = training
    self.data_format = data_format
    self.bn_momentum = bn_momentum
    self.bn_epsilon = bn_epsilon

  def __call__(self, input_layer, text_len):
    if self.data_format == "channels_last":
      top_layer = input_layer
    else:
      top_layer = tf.transpose(input_layer, [0, 2, 1])

    for i, conv_params in enumerate(self.conv_layers):
      ch_out = conv_params["num_channels"]
      kernel_size = conv_params["kernel_size"]  # [time, freq]
      strides = conv_params["stride"]
      padding = conv_params["padding"]

      if padding == "VALID":
        text_len = (text_len - kernel_size[0] + strides[0]) // strides[0]
      else:
        text_len = (text_len + strides[0] - 1) // strides[0]

      top_layer = conv_bn_actv(
        layer_type="conv1d",
        name="conv{}".format(i + 1),
        inputs=top_layer,
        filters=ch_out,
        kernel_size=kernel_size,
        activation_fn=self.activation_fn,
        strides=strides,
        padding=padding,
        regularizer=self.regularizer,
        training=self.training,
        data_format=self.data_format,
        bn_momentum=self.bn_momentum,
        bn_epsilon=self.bn_epsilon
      )
      top_layer = tf.layers.dropout(top_layer, rate=self.cnn_dropout_prob, training=self.training)

    if self.data_format == "channels_first":
      top_layer = tf.transpose(top_layer, [0, 2, 1])

    return top_layer


class TransformerTTSEncoder(Encoder):
  """Transformer model encoder"""

  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_required_params(), **{
      "encoder_layers": int,
      "hidden_size": int,
      "num_heads": int,
      "attention_dropout": float,
      "filter_size": int,
      "src_vocab_size": int,
      "relu_dropout": float,
      "layer_postprocess_dropout": float,
      "remove_padding": bool,
      "conv_layers": list,
      "activation_fn": None,
      "cnn_dropout_prob": float,
    })

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_optional_params(), **{
      "regularizer": None,  # any valid TensorFlow regularizer
      "regularizer_params": dict,
      "initializer": None,  # any valid TensorFlow initializer
      "initializer_params": dict,
      "pad_embeddings_2_eight": bool,
      "data_format": str,
      "bn_momentum": float,
      "bn_epsilon": float,
      "window_size": int
    })

  def __init__(self, params, model, name="transformer_tts_encoder", mode="train"):
    super(TransformerTTSEncoder, self).__init__(
      params, model, name=name, mode=mode,
    )
    self.layers = []
    self.output_normalization = None
    self._mode = mode

    self.embedding_softmax_layer = None
    self.prenet = None

  def _call(self, encoder_inputs, attention_bias, inputs_padding):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    return self.output_normalization(encoder_inputs)

  def _encode(self, input_dict):
    training = self.mode == "train"

    if len(self.layers) == 0:
      # prepare encoder graph
      self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        self.params["src_vocab_size"], self.params["hidden_size"],
        pad_vocab_to_eight=self.params.get("pad_embeddings_2_eight", False),
      )

      # initialize Encoder Pre-net
      self.prenet = Prenet(
        conv_layers=self.params["conv_layers"],
        activation_fn=self.params["activation_fn"],
        cnn_dropout_prob=self.params["cnn_dropout_prob"],
        regularizer=self.params.get("regularizer", None),
        training=training,
        data_format=self.params.get("data_format", "channels_last"),
        bn_momentum=self.params.get("bn_momentum", 0.1),
        bn_epsilon=self.params.get("bn_epsilon", 1e-5)
      )

      for _ in range(self.params["encoder_layers"]):
        # Create sublayers for each layer.
        self_attention_layer = attention_layer.SelfAttention(
          self.params["hidden_size"], self.params["num_heads"],
          self.params["attention_dropout"], training
        )
        feed_forward_network = ffn_layer.FeedFowardNetwork(
          self.params["hidden_size"], self.params["filter_size"],
          self.params["relu_dropout"], training
        )

        self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, self.params, training),
          PrePostProcessingWrapper(feed_forward_network, self.params, training)
        ])

      # Create final layer normalization layer.
      self.output_normalization = LayerNormalization(self.params["hidden_size"])

    # actual encoder part
    with tf.name_scope("encode"):
      inputs = input_dict["source_tensors"][0]
      text_len = input_dict["source_tensors"][1]

      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.

      embedded_inputs = self.embedding_softmax_layer(inputs)
      prenet_inputs = self.prenet(embedded_inputs, text_len)

      if self.params["remove_padding"]:
        inputs_padding = utils.get_padding(inputs)
      else:
        inputs_padding = None

      inputs_attention_bias = utils.get_padding_bias(inputs)
      self_attention_bias = inputs_attention_bias

      # with tf.name_scope("add_pos_encoding"):
      #   length = tf.shape(prenet_inputs)[1]
      #   pos_encoding = utils.get_position_encoding(length, self.params["hidden_size"])
      #   encoder_inputs = prenet_inputs + tf.cast(x=pos_encoding, dtype=prenet_inputs.dtype)
      encoder_inputs = prenet_inputs

      if self.mode == "train":
        encoder_inputs = tf.nn.dropout(encoder_inputs, keep_prob=1.0 - self.params["layer_postprocess_dropout"])

      linear_projection = tf.layers.Dense(name="linear_projection", units=self.params["hidden_size"])
      encoder_inputs = linear_projection(encoder_inputs)

      length = tf.shape(encoder_inputs)[1]
      window_size = self.params.get("window_size", -1)
      self_attention_bias += get_window_attention_bias(length, window_size)

      encoded = self._call(encoder_inputs, self_attention_bias, inputs_padding)

      return {
        "outputs": encoded,
        "inputs_attention_bias": inputs_attention_bias,
        "src_lengths": input_dict["source_tensors"][1]
      }
