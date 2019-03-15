import tensorflow as tf

from open_seq2seq.encoders import Encoder
from open_seq2seq.parts.transformer import embedding_layer
from open_seq2seq.parts.transformer import utils


class BatchNorm1D:
  def __init__(self, *args, **kwargs):
    super(BatchNorm1D, self).__init__()
    self.norm = tf.layers.BatchNormalization(*args, **kwargs)

  def __call__(self, x, training):
    with tf.variable_scope("batch_norm_1d"):
      y = tf.expand_dims(x, axis=1)
      y = self.norm(y, training=training)
      y = tf.squeeze(y, axis=1)
      return y


class ConvBlock:
  def __init__(self, name, conv, norm, activation_fn, dropout, training, is_residual, is_causal):
    self.name = name
    self.conv = conv
    self.norm = norm
    self.activation_fn = activation_fn
    self.dropout = dropout
    self.training = training
    self.is_residual = is_residual
    self.is_casual = is_causal

  def __call__(self, x):
    with tf.variable_scope(self.name):
      if self.is_casual:
        # Add padding from the left side to avoid looking to the future
        pad_size = self.conv.kernel_size[0] - 1
        y = tf.pad(x, [[0, 0], [pad_size, 0], [0, 0]])
      else:
        y = x

      y = self.conv(y)

      if self.norm is not None:
        y = self.norm(y, training=self.training)

      if self.activation_fn is not None:
        y = self.activation_fn(y)

      if self.dropout is not None:
        y = self.dropout(y, training=self.training)

      return x + y if self.is_residual else y

  @staticmethod
  def create(index,
             conv_params,
             regularizer,
             bn_momentum,
             bn_epsilon,
             cnn_dropout_prob,
             training,
             is_residual=True,
             is_causal=False):
    activation_fn = conv_params.get("activation_fn", tf.nn.relu)

    conv = tf.layers.Conv1D(
      name="conv_%d" % index,
      filters=conv_params["num_channels"],
      kernel_size=conv_params["kernel_size"],
      strides=conv_params["stride"],
      padding=conv_params["padding"],
      kernel_regularizer=regularizer
    )

    norm = BatchNorm1D(
      name="bn_%d" % index,
      gamma_regularizer=regularizer,
      momentum=bn_momentum,
      epsilon=bn_epsilon
    )

    dropout = tf.layers.Dropout(
      name="dropout_%d" % index,
      rate=cnn_dropout_prob
    )

    if conv_params.get("is_causal", False):
      is_causal = True

    return ConvBlock("layer_%d" % index, conv, norm, activation_fn, dropout, training, is_residual, is_causal)


class ConvTTSEncoder(Encoder):
  def __init__(self, params, model, name="conv_tts_encoder", mode="train"):
    super(ConvTTSEncoder, self).__init__(params, model, name=name, mode=mode)
    self.training = mode == "train"
    self.layers = []

  def _build_layers(self):
    regularizer = self._params.get("regularizer", None)

    embedding = embedding_layer.EmbeddingSharedWeights(
      vocab_size=self._params["src_vocab_size"],
      hidden_size=self._params["embedding_size"],
      pad_vocab_to_eight=self.params.get("pad_embeddings_2_eight", False),
      regularizer=regularizer
    )
    self.layers.append(embedding)

    cnn_dropout_prob = self._params.get("cnn_dropout_prob", 0.5)
    bn_momentum = self._params.get("bn_momentum", 0.95)
    bn_epsilon = self._params.get("bn_epsilon", -1e8)

    for index, params in enumerate(self._params["conv_layers"]):
      layer = ConvBlock.create(
        index=index,
        conv_params=params,
        regularizer=regularizer,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        cnn_dropout_prob=cnn_dropout_prob,
        training=self.training
      )

      self.layers.append(layer)

    linear_projection = tf.layers.Dense(
      name="linear_projection",
      units=self._params["output_size"],
      use_bias=False,
      kernel_regularizer=regularizer
    )
    self.layers.append(linear_projection)

  def _encode(self, input_dict):
    if not self.layers:
      self._build_layers()
    x = input_dict["source_tensors"][0]
    text_len = input_dict["source_tensors"][1]

    # Apply all layers
    y = x
    for layer in self.layers:
      y = layer(y)

    inputs_attention_bias = utils.get_padding_bias(x)

    return {
      "outputs": y,
      "inputs_attention_bias": inputs_attention_bias,
      "src_lengths": text_len
    }

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
      "src_vocab_size": int,
      "embedding_size": int,
      "output_size": int,
      "conv_layers": list
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
      "pad_embeddings_2_eight": bool,
      "regularizer": None,
      "bn_momentum": float,
      "bn_epsilon": float,
      "cnn_dropout_prob": float,
      "norm_type": str
    })
