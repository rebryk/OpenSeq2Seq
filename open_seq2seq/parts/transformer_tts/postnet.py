import tensorflow as tf

from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv


class Postnet:
  """
  Decoder Post-net for TTS Transformer.

  Consists of convolution layers.
  """

  def __init__(self, conv_layers, num_audio_features, dropout_keep_prob, regularizer,
               training, data_format, bn_momentum, bn_epsilon):
    self.conv_layers = conv_layers
    self.num_audio_features = num_audio_features
    self.dropout_keep_prob = dropout_keep_prob
    self.regularizer = regularizer
    self.training = training
    self.data_format = data_format
    self.bn_momentum = bn_momentum
    self.bn_epsilon = bn_epsilon

  def __call__(self, decoder_output):
    top_layer = decoder_output

    for i, conv_params in enumerate(self.conv_layers):
      ch_out = conv_params["num_channels"]
      kernel_size = conv_params["kernel_size"]
      strides = conv_params["stride"]
      padding = conv_params["padding"]
      activation_fn = conv_params["activation_fn"]

      if ch_out == -1:
          ch_out = self.num_audio_features

      top_layer = conv_bn_actv(
        layer_type="conv1d",
        name="conv{}".format(i + 1),
        inputs=top_layer,
        filters=ch_out,
        kernel_size=kernel_size,
        activation_fn=activation_fn,
        strides=strides,
        padding=padding,
        regularizer=self.regularizer,
        training=self.training,
        data_format=self.data_format,
        bn_momentum=self.bn_momentum,
        bn_epsilon=self.bn_epsilon
      )

      top_layer = tf.layers.dropout(top_layer, rate=1.0 - self.dropout_keep_prob, training=self.training)

    return top_layer


class MagSpecPostnet:
  def __init__(self, params, n_feats, exp_mag, training):
    self.params = params
    self.n_feats = n_feats
    self.exp_mag = exp_mag
    self.training = training

  def __call__(self, spectrogram_prediction):
    mag_spec_prediction = spectrogram_prediction

    mag_spec_prediction = conv_bn_actv(
      layer_type="conv1d",
      name="conv_0",
      inputs=mag_spec_prediction,
      filters=256,
      kernel_size=4,
      activation_fn=tf.nn.relu,
      strides=1,
      padding="SAME",
      regularizer=self.params.get("regularizer", None),
      training=self.training,
      data_format=self.params.get("postnet_data_format", "channels_last"),
      bn_momentum=self.params.get("postnet_bn_momentum", 0.1),
      bn_epsilon=self.params.get("postnet_bn_epsilon", 1e-5),
    )

    mag_spec_prediction = conv_bn_actv(
      layer_type="conv1d",
      name="conv_1",
      inputs=mag_spec_prediction,
      filters=512,
      kernel_size=4,
      activation_fn=tf.nn.relu,
      strides=1,
      padding="SAME",
      regularizer=self.params.get("regularizer", None),
      training=self.training,
      data_format=self.params.get("postnet_data_format", "channels_last"),
      bn_momentum=self.params.get("postnet_bn_momentum", 0.1),
      bn_epsilon=self.params.get("postnet_bn_epsilon", 1e-5),
    )

    if self.exp_mag:
      mag_spec_prediction = tf.exp(mag_spec_prediction)

    mag_spec_prediction = tf.layers.conv1d(mag_spec_prediction, self.n_feats, 1, name="post_net_proj", use_bias=False)

    return mag_spec_prediction
