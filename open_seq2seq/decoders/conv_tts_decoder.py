import tensorflow as tf
from tensorflow.python.ops import math_ops

from open_seq2seq.encoders.conv_tts_encoder import ConvBlock
from open_seq2seq.parts.transformer import attention_layer
from open_seq2seq.parts.transformer import utils
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, LayerNormalization
from open_seq2seq.parts.transformer.ffn_layer import FeedFowardNetwork
from .decoder import Decoder


class Prenet:
  def __init__(self,
               n_layers,
               hidden_size,
               activation_fn,
               dropout=0.5,
               regularizer=None,
               training=True,
               dtype=None,
               name="prenet"):
    self.name = name
    self.layers = []
    self.dropout = dropout
    self.training = training

    for i in range(n_layers):
      layer = tf.layers.Dense(
        name="layer_%d" % i,
        units=hidden_size,
        use_bias=True,
        activation=activation_fn,
        kernel_regularizer=regularizer,
        dtype=dtype
      )
      self.layers.append(layer)

  def __call__(self, x):
    # TODO: do we need to use dropout here?
    with tf.variable_scope(self.name):
      for layer in self.layers:
        x = tf.layers.dropout(layer(x), rate=self.dropout, training=self.training)

      return x


class AttentionBlock:
  def __init__(self,
               hidden_size,
               attention_dropout,
               layer_postprocess_dropout,
               training,
               cnn_dropout_prob,
               regularizer=None,
               conv_params=None,
               pos_encoding=False,
               filter_size=None,
               n_heads=1,
               window_size=None,
               back_step_size=None,
               name="attention_block"):
    self.name = name

    self.conv = None

    if conv_params:
      self.conv = ConvBlock.create(
        index=0,
        conv_params=conv_params,
        regularizer=regularizer,
        bn_momentum=0.95,
        bn_epsilon=1e-8,
        cnn_dropout_prob=cnn_dropout_prob,
        training=training
      )
      self.conv.name = "conv"

    attention = attention_layer.Attention(
      hidden_size=hidden_size,
      num_heads=n_heads,
      attention_dropout=attention_dropout,
      regularizer=regularizer,
      train=training,
      pos_encoding=pos_encoding,
      window_size=window_size,
      back_step_size=back_step_size
    )

    if filter_size is not None:
      feed_forward = FeedFowardNetwork(
        hidden_size=hidden_size,
        filter_size=filter_size,
        relu_dropout=0,
        regularizer=regularizer,
        train=training
      )
    else:
      feed_forward = tf.layers.Dense(
        units=hidden_size,
        use_bias=True,
        kernel_regularizer=regularizer
      )

    wrapper_params = {
      "hidden_size": hidden_size,
      "layer_postprocess_dropout": layer_postprocess_dropout
    }

    self.attention = PrePostProcessingWrapper(
      layer=attention,
      params=wrapper_params,
      training=training
    )

    self.feed_forward = PrePostProcessingWrapper(
      layer=feed_forward,
      params=wrapper_params,
      training=training
    )

    self.output_normalization = LayerNormalization(hidden_size)

  def __call__(self, decoder_inputs, encoder_outputs, attention_bias, positions=None):
    with tf.variable_scope(self.name):
      y = decoder_inputs

      if self.conv:
        y = self.conv(y)

      cnt = tf.ones([1, tf.shape(y)[1], 1], dtype=tf.float32)
      cnt = tf.cumsum(cnt, axis=1)
      y = tf.cumsum(y, axis=1) / cnt

      with tf.variable_scope("attention"):
        y = self.attention(y, encoder_outputs, attention_bias, positions=positions)

      with tf.variable_scope("feed_forward"):
        y = self.feed_forward(y)

      return self.output_normalization(y)


class ConvTTSDecoder(Decoder):
  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Decoder.get_required_params(), **{
      "prenet_layers": int,
      "prenet_hidden_size": int,
      "hidden_size": int,
      "pre_conv_layers": list,
      "post_conv_layers": list,
      "attention_dropout": float,
      "layer_postprocess_dropout": float
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
    return dict(Decoder.get_optional_params(), **{
      "prenet_activation_fn": None,
      "prenet_dropout": float,
      "prenet_use_inference_dropout": bool,
      "cnn_dropout_prob": float,
      "bn_momentum": float,
      "bn_epsilon": float,
      "reduction_factor": int,
      "attention_layers": int,
      "self_attention_conv_params": None,
      "attention_pos_encoding": bool,
      "attention_heads": int,
      "disable_attention": bool,
      "filter_size": int,
      "attention_cnn_dropout_prob": float,
      "scale_positional_encoding": bool,
      "window_size": int,
      "back_step_size": int
    })

  def __init__(self, params, model, name="conv_tts_decoder", mode="train"):
    super(ConvTTSDecoder, self).__init__(params, model, name, mode)

    data_layer_params = model.get_data_layer().params
    n_feats = data_layer_params["num_audio_features"]
    use_mag = "both" in data_layer_params["output_type"]

    self.training = mode == "train"
    self.prenet = None
    self.pre_conv_layers = []
    self.linear_projection = None
    self.attentions = []
    self.attention_pos_encoding = self._params.get("attention_pos_encoding", False)
    self.scale_positional_encoding = self._params.get("scale_positional_encoding", False)
    self.enable_attention = not self._params.get("disable_attention", False)
    self.post_conv_layers = []
    self.stop_token_projection_layer = None
    self.mel_projection_layer = None

    self.n_mel = n_feats["mel"] if use_mag else n_feats
    self.n_mag = n_feats["mag"] if use_mag else None
    self.reduction_factor = params.get("reduction_factor", 1)

  def _build_layers(self):
    regularizer = self._params.get("regularizer", None)

    # TODO: dropout during inference?
    self.prenet = Prenet(
      n_layers=self._params["prenet_layers"],
      hidden_size=self._params["prenet_hidden_size"],
      activation_fn=self._params.get("prenet_activation_fn", tf.nn.relu),
      dropout=self._params.get("prenet_dropout", 0.5),
      regularizer=regularizer,
      training=self.training or self._params.get("prenet_use_inference_dropout", False),
      dtype=self._params["dtype"]
    )

    cnn_dropout_prob = self._params.get("cnn_dropout_prob", 0.5)
    bn_momentum = self._params.get("bn_momentum", 0.95)
    bn_epsilon = self._params.get("bn_epsilon", -1e8)

    for index, params in enumerate(self._params["pre_conv_layers"]):
      layer = ConvBlock.create(
        index=index,
        conv_params=params,
        regularizer=regularizer,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        cnn_dropout_prob=cnn_dropout_prob,
        training=self.training
      )
      self.pre_conv_layers.append(layer)

    self.linear_projection = tf.layers.Dense(
      name="linear_projection",
      units=self._params["hidden_size"],
      use_bias=False,
      kernel_regularizer=regularizer,
      dtype=self._params["dtype"]
    )

    n_layers = self._params.get("attention_layers", 1)
    n_heads = self._params.get("attention_heads", 1)
    conv_params = self._params.get("self_attention_conv_params", None)

    for index in range(n_layers):
      attention = AttentionBlock(
        name="attention_block_%d" % index,
        hidden_size=self._params["hidden_size"],
        attention_dropout=self._params["attention_dropout"],
        layer_postprocess_dropout=self._params["layer_postprocess_dropout"],
        regularizer=regularizer,
        training=self.training,
        cnn_dropout_prob=self._params.get("attention_cnn_dropout_prob", 0.5),
        conv_params=conv_params,
        pos_encoding=self.attention_pos_encoding and self.enable_attention,
        filter_size=self._params.get("filter_size", None),
        n_heads=n_heads,
        window_size=self._params.get("window_size", None),
        back_step_size=self._params.get("back_step_size", None)
      )
      self.attentions.append(attention)

    for index, params in enumerate(self._params["post_conv_layers"]):
      layer = ConvBlock.create(
        index=index,
        conv_params=params,
        regularizer=regularizer,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        cnn_dropout_prob=cnn_dropout_prob,
        training=self.training
      )
      self.post_conv_layers.append(layer)

    # TODO: Do we need to use bias?
    self.mel_projection_layer = tf.layers.Dense(
      name="mel_projection",
      units=self.n_mel * self.reduction_factor,
      use_bias=True,
      kernel_regularizer=regularizer
    )

    if self.n_mag:
      # TODO: implement mag predeciotn
      pass

    # TODO: Do we need to use bias?
    self.stop_token_projection_layer = tf.layers.Dense(
      name="stop_token_projection",
      units=1 * self.reduction_factor,
      use_bias=True,
      kernel_regularizer=regularizer
    )

  def _decode(self, input_dict):
    self._build_layers()

    if "target_tensors" in input_dict:
      targets = input_dict["target_tensors"][0]
    else:
      targets = None

    encoder_outputs = input_dict["encoder_output"]["outputs"]
    inputs_attention_bias = input_dict["encoder_output"]["inputs_attention_bias"]

    spec_length = None

    if self.mode == "train" or self.mode == "eval":
      spec_length = input_dict["target_tensors"][2] if "target_tensors" in input_dict else None

    if self.training:
      return self._train(targets, encoder_outputs, inputs_attention_bias, spec_length)

    return self._infer(encoder_outputs, inputs_attention_bias, spec_length)

  def _decode_pass(self,
                   decoder_inputs,
                   encoder_outputs,
                   enc_dec_attention_bias,
                   sequence_lengths=None,
                   alignment_positions=None):
    # TODO: simplify
    # shape = tf.shape(decoder_inputs)
    # decoder_inputs = tf.Print(decoder_inputs, [tf.shape(decoder_inputs)], summarize=10)
    # y = tf.reshape(decoder_inputs, [shape[0], shape[1] * self.reduction_factor, self.n_mel])
    # y = self.prenet(y)
    # y = self._collapse(y, self.n_mel, self.reduction_factor)
    y = self.prenet(decoder_inputs)

    with tf.variable_scope("pre_conv"):
      for layer in self.pre_conv_layers:
        y = layer(y)

    y = self.linear_projection(y)

    if not self.attention_pos_encoding and self.enable_attention:
      with tf.variable_scope("decoder_pos_encoding"):
        pos_encoding = self._positional_encoding(y, self.params["dtype"])

        if self.scale_positional_encoding:
          scale = tf.get_variable(
            name="decoder_pos_encoding_scale",
            shape=[1],
            trainable=True,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32
          )
          pos_encoding *= scale

        y += pos_encoding

    if not self.attention_pos_encoding and self.enable_attention:
      with tf.variable_scope("encoder_pos_encoding"):
        pos_encoding = self._positional_encoding(encoder_outputs, self.params["dtype"])

        if self.scale_positional_encoding:
          scale = tf.get_variable(
            name="encoder_pos_encoding_scale",
            shape=[1],
            trainable=True,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32
          )
          pos_encoding *= scale

        encoder_outputs += pos_encoding

    for i, attention in enumerate(self.attentions):
      positions = alignment_positions[i, :, :, :] if alignment_positions is not None else None
      y = attention(y, encoder_outputs, enc_dec_attention_bias, positions=positions)

    with tf.variable_scope("post_conv"):
      for layer in self.post_conv_layers:
        y = layer(y)

    with tf.variable_scope("mag_projection"):
      batch_size = tf.shape(y)[0]

      if self.n_mag:
        # TODO: mag spec
        mag_spec = tf.zeros([batch_size, batch_size, batch_size * self.reduction_factor])
      else:
        mag_spec = tf.zeros([batch_size, batch_size, batch_size * self.reduction_factor])

    mel_spec = self.mel_projection_layer(y)
    stop_token_logits = self.stop_token_projection_layer(y)

    if sequence_lengths is None:
      sequence_lengths = tf.zeros([batch_size])

    return {
      "spec": mel_spec,
      "post_net_spec": mel_spec,
      "alignments": None,
      "stop_token_logits": stop_token_logits,
      "lengths": sequence_lengths,
      "mag_spec": mag_spec
    }

  def _train(self, targets, encoder_outputs, enc_dec_attention_bias, sequence_lengths):
    # Shift targets to the right, and remove the last element
    with tf.name_scope("shift_targets"):
      targets = targets[:, :, :self.n_mel]
      targets = self._collapse(targets, self.n_mel, self.reduction_factor)
      decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

    outputs = self._decode_pass(
      decoder_inputs=decoder_inputs,
      encoder_outputs=encoder_outputs,
      enc_dec_attention_bias=enc_dec_attention_bias,
      sequence_lengths=sequence_lengths
    )

    with tf.variable_scope("alignments"):
      weights = []

      for index in range(len(self.attentions)):
        op = "ForwardPass/conv_tts_decoder/attention_block_%d/attention/attention/attention_weights" % index
        weights_operation = tf.get_default_graph().get_operation_by_name(op)
        weight = weights_operation.values()[0]
        weights.append(weight)

      outputs["alignments"] = [tf.stack(weights)]

    return self._convert_outputs(outputs, self.reduction_factor, self._model.params["batch_size_per_gpu"])

  def _infer(self, encoder_outputs, enc_dec_attention_bias, sequence_lengths):
    # TODO: choose better value
    if sequence_lengths is None:
      maximum_iterations = 1000
    else:
      maximum_iterations = tf.reduce_max(sequence_lengths)

    maximum_iterations //= self.reduction_factor

    state, state_shape_invariants = self._inference_initial_state(encoder_outputs, enc_dec_attention_bias)

    state = tf.while_loop(
      cond=self._inference_cond,
      body=self._inference_step,
      loop_vars=[state],
      shape_invariants=state_shape_invariants,
      back_prop=False,
      maximum_iterations=maximum_iterations,
      parallel_iterations=1
    )

    return self._convert_outputs(state["outputs"], self.reduction_factor, self._model.params["batch_size_per_gpu"])

  def _inference_initial_state(self, encoder_outputs, encoder_decoder_attention_bias):
    with tf.variable_scope("inference_initial_state"):
      batch_size = tf.shape(encoder_outputs)[0]
      num_mag_features = self.n_mag or batch_size
      n_layers = self._params.get("attention_layers", 1)
      n_heads = self._params.get("attention_heads", 1)

      state = {
        "iteration": tf.constant(0),
        "inputs": tf.zeros([batch_size, 1, self.n_mel * self.reduction_factor]),
        "finished": tf.cast(tf.zeros([batch_size]), tf.bool),
        "alignment_positions": tf.zeros([n_layers, batch_size, n_heads, 1], dtype=tf.int32),
        "outputs": {
          "spec": tf.zeros([batch_size, 0, self.n_mel * self.reduction_factor]),
          "post_net_spec": tf.zeros([batch_size, 0, self.n_mel * self.reduction_factor]),
          "alignments": [
            tf.zeros([0, 0, 0, 0, 0])
          ],
          "stop_token_logits": tf.zeros([batch_size, 0, 1 * self.reduction_factor]),
          "lengths": tf.zeros([batch_size], dtype=tf.int32),
          "mag_spec": tf.zeros([batch_size, 0, num_mag_features * self.reduction_factor])
        },
        "encoder_outputs": encoder_outputs,
        "encoder_decoder_attention_bias": encoder_decoder_attention_bias
      }

      state_shape_invariants = {
        "iteration": tf.TensorShape([]),
        "inputs": tf.TensorShape([None, None, self.n_mel * self.reduction_factor]),
        "finished": tf.TensorShape([None]),
        "alignment_positions": tf.TensorShape([n_layers, None, n_heads, None]),
        "outputs": {
          "spec": tf.TensorShape([None, None, self.n_mel * self.reduction_factor]),
          "post_net_spec": tf.TensorShape([None, None, self.n_mel * self.reduction_factor]),
          "alignments": [
            tf.TensorShape([None, None, None, None, None]),
          ],
          "stop_token_logits": tf.TensorShape([None, None, 1 * self.reduction_factor]),
          "lengths": tf.TensorShape([None]),
          "mag_spec": tf.TensorShape([None, None, None])
        },
        "encoder_outputs": encoder_outputs.shape,
        "encoder_decoder_attention_bias": encoder_decoder_attention_bias.shape
      }

      return state, state_shape_invariants

  def _inference_cond(self, state):
    with tf.variable_scope("inference_cond"):
      all_finished = math_ops.reduce_all(state["finished"])
      return tf.logical_not(all_finished)

  def _inference_step(self, state):
    decoder_inputs = state["inputs"]
    encoder_outputs = state["encoder_outputs"]
    enc_dec_attention_bias = state["encoder_decoder_attention_bias"]
    alignment_positions = state["alignment_positions"]

    outputs = self._decode_pass(
      decoder_inputs=decoder_inputs,
      encoder_outputs=encoder_outputs,
      enc_dec_attention_bias=enc_dec_attention_bias,
      alignment_positions=alignment_positions
    )

    with tf.variable_scope("inference_step"):
      # We don't have post-net, thus spec = post_net_spec
      next_inputs = outputs["post_net_spec"][:, -1:, :]

      # Set zero if sequence is finished
      next_inputs = tf.where(state["finished"], tf.zeros_like(next_inputs), next_inputs)
      next_inputs = tf.concat([decoder_inputs, next_inputs], 1)

      # Update lengths
      lengths = state["outputs"]["lengths"]
      lengths = tf.where(state["finished"], lengths, lengths + 1 * self.reduction_factor)
      outputs["lengths"] = lengths

      # Update spec, post_net_spec and mag_spec
      for key in ["spec", "post_net_spec", "mag_spec"]:
        output = outputs[key][:, -1:, :]
        output = tf.where(state["finished"], tf.zeros_like(output), output)
        outputs[key] = tf.concat([state["outputs"][key], output], 1)

      # Update stop token logits
      stop_token_logits = outputs["stop_token_logits"][:, -1:, :]
      stop_token_logits = tf.where(
        state["finished"],
        tf.zeros_like(stop_token_logits),
        stop_token_logits
      )
      stop_prediction = tf.sigmoid(stop_token_logits)
      stop_prediction = tf.reduce_max(stop_prediction, axis=-1)

      # TODO: uncomment next line if you want to use stop token predictions
      # finished = tf.reshape(tf.cast(tf.round(stop_prediction), tf.bool), [-1])
      finished = tf.reshape(tf.cast(tf.round(tf.zeros_like(stop_prediction)), tf.bool), [-1])

      stop_token_logits = tf.concat([state["outputs"]["stop_token_logits"], stop_token_logits], 1)
      outputs["stop_token_logits"] = stop_token_logits

      with tf.variable_scope("alignments"):
        forward = "ForwardPass" if self.mode == "infer" else "ForwardPass_1"
        weights = []

        for index in range(len(self.attentions)):
          op = forward + "/conv_tts_decoder/while/attention_block_%d/attention/attention/attention_weights" % index
          weights_operation = tf.get_default_graph().get_operation_by_name(op)
          weight = weights_operation.values()[0]
          weights.append(weight)

        weights = tf.stack(weights)
        outputs["alignments"] = [weights]

      alignment_positions = tf.argmax(weights, axis=-1, output_type=tf.int32)[:, :, :, -1:]
      state["alignment_positions"] = tf.concat([state["alignment_positions"], alignment_positions], axis=-1)

      state["iteration"] = state["iteration"] + 1
      state["inputs"] = next_inputs
      state["finished"] = finished
      state["outputs"] = outputs

    return state

  @staticmethod
  def _collapse(values, last_dim, reduction_factor):
    shape = tf.shape(values)
    values = tf.reshape(values, [shape[0], shape[1] // reduction_factor, last_dim * reduction_factor])
    return values

  @staticmethod
  def _extend(values, reduction_factor):
    shape = tf.shape(values)
    values = tf.reshape(values, [shape[0], shape[1] * reduction_factor, shape[2] // reduction_factor])
    return values

  @staticmethod
  def _positional_encoding(x, dtype):
    length = tf.shape(x)[1]
    features_count = tf.shape(x)[2]
    features_count_even = features_count if (features_count % 2 == 0) else (features_count + 1)
    position_encoding = tf.cast(utils.get_position_encoding(length, features_count_even), dtype)
    position_encoding = position_encoding[:, :features_count]
    return position_encoding

  @staticmethod
  def _convert_outputs(outputs, reduction_factor, batch_size):
    with tf.variable_scope("output_converter"):
      for key in ["spec", "post_net_spec", "stop_token_logits", "mag_spec"]:
        outputs[key] = ConvTTSDecoder._extend(outputs[key], reduction_factor)

      alignments = [[outputs["alignments"][it][:, sample, :, :, :] for it in range(1)] for sample in range(batch_size)]

      return {
        "outputs": [
          outputs["spec"], outputs["post_net_spec"], alignments,
          tf.sigmoid(outputs["stop_token_logits"]), outputs["lengths"], outputs["mag_spec"]
        ],
        "stop_token_logits": outputs["stop_token_logits"]
      }
