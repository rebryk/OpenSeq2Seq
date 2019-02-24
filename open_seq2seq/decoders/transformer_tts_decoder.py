# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from open_seq2seq.decoders.tacotron2_decoder import Prenet
from open_seq2seq.parts.transformer import utils
from open_seq2seq.parts.transformer_tts import TransformerDecoder, Postnet, MagSpecPostnet
from .decoder import Decoder


class TransformerTTSDecoder(Decoder):
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
      "layer_postprocess_dropout": float,
      "num_hidden_layers": int,
      "hidden_size": int,
      "num_heads": int,
      "attention_dropout": float,
      "relu_dropout": float,
      "filter_size": int
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
      "regularizer": None,  # any valid TensorFlow regularizer
      "regularizer_params": dict,
      "initializer": None,  # any valid TensorFlow initializer
      "initializer_params": dict,
      "enable_postnet": bool,
      "postnet_keep_dropout_prob": float,
      "postnet_data_format": str,
      "postnet_conv_layers": list,
      "postnet_bn_momentum": float,
      "postnet_bn_epsilon": float,
      "enable_prenet": bool,
      "prenet_layers": int,
      "prenet_units": int,
      "prenet_activation": None,
      "parallel_iterations": int
    })

  def _cast_types(self, input_dict):
    return input_dict

  def __init__(self, params, model, name="transformer_tts_decoder", mode="train"):
    super(TransformerTTSDecoder, self).__init__(params, model, name, mode)

    self.model = model
    self._mode = mode
    self.training = (mode == "train")
    self.n_feats = self.model.get_data_layer().params["num_audio_features"]

    if "both" in self.model.get_data_layer().params["output_type"]:
      self.both = True
      if not self.params.get("enable_postnet", True):
        raise ValueError("postnet must be enabled for both mode")
    else:
      self.both = False

    if not self.params.get("enable_postnet", True):
      raise ValueError("You should use pre-net!")

    if self.params.get("enable_postnet", True):
      if "postnet_conv_layers" not in self.params:
        raise ValueError("postnet_conv_layers must be passed from config file if postnet is enabled")

    if not self.params.get("enable_postnet", True):
      raise ValueError("You should use post-net!")

    self.features_count = sum(self.n_feats.values()) if self.both else self.n_feats

    self.regularizer = self.params.get("regularizer", None)

    self.prenet = None
    self.decoder = None
    self.postnet = None
    self.output_projection_layer = None
    self.stop_token_projection_layer = None
    self.mag_spec_postnet = None
    self.output_normalization = None

  def _regularize(self):
    vars_to_regularize = []
    vars_to_regularize += self.output_projection_layer.trainable_variables
    vars_to_regularize += self.stop_token_projection_layer.trainable_variables

    for weights in vars_to_regularize:
      if "bias" not in weights.name:
        if weights.dtype.base_dtype == tf.float16:
          tf.add_to_collection("REGULARIZATION_FUNCTIONS", (weights, self.regularizer))
        else:
          tf.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, self.regularizer(weights))

    # TODO: why do we add regularization for pre-net and don"t do it for post-net?
    if self.params.get("enable_prenet", True):
      self.prenet.add_regularization(self.regularizer)

  def decode_pass(self, decoder_inputs, encoder_outputs, encoder_decoder_attention_bias, sequence_lengths=None):
    batch_size = tf.shape(decoder_inputs)[0]
    length = tf.shape(decoder_inputs)[1]

    with tf.name_scope("add_pos_encoding"):
      position_encoding = tf.cast(utils.get_position_encoding(length, self.features_count + 1), self.params["dtype"])

      if self.features_count % 2 == 1:
        position_encoding = position_encoding[:, :-1]

      decoder_inputs += position_encoding

    decoder_inputs = self.prenet(decoder_inputs)
    decoder_self_attention_bias = utils.get_decoder_self_attention_bias(length)

    decoder_output = self.decoder(
      decoder_inputs=decoder_inputs,
      encoder_outputs=encoder_outputs,
      decoder_self_attention_bias=decoder_self_attention_bias,
      attention_bias=encoder_decoder_attention_bias,
    )

    decoder_spec_output = self.output_projection_layer(decoder_output)
    stop_token_logits = self.stop_token_projection_layer(decoder_spec_output)
    spectrogram_prediction = decoder_spec_output + self.postnet(decoder_spec_output)

    if self.mag_spec_postnet:
      mag_spec_prediction = self.mag_spec_postnet(spectrogram_prediction)
    else:
      mag_spec_prediction = tf.zeros([batch_size, batch_size, batch_size])

    if sequence_lengths is None:
      sequence_lengths = tf.zeros([batch_size])

    return {
        "spec": decoder_spec_output,
        "post_net_spec": spectrogram_prediction,
        "alignments": tf.zeros([batch_size, batch_size, batch_size]),
        "stop_token_logits": stop_token_logits,
        "lengths": sequence_lengths,
        "mag_spec": mag_spec_prediction
    }

  @staticmethod
  def _convert_outputs(outputs):
    return {
      "outputs": [
        outputs["spec"], outputs["post_net_spec"], outputs["alignments"],
        tf.sigmoid(outputs["stop_token_logits"]), outputs["lengths"], outputs["mag_spec"]
      ],
      "stop_token_logits": outputs["stop_token_logits"]
    }

  def _decode(self, input_dict):
    if "target_tensors" in input_dict:
      targets = input_dict["target_tensors"][0]
    else:
      targets = None

    encoder_outputs = input_dict["encoder_output"]["outputs"]
    inputs_attention_bias = input_dict["encoder_output"]["inputs_attention_bias"]

    spec_length = None

    if self.mode == "train" or self.mode == "eval":
      spec_length = input_dict["target_tensors"][2] if "target_tensors" in input_dict else None

    num_audio_features = self.n_feats["mel"] if self.both else self.n_feats

    self.prenet = Prenet(
      self.params.get("prenet_units", 256),
      self.params.get("prenet_layers", 2),
      self.params.get("prenet_activation", tf.nn.relu),
      self.params["dtype"]
    )

    self.decoder = TransformerDecoder(self.params, self.training)

    # The same decoder post-net is used in Tacotron2
    self.postnet = Postnet(
      conv_layers=self.params["postnet_conv_layers"],
      num_audio_features=num_audio_features,
      dropout_keep_prob=self.params.get("postnet_keep_dropout_prob", 0.5),
      regularizer=self.params.get("regularizer", None),
      training=self.training,
      data_format=self.params.get("postnet_data_format", "channels_last"),
      bn_momentum=self.params.get("postnet_bn_momentum", 0.1),
      bn_epsilon=self.params.get("postnet_bn_epsilon", 1e-5)
    )

    self.output_projection_layer = tf.layers.Dense(name="output_proj", units=num_audio_features, use_bias=True)
    self.stop_token_projection_layer = tf.layers.Dense(name="stop_token_proj", units=1, use_bias=True)

    if self.both:
      self.mag_spec_postnet = MagSpecPostnet(
        self.params,
        self.n_feats["magnitude"],
        self.model.get_data_layer()._exp_mag,
        self.training
      )

    if self.regularizer and self.training:
      self._regularize()

    if not self.training:
      return self._predict(encoder_outputs, inputs_attention_bias, spec_length)

    return self._train(targets, encoder_outputs, inputs_attention_bias, spec_length)

  def _train(self, targets, encoder_outputs, encoder_decoder_attention_bias, sequence_lengths):
    # TODO: verify that it works properly
    # Shift targets to the right, and remove the last element
    with tf.name_scope("shift_targets"):
      decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

    outputs = self.decode_pass(
      decoder_inputs,
      encoder_outputs,
      encoder_decoder_attention_bias,
      sequence_lengths
    )

    return self._convert_outputs(outputs)


  def _inference_initial_state(self, encoder_outputs, encoder_decoder_attention_bias):
    # TODO: change channel size
    # TODO: make mag spec optional

    batch_size = tf.shape(encoder_outputs)[0]

    state = {
      "iteration": tf.constant(0),
      "inputs": tf.zeros([batch_size, 1, self.features_count]),
      "finished": tf.cast(tf.zeros([batch_size]), tf.bool),
      "outputs": {
        "spec": tf.zeros([batch_size, 0, 80]),
        "post_net_spec": tf.zeros([batch_size, 0, 80]),
        "alignments": tf.zeros([batch_size, batch_size, batch_size]),
        "stop_token_logits": tf.zeros([batch_size, 0, 1]),
        "lengths": tf.zeros([batch_size], dtype=tf.int32),
        "mag_spec": tf.zeros([batch_size, 0, 513])
      },
      "encoder_outputs": encoder_outputs,
      "encoder_decoder_attention_bias": encoder_decoder_attention_bias
    }

    state_shape_invariants = {
      "iteration": tf.TensorShape([]),
      "inputs": tf.TensorShape([None, None, self.features_count]),
      "finished": tf.TensorShape([None]),
      "outputs": {
        "spec": tf.TensorShape([None, None, 80]),
        "post_net_spec": tf.TensorShape([None, None, 80]),
        "alignments": tf.TensorShape([None, None, None]),
        "stop_token_logits": tf.TensorShape([None, None, 1]),
        "lengths": tf.TensorShape([None]),
        "mag_spec": tf.TensorShape([None, None, 513])
      },
      "encoder_outputs": encoder_outputs.shape,
      "encoder_decoder_attention_bias": encoder_decoder_attention_bias.shape
    }

    return state, state_shape_invariants

  def _inference_cond(self, state):
    all_finished = math_ops.reduce_all(state["finished"])
    return tf.logical_not(all_finished)

  def _inference_step(self, state):
    # TODO: calculate sequence lengths
    # TODO: calculate alignments
    inputs = state["inputs"]
    encoder_outputs = state["encoder_outputs"]
    encoder_decoder_attention_bias = state["encoder_decoder_attention_bias"]

    outputs = self.decode_pass(
      inputs,
      encoder_outputs,
      encoder_decoder_attention_bias
    )

    spectrogram_prediction = outputs["post_net_spec"][:, -1:, :]
    mag_spec_prediction = outputs["mag_spec"][:, -1:, :]

    if self.both:
      next_inputs = tf.concat([spectrogram_prediction, mag_spec_prediction], -1)
    else:
      next_inputs = spectrogram_prediction

    # Set zero if sequence is finished
    next_inputs = tf.where(state["finished"], tf.zeros_like(next_inputs), next_inputs)
    next_inputs = tf.concat([inputs, next_inputs], 1)

    # Update lengths
    lengths = state["outputs"]["lengths"]
    lengths = tf.where(state["finished"], lengths, lengths + 1)
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
    finished = tf.reshape(tf.cast(tf.round(stop_prediction), tf.bool), [-1])
    stop_token_logits = tf.concat([state["outputs"]["stop_token_logits"], stop_token_logits], 1)
    outputs["stop_token_logits"] = stop_token_logits

    state["iteration"] = state["iteration"] + 1
    state["inputs"] = next_inputs
    state["finished"] = finished
    state["outputs"] = outputs

    return state

  def _predict(self, encoder_outputs, encoder_decoder_attention_bias, sequence_lengths):
    # TODO: choose better value
    if sequence_lengths is None:
      maximum_iterations = 1000
    else:
      maximum_iterations = tf.reduce_max(sequence_lengths)

    state, state_shape_invariants = self._inference_initial_state(encoder_outputs, encoder_decoder_attention_bias)

    state = tf.while_loop(
      cond=self._inference_cond,
      body=self._inference_step,
      loop_vars=[state],
      shape_invariants=state_shape_invariants,
      back_prop=False,
      maximum_iterations=maximum_iterations,
      parallel_iterations=self.params.get("parallel_iterations", 32)
    )

    return self._convert_outputs(state["outputs"])

