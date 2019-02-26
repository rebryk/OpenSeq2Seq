import tensorflow as tf


_NEG_INF = -1e9


def get_encoder_self_attention_bias(inputs, size):
  """Calculate bias for encoder that allows to see just `size` symbols from the both sides.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """

  length = tf.shape(inputs)[1]

  with tf.name_scope("encoder_self_attention_bias"):
    valid_locs = tf.matrix_band_part(tf.ones([length, length]), size, size)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    encoder_bias = _NEG_INF * (1.0 - valid_locs)

  return encoder_bias
