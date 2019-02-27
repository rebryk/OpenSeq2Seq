import tensorflow as tf


NEG_INF = -1e9


def get_window_attention_bias(n_rows, size, causal=False, n_cols=None):
  if n_cols is None:
    n_cols = n_rows

  with tf.name_scope("window_self_attention_bias"):
    size = tf.minimum(size, n_rows)
    valid_locs = tf.matrix_band_part(tf.ones([n_rows, n_cols]), size, 0 if causal else size)
    valid_locs = tf.reshape(valid_locs, [1, 1, n_rows, n_cols])
    encoder_bias = NEG_INF * (1.0 - valid_locs)

  return encoder_bias
