# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(
      self,
      hidden_size,
      num_heads,
      attention_dropout,
      train,
      mode="loung",
      monotonic=False
  ):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train
    self.mode = mode
    self.monotonic = monotonic

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    if self.mode == "loung":
      # Scale q to prevent the dot product between q and k from growing too large.
      depth = (self.hidden_size // self.num_heads)
      q *= depth ** -0.5

      # Calculate dot product attention
      #logits = tf.matmul(q, k, transpose_b=True)
      #logits += bias
      #weights = tf.nn.softmax(logits, name="attention_weights")
      logits = tf.matmul(q, k, transpose_b=True)
      dtype = logits.dtype
      if dtype != tf.float32:
        # upcast softmax inputs
        logits = tf.cast(x=logits, dtype=tf.float32)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # downcast softmax output
        weights = tf.cast(weights, dtype=dtype)
      else:
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")

        if self.monotonic and self.train:
          init_length = 5
          n_samples = tf.shape(weights)[0]
          n_head = tf.shape(weights)[1]
          n_audio = tf.shape(weights)[2]
          n_text = tf.shape(weights)[3]

          # Create init attention
          mask_one = tf.ones([n_samples, n_head, 1, init_length], dtype=dtype)
          mask_zero = tf.zeros([n_samples, n_head, 1, n_text - init_length], dtype=dtype)
          mask = tf.concat([mask_one, mask_zero], axis=3) / init_length

          weights = tf.concat([mask, weights], axis=2)
          weights = tf.concat([tf.zeros([n_samples, n_head, n_audio + 1, 1]), weights], axis=3)

          # weights = tf.Print(weights, [weights[0][0][1:]], "before cumsum: ", summarize=20)
          weights = tf.cumsum(weights, axis=3)
          # weights = tf.Print(weights, [tf.reduce_max(weights)], "max: ")
          # weights = tf.Print(weights, [tf.reduce_min(weights)], "min: ")

          # weights = tf.Print(weights, [weights[0][0][1:]], "before cumprod: ", summarize=20)
          weights = tf.cumprod(weights, axis=2)
          # weights = tf.Print(weights, [tf.reduce_max(weights)], "max after: ")
          # weights = tf.Print(weights, [tf.reduce_min(weights)], "min after: ")

          # weights = tf.Print(weights, [weights[0][0][1:]], "before sub: ", summarize=20)
          weights = weights[:, :, :, 1:] - weights[:, :, :, :-1]
          weights = weights[:, :, 1:, :]
          # weights = tf.Print(weights, [weights[0][0]], "after sub: ", summarize=20)

          # weights = tf.nn.softmax(weights + 1)

        if self.monotonic and not self.train:
          # TODO: implement
          pass

    elif self.mode == "bahdanau":
      att_v = tf.get_variable(
          "attention_v", [self.hidden_size // self.num_heads], dtype=q.dtype
      )

      # Compute the attention score
      if bias is not None:
        weights = tf.reduce_sum(
            tf.nn.tanh(att_v * tf.nn.tanh(k + q + bias)), 3
        )
      else:
        weights = tf.reduce_sum(
            tf.nn.tanh(att_v * tf.nn.tanh(k + q)), 3
        )
      weights = tf.nn.softmax(weights)
      weights = tf.expand_dims(weights, 2)
    else:
      raise ValueError(
          "Mode for multi-head attention must be either loung for dot-product",
          "attention, or bahdanau for content-based/additive/mlp-base attention"
      )

    if self.train:
      weights = tf.nn.dropout(weights, keep_prob = 1 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
