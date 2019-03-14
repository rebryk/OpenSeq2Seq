# pylint: skip-file
import os

import tensorflow as tf

from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.decoders import ConvTTSDecoder
from open_seq2seq.encoders import ConvTTSEncoder
from open_seq2seq.losses import TransformerTTSLoss
from open_seq2seq.models import Text2Speech
from open_seq2seq.optimizers.lr_policies import exp_decay

base_model = Text2Speech

dataset = "LJ"
dataset_location = "/data/LJSpeech"
output_type = "mel"

if dataset == "MAILABS":
  trim = True
  mag_num_feats = 401
  train = "train.csv"
  val = "val.csv"
  batch_size = 32
elif dataset == "LJ":
  trim = False
  mag_num_feats = 513
  train = "train.csv"
  val = "test.csv"
  batch_size = 64
else:
  raise ValueError("Unknown dataset")

exp_mag = False
if output_type == "magnitude":
  num_audio_features = mag_num_feats
  data_min = 1e-5
elif output_type == "mel":
  num_audio_features = 80
  data_min = 1e-2
elif output_type == "both":
  num_audio_features = {
      "mel": 80,
      "magnitude": mag_num_feats
  }
  data_min = {
      "mel": 1e-2,
      "magnitude": 1e-5,
  }
  exp_mag = True
else:
  raise ValueError("Unknown param for output_type")

num_gpus = 8

reduction_factor = 8
encoder_window_size = 2
decoder_window_size = 20 // reduction_factor

encoder_hidden_size = 256
decoder_hidden_size = 512

num_heads = 8
num_layers = 4

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "max_steps": 1000000,
  "bench_start": 0,

  "num_gpus": num_gpus,
  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 1000,
  "print_loss_steps": 1000,
  "print_samples_steps": 1000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "save_to_tensorboard": True,
  "logdir": "result/transformer-LJ-float-65",
  "max_grad_norm": 1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 20000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 45000,
    "min_lr": 1e-5,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    "scale": 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ["learning_rate", "variables", "gradients", "larc_summaries",
                "variable_norm", "gradient_norm", "global_gradient_norm"],

  "encoder": ConvTTSEncoder,
  "encoder_params": {
    "src_vocab_size": 94,
    "embedding_size": encoder_hidden_size,
    "output_size": encoder_hidden_size,
    "pad_embeddings_2_eight": True,
    "cnn_dropout_prob": 0.5,
    "conv_layers": [
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      }
    ]
  },

  "decoder": ConvTTSDecoder,
  "decoder_params": {
    "hidden_size": decoder_hidden_size,
    "reduction_factor": reduction_factor,
    "prenet_layers": 2,
    "prenet_hidden_size": decoder_hidden_size,
    "cnn_dropout_prob": 0.5,
    "pre_conv_layers": [],
    "post_conv_layers":
      [
        {
          "kernel_size": [3], "stride": [1],
          "num_channels": decoder_hidden_size, "padding": "VALID",
          "activation_fn": tf.nn.relu
        }
      ] +
      [
        {
          "kernel_size": [3], "stride": [1],
          "num_channels": decoder_hidden_size, "padding": "SAME",
          "activation_fn": tf.nn.relu
        }
      ] * 3,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1
  },

  "loss": TransformerTTSLoss,
  "loss_params": {
    "use_mask": True
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset": dataset,
    # "n_samples_train": 13100,
    "n_samples_eval": 64,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    "dataset_location": dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min": data_min,
    "mel_type": "htk",
    "trim": trim,   
    "duration_max": 1024,
    "duration_min": 24,
    "exp_mag": exp_mag
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, train),
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, val),
    ],
    "duration_max": 10000,
    "duration_min": 0,
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, "infer.csv"),
    ],
    "duration_max": 10000,
    "duration_min": 0,
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "duration_max": 10000,
    "duration_min": 0,
    "shuffle": False,
  },
}