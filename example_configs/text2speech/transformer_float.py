# pylint: skip-file
import os

import tensorflow as tf

from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.decoders import TransformerTTSDecoder
from open_seq2seq.encoders import TransformerTTSEncoder
from open_seq2seq.losses import TacotronLoss
from open_seq2seq.models import Text2Speech
from open_seq2seq.optimizers.lr_policies import exp_decay

base_model = Text2Speech

dataset = "LJ"
dataset_location = "/data/LJSpeech"
output_type = "both"

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
  batch_size = 8
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

d_model = 512
num_layers = 6

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "max_steps": 100000,
  "bench_start": 0,

  "num_gpus": 8,
  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 500,
  "save_to_tensorboard": True,
  "logdir": "result/transformer-LJ-float",
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
    'scale': 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": TransformerTTSEncoder,
  "encoder_params": {
    "cnn_dropout_prob": 0.5,
    "conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      }
    ],
    "activation_fn": tf.nn.relu,
    "data_format": "channels_last",

    "src_vocab_size": 94,
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 4 * d_model,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "pad_embeddings_2_eight": True,
    "remove_padding": True,
  },

  "decoder": TransformerTTSDecoder,
  "decoder_params": {
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 512,

    'enable_postnet': True,
    "postnet_keep_dropout_prob": 0.5,
    "postnet_data_format": "channels_last",
    "postnet_conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": -1, "padding": "SAME",
        "activation_fn": tf.nn.relu
      }
    ],

    "layer_postprocess_dropout": 0.1,
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "filter_size": 4 * d_model,
  },

  "loss": TacotronLoss,
  "loss_params": {
    "use_mask": True
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset": dataset,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location': dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min": data_min,
    "mel_type": 'htk',
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
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, "test.csv"),
    ],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
  },
}
