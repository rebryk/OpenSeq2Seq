# Copyright (c) 2018 NVIDIA Corporation
"""
Losses to be used in seq2seq models
"""
from .cross_entropy_loss import CrossEntropyLoss
from .ctc_loss import CTCLoss
from .jca_loss import MultiTaskCTCEntropyLoss
from .sequence_loss import BasicSequenceLoss, CrossEntropyWithSmoothing, \
    PaddedCrossEntropyLossWithSmoothing, BasicSampledSequenceLoss
from .tacotron_loss import TacotronLoss
from .transformer_tts_loss import TransformerTTSLoss
from .wavenet_loss import WavenetLoss
