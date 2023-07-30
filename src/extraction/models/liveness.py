from typing import List

import torch
import torch.nn as nn
from assets.config.config import ExtractorConfig, CNN_CONFIG
from src.extraction.data import Network

__all__ = [
    'LivenessNetwork'
]

"""
# Extracted audio feature : Liveness 
"""

@torch.jit.interface
class ModuleInterface(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ImplementsInterface(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class LivenessNetwork(Network):
    #
    SEQ_DIM = 0
    FEAT_DIM = 1

    DIVIDE = 2

    INPUT_REPRESENTATION = 'spectrogram' # mel

    def __init__(self, extractor_config: ExtractorConfig, input_shape: tuple, seed: int = 0):
        super().__init__()

        self.cnn_config = extractor_config.cnn_config
        self.rnn_config = extractor_config.rnn_config

        self.numoffeature = 1
        self.input_shape = input_shape

        # torch seed initialization
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # build a spatial part of CNNs
        # B x C x H x W : batch x 1 x seq_len x spectra-dim
        feats = 0
        for _cnn_config in self.cnn_config:
            #
            if _cnn_config.type == self.INPUT_REPRESENTATION:
                feats += _cnn_config.numoflayers[-1]
                self.cnn_layers = self._build_conv_layer(cnn_config=_cnn_config)

        self.rnn_cell = self._build_rnn_layer(feats * self.input_shape[self.FEAT_DIM])

        seq_len = self.input_shape[self.SEQ_DIM]
        for s in self.cnn_config[self.SEQ_DIM].strides:
            seq_len = (seq_len + s[self.SEQ_DIM] - 1) // s[self.SEQ_DIM]

        self.fully_connected_layer = nn.Linear(self.rnn_config.numofneurons // self.DIVIDE *
                                               (2 if self.rnn_config.is_bidirectional else 1) * seq_len,
                                               self.numoffeature)
        self.outputs = nn.Sigmoid()

    def __str__(self):
        return f'LivenessNetwork'

    @classmethod
    def alias(cls):
        return f'liveness'

    def forward(self, x:torch.Tensor):
        """
        :param x: audio features
        :return:
        """

        # batch x seq x feat -> batch x channel (1) x seq x feat
        tmp_x = x.unsqueeze(1)
        c_r = self.cnn_layers(tmp_x)

        # B x C x seq_len x F to B x seq_len x C x F
        c_r = c_r.transpose(1, 2)

        # B x seq_len x (C x F)
        sizes = c_r.size()
        c_r = c_r.reshape(sizes[0], sizes[1], sizes[2] * sizes[3])

        r_r, h_s = self.rnn_cell(c_r)

        # temperature scaling
        # B x M -> B x F x M
        y = self.fully_connected_layer(r_r.reshape(r_r.shape[0], -1))
        return self.outputs(y)

    def _build_conv_layer(self, cnn_config: CNN_CONFIG) -> List[nn.Conv2d]:
        feats = 1
        layers = []
        for idx, x in enumerate(cnn_config.numoflayers):
            layers.append(nn.Conv2d(in_channels=feats, out_channels=x, kernel_size=cnn_config.kernel_sizes[idx],
                                    stride=cnn_config.strides[idx], padding=[cnn_config.kernel_sizes[idx][0] // 2,
                                                                             cnn_config.kernel_sizes[idx][1] // 2]))
            layers.append(nn.BatchNorm2d(x))
            layers.append(nn.ReLU(inplace=True))
            feats = x
        return nn.Sequential(*layers)

    def _build_rnn_layer(self, feats: int):
        if self.rnn_config.rnn_type == 'gru':
            rnn_cell = torch.nn.GRU
        elif self.rnn_config.rnn_type == 'lstm':
            rnn_cell = torch.nn.LSTM

        # [batch, seq_len, input_size]
        #
        return rnn_cell(input_size=feats, hidden_size=self.rnn_config.numofneurons // self.DIVIDE, num_layers=self.rnn_config.numoflayers,
                        batch_first=True, bidirectional=self.rnn_config.is_bidirectional,
                        dropout=self.rnn_config.dropout)
