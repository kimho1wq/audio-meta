from typing import List, Dict

import torch
import torch.nn as nn
from assets.config.config import ExtractorConfig, CNN_CONFIG

from src.extraction.data import Network

__all__ = [
    'MultiAudioMetaNetwork'
]

"""
# Extracted audio features : Energy, valence, acousticness 
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

class MultiAudioMetaNetwork(Network):
    #
    NUM_OF_FILTERS_IN_MERGE_LAYER = 256
    SEQ_DIM = 0
    FEAT_DIM = 1

    def __init__(self, extractor_config: ExtractorConfig, input_shape: tuple, seed: int = 0):
        super().__init__()

        self.cnn_config = extractor_config.cnn_config
        self.rnn_config = extractor_config.rnn_config
        self.dense_config = extractor_config.dense_config
        self.task_type = extractor_config.task_config.task_type.type
        self.temperature = extractor_config.task_config.temperature

        self.audio_features = extractor_config.task_config.task_type.multi_task
        self.numoffeature = 1 if self.task_type[0] == 'r' else extractor_config.task_config.q_index
        self.input_shape = input_shape

        # torch seed initialization
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # build a spatial part of CNNs
        # B x C x H x W : batch x 1 x seq_len x spectra-dim
        feats = 0
        self.cnn_layers = {}
        for _cnn_config in self.cnn_config:
            #
            feats += _cnn_config.numoflayers[-1]
            self.cnn_layers.update({_cnn_config.type: ImplementsInterface(self._build_conv_layer(cnn_config=_cnn_config))})

        self.cnn_layers = nn.ModuleDict(self.cnn_layers)
        self.merge_layer = self._build_conv_merge_layer(feats)
        self.rnn_cell = self._build_rnn_layer(self.NUM_OF_FILTERS_IN_MERGE_LAYER * self.input_shape[self.FEAT_DIM])

        seq_len = self.input_shape[self.SEQ_DIM]
        for s in self.cnn_config[self.SEQ_DIM].strides:
            seq_len = (seq_len + s[self.SEQ_DIM] - 1) // s[self.SEQ_DIM]

        self.fully_connected_layer = nn.ModuleDict({feat: ImplementsInterface(self._build_dense_layer((self.rnn_config.numofneurons *
                                                    (2 if self.rnn_config.is_bidirectional else 1) * seq_len,
                                                    self.numoffeature))) for feat in self.audio_features})

        self.outputs = nn.Sigmoid()
        self.seq_len = self.rnn_config.numofneurons * (2 if self.rnn_config.is_bidirectional else 1) * seq_len

    def __str__(self):
        return f'MultiAudioMetaNetwork'

    @classmethod
    def alias(cls):
        return f'multi_audio_meta'

    def forward(self, x:Dict[str, torch.Tensor]):
        """
        :param x: audio features
        :return:
        """

        # Feature extraction with pre-trained model
        c_r = []
        for key in x.keys():
            # batch x seq x feat -> batch x channel (1) x seq x feat
            tmp_x = x[key].unsqueeze(1)
            _layers: ModuleInterface = self.cnn_layers[key]
            tmp_x = _layers.forward(tmp_x)
            c_r.append(tmp_x)

        # concat
        c_r = torch.cat(c_r, dim=1)
        c_r = self.merge_layer(c_r)

        # B x C x seq_len x F to B x seq_len x C x F
        c_r = c_r.transpose(1, 2)

        # B x seq_len x (C x F)
        sizes = c_r.size()
        c_r = c_r.reshape(sizes[0], sizes[1], sizes[2] * sizes[3])

        r_r, h_s = self.rnn_cell(c_r)

        # temperature scaling
        res = []
        for key in self.audio_features:
            _layers: ModuleInterface = self.fully_connected_layer[key]

            # B x M -> B x F x M
            y = _layers.forward(r_r.reshape(r_r.shape[0], -1))
            res.append(y / self.temperature if self.task_type[0] == 'c' else self.outputs(y))
        return torch.stack(res, dim=1)

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

    def _build_conv_merge_layer(self, feats: int) -> nn.Conv2d:
        return nn.Sequential(
                    nn.Conv2d(in_channels=feats, out_channels=self.NUM_OF_FILTERS_IN_MERGE_LAYER, kernel_size=[1, 1],
                              stride=[1, 1]),
                    nn.BatchNorm2d(self.NUM_OF_FILTERS_IN_MERGE_LAYER),
                    nn.ReLU(inplace=True)
        )

    def _build_rnn_layer(self, feats: int):
        if self.rnn_config.rnn_type == 'gru':
            rnn_cell = torch.nn.GRU
        elif self.rnn_config.rnn_type == 'lstm':
            rnn_cell = torch.nn.LSTM

        # [batch, seq_len, input_size]
        #
        return rnn_cell(input_size=feats, hidden_size=self.rnn_config.numofneurons, num_layers=self.rnn_config.numoflayers,
                        batch_first=True, bidirectional=self.rnn_config.is_bidirectional,
                        dropout=self.rnn_config.dropout)

    def _build_dense_layer(self, input_shape:tuple):
        _in = input_shape[0]
        _out = input_shape[1]

        layers = []
        for i in range(self.dense_config.numoflayers):
            layers.append(nn.Linear(_in, self.dense_config.numofneurons[i]))
            layers.append(nn.Dropout(p=self.dense_config.dropout, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            _in = self.dense_config.numofneurons[i]
        layers.append(nn.Linear(_in, _out))

        return nn.Sequential(*layers)

