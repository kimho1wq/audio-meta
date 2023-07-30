from __future__ import annotations

import json
import numpy as np
import bisect

from torch.utils.data import Dataset
import torch

from src.audio import Spectrogram
from assets.config.config import TASK_CONFIG


__all__ = [
    'InstrumentalnessCustomDataset',
    'LivenessCustomDataset',
    'AudioMetaCustomDataset',
    'MusicDetectionDataset'
]

class InstrumentalnessCustomDataset(Dataset):
    SEQ_DIM = 0
    FEAT_DIM = 1

    X = 0
    Y = 1

    def __init__(self, data, input_shape: list):
        super().__init__()
        self.data = data
        self.input_shape = input_shape
        self.input_feature = 'spectrogram'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.create_data_arr(self.data[idx])

    def create_data_arr(self, input_spec: dict):
        # TODO
        # from heuristic value to min-max normalization
        PARAMETERS = {
            'spectrogram': 40.
                      }
        x, y = input_spec[self.X], input_spec[self.Y]

        spec = np.load(x)
        empty_arr = np.zeros((self.input_shape[self.SEQ_DIM], spec.shape[self.FEAT_DIM]))
        boundary = spec.shape[self.SEQ_DIM]
        empty_arr[:boundary] = spec / PARAMETERS[self.input_feature]

        with open(y, 'r') as f:
            l = np.asarray([int(f.read())])

        return torch.from_numpy(empty_arr.astype('float32')), torch.from_numpy(l.astype('float32'))

class LivenessCustomDataset(Dataset):
    SEQ_DIM = 0
    FEAT_DIM = 1

    X = 0
    Y = 1

    def __init__(self, data, input_shape: list):
        super().__init__()
        self.data = data
        self.input_shape = input_shape
        self.input_feature = 'spectrogram'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.create_data_arr(self.data[idx])


    def create_data_arr(self, input_spec: dict):
        # TODO
        # from heuristic value to min-max normalization
        PARAMETERS = {
            'spectrogram': 40.
                      }
        x, y = input_spec[self.X], input_spec[self.Y]

        spec = np.load(x)
        empty_arr = np.zeros((self.input_shape[self.SEQ_DIM], spec.shape[self.FEAT_DIM]))
        boundary = spec.shape[self.SEQ_DIM]
        empty_arr[:boundary] = spec / PARAMETERS[self.input_feature]

        with open(y, 'r') as f:
            l = np.asarray([int(f.read())])

        return torch.from_numpy(empty_arr.astype('float32')), torch.from_numpy(l.astype('float32'))

class AudioMetaCustomDataset(Dataset):
    X = 0
    Y = 1

    SEQ_DIM = 0
    FEAT_DIM = 1

    def __init__(self, data, audio_features: list, input_shape: list, task_config: TASK_CONFIG):
        super().__init__()
        self.data = data
        self.audio_features = audio_features
        self.input_shape = input_shape
        self.task_config = task_config

        if self.task_config.task_type.type[0] == 'c':
            v = 1 / self.task_config.q_index
            self.q_depth = [v * i for i in range(self.task_config.q_index)]
        else:
            self.q_depth = None

    def __len__(self):
        return len(self.data[self.Y])

    def __getitem__(self, idx):
        x = self.create_input_arr(self.data[self.X][idx])
        y = self.create_labels(self.data[self.Y][idx])

        return x, y

    def create_labels(self, label_path: str):
        with open(label_path, 'r') as f:
            meta = json.load(f)
            _type = 'float32'
            res = []

            if self.q_depth:
                for i, feat in enumerate(self.audio_features):
                    outputs = np.zeros(len(self.q_depth))
                    outputs[int(bisect.bisect_right(self.q_depth, meta['feats'][feat])) - 1] = 1.
                    res.append(outputs)
            else:
                for i, feat in enumerate(self.audio_features):
                    outputs = np.zeros(1)
                    outputs[0] = meta['feats'][feat]
                    res.append(outputs)

        return torch.from_numpy(np.asarray(res).astype(_type))

    def create_input_arr(self, input_spec: dict):
        x = {}

        # TODO
        # from heuristic value to min-max normalization
        PARAMETERS = {
            'chromagram': 1.,
            'cqt': 40.,
            'tempogram': 1.,
            'spectrogram': 40.
                      }

        for tf in input_spec.keys():
            spec = Spectrogram.load_with_file_info(input_spec[tf]).arr
            empty_arr = np.zeros((self.input_shape[self.SEQ_DIM], spec.shape[self.FEAT_DIM]))
            boundary = spec.shape[self.SEQ_DIM]
            empty_arr[:boundary] = spec / PARAMETERS[tf]

            x[tf] = torch.from_numpy(empty_arr.astype('float32'))

        return x

class MusicDetectionDataset(Dataset):
    def __init__(self, data: list, labels: list, hop_length: int):
        super().__init__()
        self.data = data
        self.labels = labels
        self.hop_length = hop_length

        self.SEQ_DIM = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = {}
        for key in self.data.keys():
            spec = self.data[key][idx]
            if spec.shape[self.SEQ_DIM] > self.hop_length:
                s_idx = np.random.randint(low=0, high=spec.shape[self.SEQ_DIM]-self.hop_length)
                spec = spec[s_idx : s_idx+self.hop_length]
            elif spec.shape[self.SEQ_DIM] < self.hop_length:
                n_duplicate = int(self.hop_length / spec.shape[self.SEQ_DIM]) + 1
                spec = np.tile(spec, (n_duplicate, 1))[:self.hop_length]
            else:
                spec = spec
            data[key] = torch.from_numpy(spec.T).unsqueeze(0)
            
        is_speech = [1] if self.labels[idx] == 'speech' else [0]
        is_music = [1] if self.labels[idx] == 'music' else [0]
        
        data['speech_label'] = torch.tensor(is_speech, dtype=torch.float32)
        data['music_label'] = torch.tensor(is_music, dtype=torch.float32)

        return data