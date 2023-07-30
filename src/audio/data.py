from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from src.utils import DictConvertable
from src.database.data import AudioFile
from assets.config.config import AudioInfo, TransformInfo

__all__ = [
    'AudioSeqInfo',
    'AudioSequence',
    'AudioSeqFileInfo',
    'SpecFileInfo',
    'Spectrogram'
]



@dataclass
class AudioSeqInfo(DictConvertable):
    audio_info: AudioInfo
    meta: AudioFile

    def get_audio_info(self) -> Tuple[int, bool]:
        return self.audio_info.get_audio_info()

    def to_dict(self) -> dict:
        return {'audio_info': self.audio_info.to_dict(),
                'f_alias': self.meta.alias(),
                'meta': self.meta.to_dict()}

    @classmethod
    def from_dict(cls, f_dict: dict) -> AudioSeqFileInfo:
        return cls(
            meta=AudioFile.AUDIOFILE[f_dict['f_alias']](f_dict['meta']),
            audio_info=AudioInfo.from_dict(f_dict['audio_info'])
        )


@dataclass
class AudioSeqFileInfo(DictConvertable):
    audio_seq_info: AudioSeqInfo
    path: Path

    def to_dict(self) -> dict:
        return {'audio_seq_info': self.audio_seq_info.to_dict(),
                'path': str(self.path)}

    @classmethod
    def from_dict(cls, f_dict: dict) -> AudioSeqFileInfo:
        return cls(
            audio_seq_info=AudioSeqInfo.from_dict(f_dict['audio_seq_info']),
            path=f_dict['path']
        )


class AudioSequence:
    """
    Class for audio sequence.
    """

    NUM_DIM = 1
    SAMPLE_DIM = 0

    def __init__(self, arr: np.ndarray, aud_seq_info: AudioSeqInfo):
        if len(arr.shape) != self.NUM_DIM:
            raise ValueError(
                f'Invalid dimension of input: Expected {self.NUM_DIM}, '
                f'but got {len(arr.shape)}'
            )

        self.arr = arr
        self.aud_seq_info = aud_seq_info

    def __len__(self):
        return self.get_num_samples()

    def get_num_samples(self):
        return self.arr.shape[self.SAMPLE_DIM]

    def get_num_frames(self, window_size: int, hop_size: int) -> int:
        return (self.get_num_samples() - window_size) // hop_size + 1

    @staticmethod
    def load_with_file_info(seq_info: AudioSeqFileInfo, mmap_mode: str = None) -> AudioSequence:
        arr = np.load(seq_info.path, mmap_mode=mmap_mode)
        return AudioSequence(arr=arr, aud_seq_info=seq_info.audio_seq_info)

@dataclass
class SpecFileInfo(DictConvertable):
    # Spectral information
    file_info: AudioSeqInfo
    transform_info: TransformInfo
    path: Path

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            file_info = AudioSeqInfo.from_dict(f_dict['file_info']),
            transform_info=TransformInfo.from_dict(f_dict['transform_info']),
            path=f_dict['path']
        )

    def to_dict(self) -> dict:
        return {
            'file_info': self.file_info.to_dict(),
            "transform_info": self.transform_info.to_dict(),
            "path": str(self.path)
        }


@dataclass
class Spectrogram:
    # Spectral information
    arr: np.ndarray
    transform_info: TransformInfo

    SEQ_DIM = 0
    FREQ_DIM = 1

    def __getitem__(self, item):
        self.arr[item]

    def get_num_frames(self):
        return self.arr.shape[self.SEQ_DIM]

    def get_num_freq_bins(self):
        return self.arr.shape[self.FREQ_DIM]

    @classmethod
    def load_with_file_info(cls, file_info: SpecFileInfo, mmap_mode:str = None) -> Spectrogram:
        loaded_arr = np.load(file_info.path, mmap_mode=mmap_mode)
        return cls(arr=loaded_arr, transform_info=file_info.transform_info)


