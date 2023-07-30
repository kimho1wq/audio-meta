from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

__all__ = [
    'AudioInfo',
    'PreProcConfig',
    'TransformInfo',
    'TransformConfig',
    'CNN_CONFIG',
    'RNN_CONFIG',
    'DENSE_CONFIG',
    'TASK_CONFIG',
    'TASK_TYPE',
    'ExtractorConfig',
    'AnalysisInfo',
    'AnalysisExtractorConfig',
    'MusicDetectionConfig',
    'PipelineVersion'
]


@dataclass
class AudioInfo:
    sampling_rate: int
    is_mono: bool

    def get_audio_info(self) -> Tuple[int, bool]:
        return tuple(self.sampling_rate, self.is_mono)

    def to_dict(self) -> dict:
        return {'sampling_rate': self.sampling_rate,
                'is_mono': self.is_mono}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            sampling_rate=f_dict['sampling_rate'],
            is_mono=f_dict['is_mono']
        )

@dataclass
class PreProcConfig:
    audio_info: AudioInfo
    aud_db_sz: int = None

    def to_dict(self) -> dict:
        return {'audio_info': self.audio_info.to_dict(),
                'aud_db_sz': self.aud_db_sz}

    @classmethod
    def from_dict(cls, f_dict: dict) -> PreProcConfig:
        return cls(
            audio_info=AudioInfo.from_dict(f_dict['audio_info']),
            aud_db_sz=f_dict['aud_db_sz']
        )

@dataclass
class TransformInfo:
    transform_type: str
    param: dict

    def to_dict(self) -> dict:
        return {'transform_type': self.transform_type,
                'param': self.param}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            transform_type=f_dict['transform_type'],
            param=f_dict['param']
        )

@dataclass
class TransformConfig:
    transform_info: List[TransformInfo]
    aud_db_sz: int = None

    def to_dict(self) -> dict:
        return {'transform_info': {idx:t_info.to_dict() for idx, t_info in enumerate(self.transform_info)},
                'aud_db_sz': self.aud_db_sz}

    @classmethod
    def from_dict(cls, f_dict: dict) -> TransformConfig:
        return cls(
            transform_info=[TransformInfo.from_dict(f_dict['transform_info'][key])
                            for key in f_dict['transform_info'].keys()],
            aud_db_sz=f_dict['aud_db_sz']
        )

@dataclass
class CNN_CONFIG:
    type: str
    numoflayers: int
    kernel_sizes: List[list]
    strides: List[list]

    def to_dict(self) -> dict:
        return {'type': self.type,
                'numoflayers': self.numoflayers,
                'kernel_sizes': self.kernel_sizes,
                'strides': self.strides}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            type=f_dict['type'],
            numoflayers=f_dict['numoflayers'],
            kernel_sizes=f_dict['kernel_sizes'],
            strides=f_dict['strides']
        )


@dataclass
class RNN_CONFIG:
    numoflayers: int
    numofneurons: int
    rnn_type: str # choices = ['gru', 'lstm']
    is_bidirectional: bool
    dropout: float

    def to_dict(self) -> dict:
        return {'numoflayers': self.numoflayers,
                'numofneurons': self.numofneurons,
                'rnn_type': self.rnn_type,
                'is_bidirectional': self.is_bidirectional,
                'dropout': self.dropout}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            numoflayers=f_dict['numoflayers'],
            numofneurons=f_dict['numofneurons'],
            rnn_type=f_dict['rnn_type'],
            is_bidirectional=f_dict['is_bidirectional'],
            dropout=f_dict['dropout']
        )

@dataclass
class DENSE_CONFIG:
    numoflayers: int
    numofneurons: list
    dropout: float

    def to_dict(self) -> dict:
        return {'numoflayers': self.numoflayers,
                'numofneurons' : self.numofneurons,
                'dropout': self.dropout}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            numoflayers=f_dict['numoflayers'],
            numofneurons=f_dict['numofneurons'],
            dropout=f_dict['dropout']
        )

@dataclass
class TASK_TYPE:
    multi_task: list
    single_task: list
    type: str

    def to_dict(self) -> dict:
        return {'type': self.type,
                'multi_task': self.multi_task,
                'single_task': self.single_task}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            type=f_dict['type'],
            multi_task=f_dict['multi_task'],
            single_task=f_dict['single_task']
        )

@dataclass
class TASK_CONFIG:
    task_type: TASK_TYPE
    q_index: int
    temperature: float

    def to_dict(self) -> dict:
        return {'task_type': self.task_type.to_dict(),
                'q_index': self.q_index,
                'temperature': self.temperature}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            task_type=TASK_TYPE.from_dict(f_dict['task_type']),
            q_index=f_dict['q_index'],
            temperature=f_dict['temperature']
        )

@dataclass
class ExtractorConfig:
    cnn_config: List[CNN_CONFIG]
    rnn_config: RNN_CONFIG
    dense_config: DENSE_CONFIG
    max_length_limit_ms: int
    task_config: TASK_CONFIG
    result_type: str

    def to_dict(self) -> dict:
        return {'cnn_config': [c_info.to_dict() for c_info in self.cnn_config],
                'rnn_config': self.rnn_config.to_dict(),
                'dense_config': self.dense_config.to_dict(),
                'max_length_limit_ms': self.max_length_limit_ms,
                'task_config': self.task_config.to_dict(),
                'result_type': self.result_type}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            cnn_config=[CNN_CONFIG.from_dict(f_dict['cnn_config'][key])
                        for key in f_dict['cnn_config'].keys()],
            rnn_config=RNN_CONFIG.from_dict(f_dict['rnn_config']),
            dense_config=DENSE_CONFIG.from_dict(f_dict['dense_config']),
            max_length_limit_ms=f_dict['max_length_limit_ms'],
            task_config=TASK_CONFIG.from_dict(f_dict['task_config']),
            result_type=f_dict['result_type']
        )

@dataclass
class AnalysisInfo:
    audio_feature: str
    target_transform: str
    param: dict

    def to_dict(self) -> dict:
        return {'audio_feature': self.audio_feature,
                'target_transform': self.target_transform,
                'param': self.param}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            audio_feature=f_dict['audio_feature'],
            target_transform=f_dict['target_transform'],
            param=f_dict['param']
        )

@dataclass
class AnalysisExtractorConfig:
    analysis_info: List[AnalysisInfo]
    max_length_limit_ms: int = None

    def to_dict(self) -> dict:
        return {'analysis_info': {idx:t_info.to_dict() for idx, t_info in enumerate(self.analysis_info)},
                'max_length_limit_ms': self.max_length_limit_ms}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            analysis_info=[AnalysisInfo.from_dict(f_dict['analysis_info'][key])
                            for key in f_dict['analysis_info'].keys()],
            max_length_limit_ms=f_dict['max_length_limit_ms']
        )


@dataclass
class MusicDetectionConfig:
    cnn_config: List[CNN_CONFIG]
    num_epochs: int
    num_gpus: int
    num_workers: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hop_length: int
    check_val_every_n_epoch: int

    def to_dict(self) -> dict:
        return {'cnn_config': [c_info.to_dict() for c_info in self.cnn_config],
                'num_epochs': self.num_epochs,
                'num_gpus': self.num_gpus,
                'num_workers': self.num_workers,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'check_val_every_n_epoch': self.check_val_every_n_epoch,
                'hop_length': self.hop_length}

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            cnn_config=[CNN_CONFIG.from_dict(f_dict['cnn_config'][key])
                        for key in f_dict['cnn_config'].keys()],
            num_epochs=f_dict['num_epochs'],
            num_gpus=f_dict['num_gpus'],
            num_workers=f_dict['num_workers'],
            batch_size=f_dict['batch_size'],
            learning_rate=f_dict['learning_rate'],
            weight_decay=f_dict['weight_decay'],
            check_val_every_n_epoch=f_dict['check_val_every_n_epoch'],
            hop_length=f_dict['hop_length']
        )


class ProcessAbbr(Enum):
    """
    Abbreviation for pipeline processes with TWO alphabets each
    """
    database = 'db'
    pre_processing = 'pp'
    transform = 'tr'
    extractor = 'ex'


ABBR_LEN = 2


@dataclass
class PipelineVersion:
    database_ver: int = None
    pre_processing_ver: int = None
    transform_ver: int = None
    extractor_ver: int = None

    @classmethod
    def from_str(cls, str_ver: str):
        subvs = str_ver.split('.')

        assert 1 <= len(subvs) <= 4

        valid_abbrs = set([abbr.value for abbr in ProcessAbbr])

        kwargs = {}

        for subv in subvs:
            abbr, ver_dix = subv[:ABBR_LEN], subv[ABBR_LEN:]

            assert ver_dix.isdigit()
            assert abbr in valid_abbrs

            ver_idx = int(ver_dix)
            abbr = ProcessAbbr(abbr)

            kwargs[f'{abbr.name}_ver'] = ver_idx

        return cls(**kwargs)

    def get_audio_db_ver(self) -> str:
        """
        Get abbreviated audio database version
        """
        return f'{ProcessAbbr.database.value}{self.database_ver}'

    def get_audio_db_fn(self) -> str:
        """
        Get filename for directory where db audios are saved
        """
        return f'{self.get_audio_db_ver()}'

    def get_preproc_ver(self) -> str:
        """
        Get abbreviated preprocessing version
        """
        return f'{ProcessAbbr.pre_processing.value}{self.pre_processing_ver}'

    def get_preproc_fn(self) -> str:
        """
        Get filename for directory where db audios are saved
        """
        return f'{self.get_audio_db_fn()}.{self.get_preproc_ver()}'

    def get_transform_ver(self) -> str:
        """
        Get abbreviated transform version
        """
        return f'{ProcessAbbr.transform.value}{self.transform_ver}'

    def get_transform_fn(self) -> str:
        """
        Get filename for directory where spectrum are saved
        """
        return f'{self.get_preproc_fn()}.{self.get_transform_ver()}'

    def get_extractor_ver(self) -> str:
        """
        Get abbreviated extractor version
        """
        return f'{ProcessAbbr.extractor.value}{self.extractor_ver}'

    def get_extractor_fn(self) -> str:
        """
        Get filename for directory where extractor are saved
        """
        return f'{self.get_transform_fn()}.{self.get_extractor_ver()}'

