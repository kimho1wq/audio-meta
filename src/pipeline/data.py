from __future__ import annotations
import json

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, List

from src.utils.data import DictConvertable
from src.database.data import AudioFile
from src.audio import AudioSeqFileInfo, SpecFileInfo
from assets.config.config import TransformInfo, PreProcConfig, TransformConfig

__all__ = [
    'AudioDBConfig',
    'PipelineGenerator',
    'PipelineSpecGenerator',
    'AudioDBMetadata',
    'AudioSeqMetadata',
    'SpecMetadata'
]


class PipelineGenerator(ABC):
    """
    ABC for generator in data generating pipelines
    """

    DATA_KEY = 'dataset'
    GENERATOR = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.GENERATOR[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.GENERATOR[alias]

    @abstractmethod
    def generate(self):
        pass

class PipelineSpecGenerator(ABC):
    """
    ABC for generator in data generating pipelines
    """

    DATA_KEY = 'dataset'
    SPEC_GENERATOR = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.SPEC_GENERATOR[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.SPEC_GENERATOR[alias]

    @abstractmethod
    def generate(self):
        pass


@dataclass
class LabelConfig(DictConvertable):
    name: str
    params: dict

    def to_dict(self) -> dict:
        return {'name': self.name,
                'params': self.params}

    @classmethod
    def from_dict(cls, f_dict: dict) -> AudioDBConfig:
        return cls(
            name=f_dict['name'],
            params=f_dict['params']
        )

@dataclass
class AudioDBConfig(DictConvertable):
    label: LabelConfig
    aud_db_sz: None

    def to_dict(self) -> dict:
        return {'label': self.label.to_dict(),
                'aud_db_sz': self.aud_db_sz}

    @classmethod
    def from_dict(cls, f_dict: dict) -> AudioDBConfig:
        return cls(
            label=LabelConfig.from_dict(f_dict['label']),
            aud_db_sz=f_dict['aud_db_sz']
        )


MetaType = TypeVar('MetaType', bound='Metadata')


class Metadata(DictConvertable):
    DATA_KEY = 'dataset'
    DEFAULT_FILENAME = 'metadata.json'

    @classmethod
    def load_from_file(cls: Type[MetaType], file_dir: Path,
                       filename: str = DEFAULT_FILENAME) -> MetaType:
        with open(file_dir.joinpath(filename), 'r') as f:
            metadata = json.load(f)
        return cls.from_dict(metadata)

    def save_to_file(self, save_dir: Path, filename: str = DEFAULT_FILENAME):
        with open(save_dir.joinpath(filename), 'w') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_dict(cls: Type[MetaType], f_dict: dict) -> MetaType:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()


@dataclass
class AudioDBMetadata(Metadata):
    """
    Metadata for a dataset of Audio database.
    """
    audio_db_info: AudioDBConfig
    audio_file_infos: List[AudioFile]
    db_sz: int

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            audio_db_info=AudioDBConfig.from_dict(f_dict['audio_db_info']),
            audio_file_infos=[AudioFile.AUDIOFILE[f_dict['f_alias']](audio_file_infos) for audio_file_infos in f_dict[cls.DATA_KEY]],
            db_sz=f_dict['db_sz']
        )

    def to_dict(self) -> dict:
        dict_data = {'audio_db_info': self.audio_db_info.to_dict(),
                     'f_alias': self.audio_file_infos[0].alias()}

        dict_data.update({
            'db_sz': self.db_sz,
            self.DATA_KEY: [seq_info.to_dict() for seq_info in self.audio_file_infos]
        })

        return dict_data


@dataclass
class AudioSeqMetadata(Metadata):
    """
    Metadata for a dataset of AudioSequence.
    """
    pre_proc_info: PreProcConfig
    seq_infos: List[AudioSeqFileInfo]
    db_sz: int

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            pre_proc_info=PreProcConfig.from_dict(f_dict['pre_proc_info']),
            seq_infos=[AudioSeqFileInfo.from_dict(seq_dict) for seq_dict in f_dict[cls.DATA_KEY]],
            db_sz=f_dict['db_sz']
        )

    def to_dict(self) -> dict:
        dict_data = {'pre_proc_info': self.pre_proc_info.to_dict()}

        dict_data.update({
            'db_sz': self.db_sz,
            self.DATA_KEY: [seq_info.to_dict() for seq_info in self.seq_infos]
        })

        return dict_data


@dataclass
class SpecMetadata(Metadata):
    """
    Metadata for a dataset of Spectrogram.
    """
    pre_proc_info: PreProcConfig
    transform_info: TransformInfo
    spec_infos: List[SpecFileInfo]
    db_sz: int

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            pre_proc_info=PreProcConfig.from_dict(f_dict['pre_proc_info']),
            transform_info=TransformConfig.from_dict(f_dict['transform_info']),
            spec_infos=[SpecFileInfo.from_dict(seq_dict) for seq_dict in f_dict[cls.DATA_KEY]],
            db_sz=f_dict['db_sz']
        )

    def to_dict(self) -> dict:
        dict_data = {'pre_proc_info': self.pre_proc_info.to_dict(),
                     'transform_info': self.transform_info.to_dict()}

        dict_data.update({
            'db_sz': self.db_sz,
            self.DATA_KEY: [spec_infos.to_dict() for spec_infos in self.spec_infos]
        })

        return dict_data
