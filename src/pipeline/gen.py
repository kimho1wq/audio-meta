import numpy as np

from pathlib import Path
from typing import List
from multiprocessing import Pool

from src.pipeline.data import (PipelineGenerator, PipelineSpecGenerator, AudioDBMetadata,
                               AudioSeqMetadata, SpecMetadata)
from src.audio.data import *
from src.utils.data import custom_mkdir
from src.database.data import AudioFile
from src.preprocessing import Loader
from src.transform import AudioTransformer
from assets.config.config import PreProcConfig, TransformConfig

__all__ = [
    'AudioMetaDBAudioGenerator',
    'AudioMetaDBSpectrogramGenerator',
    'AudioAnalysisDBAudioGenerator',
    'AudioAnalysisDBSpectrogramGenerator',
    'MusicDetectionDBAudioGenerator',
    'MusicDetectionDBSpectrogramGenerator'
]


class AudioMetaDBAudioGenerator(PipelineGenerator):
    """
    Converter for a given database of audio files into raw audio samples
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)

        self.audio_load = Loader(self.pre_proc_config.audio_info)

        custom_mkdir(self.out_dir)

    @classmethod
    def alias(cls):
        return 'audio_meta'

    def generate(self):
        meta_list = self._load_vid_meta()

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._load_and_save, meta_list))

        audio_seq_meta = AudioSeqMetadata(
            pre_proc_info=self.pre_proc_config,
            seq_infos=save_meta_list,
            db_sz=self.pre_proc_config.aud_db_sz
        )

        audio_seq_meta.save_to_file(self.out_dir)

    def _load_vid_meta(self) -> List[AudioFile]:
        aud_meta_list = AudioDBMetadata.load_from_file(self.in_dir).audio_file_infos

        # sorted by uid
        aud_meta_list = sorted(aud_meta_list, key=lambda x: x.audio_file_meta.trackinfo.trackId)

        if self.pre_proc_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(aud_meta_list) < self.pre_proc_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(aud_meta_list)},'
                                 f'but asked {self.pre_proc_config.aud_db_sz}')

            aud_meta_list = aud_meta_list[:self.pre_proc_config.aud_db_sz]
        else:
            self.pre_proc_config.aud_db_sz = len(aud_meta_list)

        return aud_meta_list

    def _load_and_save(self, meta: AudioFile,
                       save_file_format: str = '{}.npy') -> AudioSeqFileInfo:
        arr_out_path = self.out_dir.joinpath(save_file_format.format(meta.audio_file_meta.trackinfo.trackId))

        # Load audio file..
        y = self.audio_load(Path(meta.path))

        with open(arr_out_path, 'wb') as f:
            np.save(f, y)

        audio_file_info = AudioSeqFileInfo(
            audio_seq_info=AudioSeqInfo(
                audio_info=self.pre_proc_config.audio_info,
                meta=meta
            ),
            path=arr_out_path
        )

        print(f'*---------------------------------------------------\n'
              f"DBAudioGenerator - Audio sequence Saving Success\n"
              f"Video sequence file info. :"
              f"\n - {audio_file_info.audio_seq_info} "
              f"\n - {audio_file_info.path}"
              f'\n---------------------------------------------------*')

        return audio_file_info


class AudioMetaDBSpectrogramGenerator(PipelineSpecGenerator):
    """
    Converter for a given audio files into spectral domain
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)

        self.transform = [AudioTransformer.TRANSFORMER[t_config.transform_type](**t_config.param)
                          for t_config in self.transform_config.transform_info]

        self.audio_seq_meta = AudioSeqMetadata.load_from_file(self.in_dir)

        if self.transform_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(self.audio_seq_meta.seq_infos) < self.transform_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(self.audio_seq_meta.seq_infos)},'
                                 f'but asked {self.transform_config.aud_db_sz}')

            self.audio_seq_meta.seq_infos = self.audio_seq_meta.seq_infos[:self.transform_config.aud_db_sz]
        else:
            self.transform_config.aud_db_sz = len(self.audio_seq_meta.seq_infos)

        custom_mkdir(self.out_dir)

        for transform in self.transform:
            custom_mkdir(self.out_dir.joinpath(transform.alias()))

    @classmethod
    def alias(cls):
        return 'audio_meta'

    def generate(self):
        audio_seq_meta = self.audio_seq_meta.seq_infos

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._transform_and_save, audio_seq_meta))
        save_meta_list = [m for m in save_meta_list if m]

        #
        for key in save_meta_list[0].keys():
            meta_list = [meta[key] for meta in save_meta_list]

            spec_metadata = SpecMetadata(
                pre_proc_info=self.pre_proc_config,
                transform_info=self.transform_config,
                spec_infos=meta_list,
                db_sz=self.transform_config.aud_db_sz
            )

            spec_metadata.save_to_file(self.out_dir.joinpath(key))

    def _transform_and_save(self, seq_info: AudioSeqFileInfo,
                            save_file_format: str = '{}.npy') -> dict:
        # Load audio file..
        audio_seq = AudioSequence.load_with_file_info(seq_info)

        if len(audio_seq.arr) < 60 * self.pre_proc_config.audio_info.sampling_rate:
            return False

        results = {}
        for idx, transformer in enumerate(self.transform):
            arr_out_path = (self.out_dir.joinpath(transformer.alias())).joinpath(
                save_file_format.format(seq_info.audio_seq_info.meta.audio_file_meta.trackinfo.trackId))

            spec = transformer(
                y=audio_seq.arr,
                sr=audio_seq.aud_seq_info.audio_info.sampling_rate
            ).astype('float32')

            with open(arr_out_path, 'wb') as f:
                np.save(f, spec)

            audio_file_info = SpecFileInfo(
                file_info=seq_info.audio_seq_info,
                transform_info=self.transform_config.transform_info[idx],
                path=arr_out_path
            )

            results[transformer.alias()] = audio_file_info

        print(f'*---------------------------------------------------\n'
              f"DBSpectrogramGenerator - Spectrogram gen success\n"
              f"Spectrogram file info. :"
              f"\n - {[results[key] for key in results.keys()]}"
              f'\n---------------------------------------------------*')

        return results


class AudioAnalysisDBAudioGenerator(PipelineGenerator):
    """
    Converter for a given database of audio files into raw audio samples
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)

        self.audio_load = Loader(self.pre_proc_config.audio_info)

        custom_mkdir(self.out_dir)

    @classmethod
    def alias(cls):
        return 'audio_analysis'

    def generate(self):
        meta_list = self._load_vid_meta()

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._load_and_save, meta_list))

        audio_seq_meta = AudioSeqMetadata(
            pre_proc_info=self.pre_proc_config,
            seq_infos=save_meta_list,
            db_sz=self.pre_proc_config.aud_db_sz
        )

        audio_seq_meta.save_to_file(self.out_dir)

    def _load_vid_meta(self) -> List[AudioFile]:
        aud_meta_list = AudioDBMetadata.load_from_file(self.in_dir).audio_file_infos

        # sorted by uid
        aud_meta_list = sorted(aud_meta_list, key=lambda x: x.audio_file_meta.trackinfo.trackId)

        if self.pre_proc_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(aud_meta_list) < self.pre_proc_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(aud_meta_list)},'
                                 f'but asked {self.pre_proc_config.aud_db_sz}')

            aud_meta_list = aud_meta_list[:self.pre_proc_config.aud_db_sz]
        else:
            self.pre_proc_config.aud_db_sz = len(aud_meta_list)

        return aud_meta_list

    def _load_and_save(self, meta: AudioFile,
                       save_file_format: str = '{}.npy') -> AudioSeqFileInfo:
        arr_out_path = self.out_dir.joinpath(save_file_format.format(meta.audio_file_meta.trackinfo.trackId))

        # Load audio file..
        y = self.audio_load(Path(meta.path))

        with open(arr_out_path, 'wb') as f:
            np.save(f, y)

        audio_file_info = AudioSeqFileInfo(
            audio_seq_info=AudioSeqInfo(
                audio_info=self.pre_proc_config.audio_info,
                meta=meta
            ),
            path=arr_out_path
        )

        print(f'*---------------------------------------------------\n'
              f"DBAudioGenerator - Audio sequence Saving Success\n"
              f"Video sequence file info. :"
              f"\n - {audio_file_info.audio_seq_info} "
              f"\n - {audio_file_info.path}"
              f'\n---------------------------------------------------*')

        return audio_file_info


class AudioAnalysisDBSpectrogramGenerator(PipelineSpecGenerator):
    """
    Converter for a given audio files into spectral domain
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)

        self.transform = [AudioTransformer.TRANSFORMER[t_config.transform_type](**t_config.param)
                          for t_config in self.transform_config.transform_info]

        self.audio_seq_meta = AudioSeqMetadata.load_from_file(self.in_dir)

        if self.transform_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(self.audio_seq_meta.seq_infos) < self.transform_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(self.audio_seq_meta.seq_infos)},'
                                 f'but asked {self.transform_config.aud_db_sz}')

            self.audio_seq_meta.seq_infos = self.audio_seq_meta.seq_infos[:self.transform_config.aud_db_sz]
        else:
            self.transform_config.aud_db_sz = len(self.audio_seq_meta.seq_infos)

        custom_mkdir(self.out_dir)

        for transform in self.transform:
            custom_mkdir(self.out_dir.joinpath(transform.alias()))

    @classmethod
    def alias(cls):
        return 'audio_analysis'

    def generate(self):
        audio_seq_meta = self.audio_seq_meta.seq_infos

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._transform_and_save, audio_seq_meta))
        save_meta_list = [m for m in save_meta_list if m]

        #
        for key in save_meta_list[0].keys():
            meta_list = [meta[key] for meta in save_meta_list]

            spec_metadata = SpecMetadata(
                pre_proc_info=self.pre_proc_config,
                transform_info=self.transform_config,
                spec_infos=meta_list,
                db_sz=self.transform_config.aud_db_sz
            )

            spec_metadata.save_to_file(self.out_dir.joinpath(key))

    def _transform_and_save(self, seq_info: AudioSeqFileInfo,
                            save_file_format: str = '{}.npy') -> dict:
        # Load audio file..
        audio_seq = AudioSequence.load_with_file_info(seq_info)

        if len(audio_seq.arr) < 60 * self.pre_proc_config.audio_info.sampling_rate:
            return False

        results = {}
        for idx, transformer in enumerate(self.transform):
            arr_out_path = (self.out_dir.joinpath(transformer.alias())).joinpath(
                save_file_format.format(seq_info.audio_seq_info.meta.audio_file_meta.trackinfo.trackId))

            spec = transformer(
                y=audio_seq.arr,
                sr=audio_seq.aud_seq_info.audio_info.sampling_rate
            ).astype('float32')

            with open(arr_out_path, 'wb') as f:
                np.save(f, spec)

            audio_file_info = SpecFileInfo(
                file_info=seq_info.audio_seq_info,
                transform_info=self.transform_config.transform_info[idx],
                path=arr_out_path
            )

            results[transformer.alias()] = audio_file_info

        print(f'*---------------------------------------------------\n'
              f"DBSpectrogramGenerator - Spectrogram gen success\n"
              f"Spectrogram file info. :"
              f"\n - {[results[key] for key in results.keys()]}"
              f'\n---------------------------------------------------*')

        return results



class MusicDetectionDBAudioGenerator(PipelineGenerator):
    """
    Converter for a given database of audio files into raw audio samples
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)

        self.audio_load = Loader(self.pre_proc_config.audio_info)

        custom_mkdir(self.out_dir)

    @classmethod
    def alias(cls):
        return 'music_detection'

    def generate(self):
        meta_list = self._load_vid_meta()

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._load_and_save, meta_list))

        audio_seq_meta = AudioSeqMetadata(
            pre_proc_info=self.pre_proc_config,
            seq_infos=save_meta_list,
            db_sz=self.pre_proc_config.aud_db_sz
        )

        audio_seq_meta.save_to_file(self.out_dir)
        
    def _load_vid_meta(self, type: str = 'all') -> List[AudioFile]:
        aud_meta_list = [m for m in AudioDBMetadata.load_from_file(self.in_dir).audio_file_infos
                         if type == 'all' or m.audio_file_meta.mddi.type == type]
        
        # sorted by uid
        aud_meta_list = sorted(aud_meta_list, key=lambda x: x.audio_file_meta.name)

        if self.pre_proc_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(aud_meta_list) < self.pre_proc_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(aud_meta_list)},'
                                 f'but asked {self.pre_proc_config.aud_db_sz}')

            aud_meta_list = aud_meta_list[:self.pre_proc_config.aud_db_sz]
        else:
            self.pre_proc_config.aud_db_sz = len(aud_meta_list)

        return aud_meta_list

    def _load_and_save(self, meta: AudioFile, save_file_format: str = '{}.npy'):
        arr_out_path = self.out_dir.joinpath(save_file_format.format(meta.audio_file_meta.name))

        # Load audio file..
        y = self.audio_load(Path(meta.path))

        with open(arr_out_path, 'wb') as f:
            np.save(f, y)

        audio_file_info = AudioSeqFileInfo(
            audio_seq_info=AudioSeqInfo(
                audio_info=self.pre_proc_config.audio_info,
                meta=meta
            ),
            path=arr_out_path
        )

        print(f'*---------------------------------------------------\n'
              f"DBAudioGenerator - Audio sequence Saving Success\n"
              f"Video sequence file info. :"
              f"\n - {audio_file_info.audio_seq_info} "
              f"\n - {audio_file_info.path}"
              f'\n---------------------------------------------------*')

        return audio_file_info

  
class MusicDetectionDBSpectrogramGenerator(PipelineSpecGenerator):
    """
    Converter for a given audio files into spectral domain
    """

    DATA_KEY = 'dataset'

    def __init__(self, in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 num_workers: int = 1):
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)

        self.transform = [AudioTransformer.TRANSFORMER[t_config.transform_type](**t_config.param)
                          for t_config in self.transform_config.transform_info]

        self.seq_infos = AudioSeqMetadata.load_from_file(self.in_dir).seq_infos

        if self.transform_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(self.seq_infos) < self.transform_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(self.seq_infos)},'
                                 f'but asked {self.transform_config.aud_db_sz}')

            self.seq_infos = self.seq_infos[:self.transform_config.aud_db_sz]
        else:
            self.transform_config.aud_db_sz = len(self.seq_infos)

        custom_mkdir(self.out_dir)

        for transform in self.transform:
            custom_mkdir(self.out_dir.joinpath(transform.alias()))

    @classmethod
    def alias(cls):
        return 'music_detection'

    def generate(self):
        audio_seq_meta = self.seq_infos

        with Pool(self.num_workers) as pool:
            save_meta_list = list(pool.imap_unordered(self._transform_and_save, audio_seq_meta))

        #
        for key in save_meta_list[0].keys():
            meta_list = [meta[key] for meta in save_meta_list]

            spec_metadata = SpecMetadata(
                pre_proc_info=self.pre_proc_config,
                transform_info=self.transform_config,
                spec_infos=meta_list,
                db_sz=self.transform_config.aud_db_sz
            )

            spec_metadata.save_to_file(self.out_dir.joinpath(key))

    def _transform_and_save(self, seq_info: AudioSeqFileInfo,
                            save_file_format: str = '{}.npy') -> dict:
        # Load audio file..
        audio_seq = AudioSequence.load_with_file_info(seq_info)
        
        results = {}
        for idx, transformer in enumerate(self.transform):
            arr_out_path = (self.out_dir.joinpath(transformer.alias())).joinpath(
                save_file_format.format(seq_info.audio_seq_info.meta.audio_file_meta.name))

            spec = transformer(
                y=audio_seq.arr,
                sr=audio_seq.aud_seq_info.audio_info.sampling_rate
            ).astype('float32')

            with open(arr_out_path, 'wb') as f:
                np.save(f, spec)

            audio_file_info = SpecFileInfo(
                file_info=seq_info.audio_seq_info,
                transform_info=self.transform_config.transform_info[idx],
                path=arr_out_path
            )

            results[transformer.alias()] = audio_file_info

        print(f'*---------------------------------------------------\n'
              f"DBSpectrogramGenerator - Spectrogram gen success\n"
              f"Spectrogram file info. :"
              f"\n - {[results[key] for key in results.keys()]}"
              f'\n---------------------------------------------------*')

        return results