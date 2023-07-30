import os

from pathlib import Path
from multiprocessing import Pool
from functools import partial
from dataclasses import dataclass

from src.database.data import TrackInfo, AudioFile, DBCollector
from src.pipeline.data import AudioDBConfig, AudioDBMetadata
from src.database.labels.data import DataInfo


SUCCESS = 200

__all__ = [
    'MusicDetectionDBCollector',
    'MusicDetectionDataInfo',
    'MusicDetectionMeta'
]


@dataclass
class MusicDetectionDataInfo(TrackInfo):
    download_url: str
    target_path: str
    zipped_name: str
    root: str
    type: str

    def __init__(self, f_dict: dict):
        self.download_url = f_dict['download_url']
        self.target_path = f_dict['target_path']
        self.zipped_name = f_dict['zipped_name']
        self.root = f_dict['root']
        self.type = f_dict['type']

    @classmethod
    def alias(cls):
        return 'md_track_info'

    def to_dict(self):
        return {
            'download_url': self.download_url,
            'target_path': self.target_path,
            'zipped_name': self.zipped_name,
            'root': self.root,
            'type': self.type
        }

@dataclass
class MusicDetectionMeta(DataInfo):
    mddi: TrackInfo
    name: str

    def __init__(self, f_dict: dict):
        self.mddi=TrackInfo.INFO[f_dict['t_alias']](f_dict['mddi'])
        self.name=f_dict['name']

    @classmethod
    def alias(cls):
        return 'md_meta'

    def to_dict(self):
        return {
            'mddi': self.mddi.to_dict(),
            'name': self.name,
            't_alias': self.mddi.alias()
        }

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            mddi=TrackInfo.INFO[f_dict['t_alias']](f_dict['mddi']),
            name=f_dict['name']
        )

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return MusicDetectionMeta.from_dict(self.to_dict())

@dataclass
class MDAudioFile(AudioFile):
    audio_file_meta: DataInfo
    path: str

    def __init__(self, f_dict: dict):
        self.audio_file_meta = DataInfo.DATAINFO[f_dict['l_alias']](f_dict['audio_file_meta'])
        self.path = f_dict['path']

    @classmethod
    def alias(cls):
        return 'md_audio_file'

    def to_dict(self):
        return {
            'audio_file_meta': self.audio_file_meta.to_dict(),
            'path': self.path,
            'l_alias': self.audio_file_meta.alias()
        }

class MusicDetectionDBCollector(DBCollector):
    INFO_TYPE = 'md_track_info'
    DATAINFO_TYPE = 'md_meta'
    AUDIOFILE_TYPE = 'md_audio_file'

    MDDI_LIST = [
        TrackInfo.INFO[INFO_TYPE]({
            'root': 'ESC-50',
            'target_path': 'ESC-50-master/audio',
            'zipped_name': 'master.zip',
            'download_url': '',
            'type': 'noise'
        }),
        TrackInfo.INFO[INFO_TYPE]({
            'root': 'GTZAN',
            'target_path': 'GTZAN-main/audio',
            'zipped_name': 'main.zip',
            'download_url': '',
            'type': 'music'
        }),
        TrackInfo.INFO[INFO_TYPE]({
            'root': 'MUSAN',
            'target_path': 'MUSAN-main/speech/librivox',
            'zipped_name': 'main.zip',
            'download_url': '',
            'type': 'speech'
        }),
        TrackInfo.INFO[INFO_TYPE]({
            'root': 'ntv_audios',
            'target_path': 'ntv_audios-main/subset_100',
            'zipped_name': 'main.zip',
            'download_url': '',
            'type': 'music'
        }),
        TrackInfo.INFO[INFO_TYPE]({
            'root': 'ntv_audios_testset',
            'target_path': 'ntv_audios-main/ntv_audio_testset',
            'zipped_name': 'main.zip',
            'download_url': '',
            'type': 'testset'
        })
    ]


    def __init__(self, in_dir: Path = None, out_dir: Path = None, audio_db_config: AudioDBConfig = None,
                 num_workers: int = 5):
        self.out_dir = out_dir
        self.audio_db_config = AudioDBConfig.from_dict(audio_db_config)
        self.num_workers = num_workers

        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except FileNotFoundError:
            raise ValueError(f'No such file or directory: {str(self.out_dir)} \n')


    def audio_file_download(self, mddi: TrackInfo):
        target_path = Path(mddi.root).joinpath(mddi.target_path)
        
        if not os.path.exists(mddi.root):
            os.system(f"wget {mddi.download_url} -P {mddi.root}")
            os.system(f"unzip {mddi.root}/{mddi.zipped_name} -d {mddi.root}")

        if mddi.type == 'testset':
            label_file = 'vid_music_label_refined.csv'
            src = target_path.joinpath(label_file)
            dst = self.out_dir.joinpath(label_file)

            os.system(f"mv {src} {dst}")
            target_path = target_path.joinpath('audio')

        target_list = os.listdir(target_path)
        if self.audio_db_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(target_list) < self.audio_db_config.aud_db_sz:
                print(f'Warning:  invalid size of audio db : Audio DB contain only {len(target_list)},'
                      f'but asked {self.audio_db_config.aud_db_sz}')
                self.audio_db_config.aud_db_sz = len(target_list)
            target_list = target_list[:self.audio_db_config.aud_db_sz]

        else:
            self.audio_db_config.aud_db_sz = len(target_list)
            target_list = target_list[:self.audio_db_config.aud_db_sz]
            
            
        results = []
        for target in target_list:
            src = target_path.joinpath(target)
            dst = self.out_dir.joinpath(target)
            
            os.system(f"mv {src} {dst}")

            meta = DataInfo.DATAINFO[self.DATAINFO_TYPE]({
                    'mddi': mddi.to_dict(),
                    't_alias': mddi.alias(),
                    'name': str(os.path.splitext(target)[0])
                   })

            filemeta = AudioFile.AUDIOFILE[self.AUDIOFILE_TYPE]({
                'audio_file_meta': meta.to_dict(),
                'l_alias':meta.alias(),
                'path': str(dst)
            }).to_dict()
            results.append(filemeta)
            
        os.system(f"rm -rf {mddi.root}")

        return results

    @classmethod
    def alias(cls):
        return f'music_detection'

    def operate(self):
        results = []
        with Pool(self.num_workers) as pool:
            res = list(pool.map(func=partial(self.audio_file_download), iterable=self.MDDI_LIST))
        for r in res:
            results.extend(r)

        print(results)
        print(f'# of audio files : {len(results)}')
       
        AudioDBMetadata.from_dict(
            {
                'audio_db_info': self.audio_db_config.to_dict(),
                'dataset': results,
                'f_alias': self.AUDIOFILE_TYPE,
                'db_sz': len(results)
            }
        ).save_to_file(self.out_dir)
