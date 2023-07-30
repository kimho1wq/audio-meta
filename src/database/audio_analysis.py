# Spotify : audio meta extraction
# lib - spotipy==2.18.0

import requests
import json
import os
import csv

from pathlib import Path
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from dataclasses import dataclass

from src.database.data import AudioFile, DBCollector, TrackInfo
from src.pipeline.data import AudioDBConfig, AudioDBMetadata
from src.database.pycuve_class import Pycuve
from src.database.labels import LabelGenerator, DataInfo

from src.database.audio_meta import VibeTrackInfo, VibeAudioFile

SUCCESS = 200

__all__ = [
    'AudioAnalysisDBCollector'
]

class AudioAnalysisDBCollector(DBCollector):
    META_TABLE = ['<trackDownloadUrl>', 'A[', ']']
    INFO_TYPE = 'vibe_track_info'
    AUDIOFILE_TYPE = 'vibe_audio_file'

    DOWNLOAD_API_URL = ''
    META_DIM = 0
    START = 1
    END = 2

    DB_SET = Path('trackIdList_spotify.csv')

    #########################
    # Please do not change the cid & secret
    cid = ''
    secret = ''

    def __init__(self, in_dir: Path, out_dir: Path, audio_db_config: AudioDBConfig, num_workers: int = 5):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.audio_db_config = AudioDBConfig.from_dict(audio_db_config)

        self.in_dir = self.in_dir.joinpath(self.DB_SET)
        self.num_workers = num_workers
        self.pycuve = Pycuve(self.INFO_TYPE)

        # load data..
        with open(self.in_dir, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            self.trackId_list = set(tuple([int(line[0]) for line in rdr if rdr.line_num != 1]))

            # latest track order
            self.trackId_list = tuple(sorted(list(self.trackId_list), reverse=True))

        if self.audio_db_config.aud_db_sz:
            # If number of audios to be loaded (as specified by the parameter)
            # is less than the number of audios specified in the metadata,
            # print a warning message
            if len(self.trackId_list) < self.audio_db_config.aud_db_sz:
                raise ValueError(f'Invalid size of audio db : Audio DB contain only {len(self.trackId_list)},'
                                 f'but asked {self.audio_db_config.aud_db_sz}')

            self.trackId_list = self.trackId_list[:self.audio_db_config.aud_db_sz]
        else:
            self.audio_db_config.aud_db_sz = len(self.trackId_list)

        with ThreadPool(self.num_workers) as pool:
            self.db = list(tqdm(
                        pool.imap_unordered(self.pycuve.get_content_data, self.trackId_list),
                        desc='Loading the track list..',
                        total=len(self.trackId_list)
                    ))
        self.db = [l for l in self.db if l]
        self.pycuve.close()

        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except FileNotFoundError:
            raise ValueError(f'No such file or directory: {str(self.out_dir)}')

        self.label = LabelGenerator.LABELGENERATOR[self.audio_db_config.label.name](self.out_dir,
                                                                                    self.num_workers,
                                                                                    **self.audio_db_config.label.params)

    @classmethod
    def alias(cls):
        return f'audio_analysis'

    def _collection(self, trackInfo: TrackInfo) -> str:
        c_response = requests.get(self.DOWNLOAD_API_URL.format(trackId=trackInfo.trackId))

        if c_response.status_code != SUCCESS:
            print(f'Connection error - trackId : {trackInfo} failed\n'
                  f'response : {c_response.__dict__}')
            return False

        metadata = c_response.content.decode().split('\n')

        meta_FLAG = False
        for line in metadata:
            if self.META_TABLE[self.META_DIM] in line:
                download_url = line[line.find(self.META_TABLE[self.START]) + 2:line.find(self.META_TABLE[self.END])]
                meta_FLAG = True
                break

        return download_url if meta_FLAG else False

    def audio_file_download(self, audio_meta: DataInfo):
        # Retry
        count = 0
        patience = 5
        for _ in range(patience):
            url = self._collection(audio_meta.trackinfo)

            # Download and save
            if url:
                response = requests.get(url, stream=True)

                if response.status_code == SUCCESS:
                    break
            count += 1

        if count == patience:
            print(f'Audio download Failed - Url : {url}\n'
                  f'audio_meta : {audio_meta.to_dict()}\n')
            return False

        with open(self.out_dir.joinpath(str(audio_meta.trackinfo.trackId) + '.mp3'), 'wb') as f:
            f.write(response.content)

        filemeta = AudioFile.AUDIOFILE[self.AUDIOFILE_TYPE]({
            'audio_file_meta': audio_meta.to_dict(),
            'l_alias': audio_meta.alias(),
            'path': str(self.out_dir.joinpath(str(audio_meta.trackinfo.trackId) + '.mp3'))
        })

        with open(self.out_dir.joinpath(str(audio_meta.trackinfo.trackId) + '.json'), 'w') as f:
            json.dump(filemeta.to_dict(), f, indent=4, ensure_ascii=False)

        print(filemeta)
        return filemeta

    def operate(self):
        # Generating labels for each audio input..
        with Pool(self.label.num_workers) as pool:
            track_list = list(pool.imap_unordered(self.label.generate, self.db))
        track_list = [l for l in track_list if l]

        # Downloading Audio files..
        with ThreadPool(self.num_workers) as pool:
            results = list(pool.imap_unordered(self.audio_file_download, track_list))
        results = [r.to_dict() for r in results if r]

        print(f'# of audio files : {len(results)}')

        AudioDBMetadata.from_dict(
            {
                'audio_db_info': self.audio_db_config.to_dict(),
                'dataset': results,
                'f_alias': self.AUDIOFILE_TYPE,
                'db_sz': len(results)
            }
        ).save_to_file(self.out_dir)

