# Spotify : audio meta extraction
# lib - spotipy==2.18.0

import json
import os

from pathlib import Path
from dataclasses import dataclass

from src.database.data import TrackInfo
from src.database.labels.data import DataInfo, LabelGenerator

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

SUCCESS = 200

__all__ = [
    'Spotify',
    'SpotifyMeta'
]


@dataclass
class SpotifyMeta(DataInfo):
    trackinfo: TrackInfo
    s_trackId: str

    def __init__(self, f_dict:dict):
        self.trackinfo = TrackInfo.INFO[f_dict['t_alias']](f_dict['trackinfo'])
        self.s_trackId = f_dict['s_trackId']

    @classmethod
    def from_dict(cls, f_dict: dict):
        return cls(
            trackinfo=TrackInfo.INFO[f_dict['t_alias']](f_dict['trackinfo']),
            s_trackId=f_dict['s_trackId']
        )

    def to_dict(self):
        return {
            'trackinfo': self.trackinfo.to_dict(),
            't_alias': self.trackinfo.alias(),
            's_trackId': self.s_trackId
        }

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return SpotifyMeta.from_dict(self.to_dict())

    @classmethod
    def alias(cls):
        return 'spotify_meta'



class Spotify(LabelGenerator):
    #########################
    # Please do not change the cid & secret
    cid = ''
    secret = ''

    DEFAULT_PATH = 'labels'
    MAXIMUM_WORKERS = 15

    def __init__(self, out_dir: Path, num_workers: int, search_limit: int):
        self.client_credentials_manager = SpotifyClientCredentials(client_id=self.cid, client_secret=self.secret)
        self.spotify = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.search_limit = search_limit

        self.output_dir = out_dir.joinpath(self.DEFAULT_PATH)
        self.num_workers = min(num_workers, self.MAXIMUM_WORKERS)

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except FileNotFoundError:
            raise ValueError(f'No such file or directory: {str(self.output_dir)}\n')

    @classmethod
    def alias(cls):
        return 'spotify'

    def get_metadata_type(self):
        return 'spotify_meta'

    def audio_meta_extraction(self, trackId: SpotifyMeta):
        anal = self.audio_analysis(trackId.s_trackId)
        feats = self.audio_features(trackId.s_trackId)

        if not anal or not feats:
            return False

        res = {
                'feats': feats,
                'anal': anal
              }
        with open(self.output_dir.joinpath(str(trackId.trackinfo.trackId) + '.json'), 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

        return res

    def audio_analysis(self, trackId: str):
        # audio_analysis(track_id)
        # Get audio analysis for a track based upon its Spotify ID Parameters:
        #
        # track_id - a track URI, URL or ID
        audio_anal = False

        try:
            audio_anal = self.spotify.audio_analysis(trackId)
        except spotipy.exceptions.SpotifyException:
            print(f'Audio analysis failed!\n'
                  f'trackId : {trackId}')

        return audio_anal

    def audio_features(self, trackId: str):
        # audio_features(tracks=[])
        # Get audio features for one or multiple tracks based upon their Spotify IDs Parameters:
        #
        # tracks - a list of track URIs, URLs or IDs, maximum: 100 ids
        audio_feats = [False]

        try:
            audio_feats = self.spotify.audio_features(trackId)
        except spotipy.exceptions.SpotifyException:
            print(f'Audio feature extraction failed!\n'
                  f'trackId : {trackId}')

        return audio_feats[0]

    def generate(self, song: TrackInfo):
        try:
            # Find a original track with ISRC
            track_info = self.spotify.search(q='isrc:' + song.isrc, type='track', limit=self.search_limit)

            original = DataInfo.DATAINFO[self.get_metadata_type()]({
                'trackinfo': song.to_dict(),
                't_alias': song.alias(),
                's_trackId': track_info['tracks']['items'][0]['id']
            })
            meta = self.audio_meta_extraction(original)

            if not meta:
                return False
        except:
            print(f'Invalid searching result: {song.to_dict()}')
            return False

        print(original)
        return original
