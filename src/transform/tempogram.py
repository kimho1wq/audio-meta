import librosa
import numpy as np

from .data import AudioTransformer

__all__ = [
    'Tempogram'
]


class Tempogram(AudioTransformer):
    """
    Transform to tempogram
    """

    def __init__(self, window_size: int, hop_size: int, min_max_tempo: list, onset_aggregate:str=None):
        #
        self.window_size = window_size
        self.hop_size = hop_size
        self.min_max_tempo = min_max_tempo
        self.aggregate = onset_aggregate #np.median for tempo extraction

    @classmethod
    def alias(cls):
        return 'tempogram'

    def transform(self, y: np.ndarray, sr: int = 32000) -> np.ndarray:
        """
        # librosa uses hanning window as the default window
        """

        if self.aggregate=='median':
            oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

        else:
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_size, win_length=self.window_size)

        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                              win_length=self.window_size, hop_length=self.hop_size)

        # shape : seq x feat
        return tempogram[self.min_max_tempo[0]:self.min_max_tempo[1]].T
