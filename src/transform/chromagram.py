import librosa
import numpy as np

from .data import AudioTransformer

__all__ = [
    'Chromagram'
]


class Chromagram(AudioTransformer):
    """
    Transform to chromagram
    """

    def __init__(self, window_size: int, hop_size: int, n_chroma: int, is_cqt:bool=False):
        #
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_chroma = n_chroma
        self.is_cqt=is_cqt

    @classmethod
    def alias(cls):
        return 'chromagram'

    def transform(self, y: np.ndarray, sr: int = 32000) -> np.ndarray:
        """
        # librosa uses hanning window as the default window
        """

        if self.is_cqt:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=self.n_chroma, hop_length=self.hop_size)
        else:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma,
                                                 n_fft=self.window_size, hop_length=self.hop_size)

        # shape : seq x feat
        return chroma.T
