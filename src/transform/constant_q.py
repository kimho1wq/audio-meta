import numpy as np
import librosa

from .data import AudioTransformer

__all__ = [
    'ConstantQ'
]


class ConstantQ(AudioTransformer):
    """
    Transform to constant_q
    """

    def __init__(self, hop_size: int, n_bins: int = 84, bins_per_octave: int = 12):
        #
        self.n_bins = n_bins
        self.hop_size = hop_size
        self.bins_per_octave = bins_per_octave

        assert self.hop_size % (2**(self.n_bins / self.bins_per_octave)) == 0

    @classmethod
    def alias(cls):
        return 'cqt'

    def transform(self, y: np.ndarray, sr: int = 32000) -> np.ndarray:
        """
        # librosa uses hanning window as the default window
        """

        spec = np.abs(librosa.cqt(y=y, sr=sr, hop_length=self.hop_size,
                                  n_bins=self.n_bins, bins_per_octave=self.bins_per_octave))

        # shape : seq x feat
        return librosa.amplitude_to_db(spec).T
