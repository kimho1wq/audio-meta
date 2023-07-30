import numpy as np
import librosa

from .data import AudioTransformer

__all__ = [
    'Spectrogram'
]


class Spectrogram(AudioTransformer):
    """
    Transform to log spectrogram
    If is_mel is set to True, returns the log mel-spectrogram
    """

    def __init__(self, window_size: int, hop_size: int, is_mel: bool = False, numofband: int = 40):
        #
        self.window_size = window_size
        self.hop_size = hop_size
        self.is_mel = is_mel
        self.numofband = numofband

    @classmethod
    def alias(cls):
        return 'spectrogram'

    def transform(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        # librosa uses hanning window as the default window
        """
        if self.is_mel:
            spec = librosa.feature.melspectrogram(y, n_mels=self.numofband, n_fft=self.window_size,
                                                  hop_length=self.hop_size)
            log_spec = librosa.power_to_db(spec)
        else:
            spec = np.abs(librosa.stft(y, n_fft=self.window_size, hop_length=self.hop_size))
            log_spec = librosa.amplitude_to_db(spec)

        # shape : seq x feat
        return log_spec.T
