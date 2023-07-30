import io
import numpy as np
from pydub import AudioSegment

from pathlib import Path
from assets.config.config import AudioInfo


__all__ = [
    'Loader'
]


class Loader:
    """
    Audio data loader
    """
    SAMPLING_RATE_TABLE = [
        8000,
        11025,
        16000,
        22050,
        32000,
        44100,
        48000
    ]

    def __init__(self, audio_info: AudioInfo):
        self.audio_info = audio_info

        if self.audio_info.sampling_rate not in self.SAMPLING_RATE_TABLE:
            raise ValueError(f'Invalid sampling rate : must be selected within the \n'
                             f'table : {self.SAMPLING_RATE_TABLE}\n'
                             f'but got {self.sampling_rate}.')

    def __call__(self, audio):
        return self.__load(audio)

    def __load(self, audio) -> np.ndarray:
        if isinstance(audio, Path):
            if audio.suffix == '.mp3':
                y = AudioSegment.from_mp3(audio)
            elif audio.suffix == '.wav':
                y = AudioSegment.from_wav(audio)
            else:
                raise ValueError(f'Invalid suffix : must be selected within the \n'
                                 f'suffix : [.mp3, .wav]\n'
                                 f'but got {audio.suffix}.')
        elif isinstance(audio, bytes) or isinstance(audio, bytearray):
            y = AudioSegment.from_file(io.BytesIO(audio))
        elif isinstance(audio, io.BytesIO):
            y = AudioSegment.from_file(audio)
        else:
            raise ValueError(f'Invalid instance : must be selected within the \n'
                             f'instance : [Path, bytes, bytearray, io.BytesIO]')

        if self.audio_info.is_mono:
            y = y.set_channels(1)
        if y.frame_rate != self.audio_info.sampling_rate:
            y = y.set_frame_rate(self.audio_info.sampling_rate)
        y, sr = np.asarray(y.get_array_of_samples()).astype('float32') / 32767., y.frame_rate

        return y
