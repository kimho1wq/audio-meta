import numpy as np
import librosa
import scipy
import warnings
import scipy.stats


__all__ = [
    'Analysis'
]


class Analysis:

    def __call__(self,analysis_type, input, analysis_param):
        method_name='predict_{}'.format(analysis_type)
        pred=getattr(self, method_name)(input, analysis_param)

        return pred

    def predict_loudness(self, input_path:str) -> dict:
        '''

        :param input_path: input mp3 path for ffmpeg command
        :return: dict type
            key: loudness : float
        '''
        raise NotImplementedError

    def predict_key(self,x:np.ndarray, param:dict=None) -> dict:
        '''

        :param x: np.ndarray, shape=(audio_sequence_length, 12), chromagram extracted from transform operation
        :return: dict type
            key1: pred_mode : int
            key2: pred_key : int

                if pred_mode is 1, it means that key is major
                if pred_mode is 0, it means that key is minor

                pred_key is represented with index of pitch class notation
                ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        '''


        assert x.shape[1]==12, "chromagram dimension must be 12 for key prediction"

        # Coefficients from Kumhansl and Schmuckler
        # as reported here: http://rnhart.net/articles/key-finding/
        MINOR_COEFF = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        MAJOR_COEFF = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

        x = np.sum(x, axis=0).T
        x = scipy.stats.zscore(x)

        minor = scipy.stats.zscore(MINOR_COEFF)
        major = scipy.stats.zscore(MAJOR_COEFF)

        # Generate all rotations of major
        minor = scipy.linalg.circulant(minor)
        major = scipy.linalg.circulant(major)

        concat = np.concatenate([minor.T.dot(x), major.T.dot(x)])
        argmax = np.argmax(concat)

        pred_mode, pred_key = divmod(argmax, 12)

        prediction={'key': int(pred_key),'mode':int(pred_mode)}

        return prediction

    def predict_tempo(self, x:np.ndarray, param:dict) -> dict:
        '''
        :param x : tempogram
        '''
        start_bpm=param['start_bpm']
        std_bpm=param['std_bpm']
        max_tempo=param['max_tempo']
        hop_length=param['hop_size']
        sr=param['sampling_rate']

        tempogram = np.mean(x.T, axis=1, keepdims=True)

        # Get the BPM values for each bin, skipping the 0-lag bin
        bpms = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)

        # Weight the autocorrelation by a log-normal distribution
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2

        # Kill everything above the max tempo
        max_idx = np.argmax(bpms < max_tempo)
        logprior[:max_idx] = -np.inf

        # Get the maximum, weighted by the prior
        # Using log1p here for numerical stability
        best_period = np.argmax(np.log1p(1e6 * tempogram) + logprior[:, np.newaxis], axis=0)

        tempo=bpms[best_period][0]

        prediction={'tempo':float('{:.3f}'.format(tempo))}

        return prediction

    def predict_pitch(self,x, param:dict)->dict:
        '''

        :param x : chromagram

        '''

        assert x.shape[1] == 12, "chromagram dimension must be 12 for key prediction"

        n_frame = x.shape[0]
        hop_length = param['hop_size']
        sr = param['sampling_rate']
        segment_duration=param['segment_duration'] #sec

        frame_length = (hop_length/(sr/1000)) / 1000
        frame_per_segment = round(segment_duration / frame_length)

        time_indicator=librosa.frames_to_time(np.arange(n_frame),sr=sr,hop_length=hop_length)

        prediction={}

        start_idx, loop_idx = 0,0
        while True:

            if start_idx+frame_per_segment >= n_frame:
                break

            end_idx=start_idx+frame_per_segment

            target=x[start_idx:end_idx,:]
            pitches=np.mean(target,axis=0).tolist()

            segment_key = 'segment{}'.format(loop_idx)

            dict_to_write={segment_key:
                            {"start":float('{:.3f}'.format(time_indicator[start_idx])),
                            "end":float('{:.3f}'.format(time_indicator[end_idx])),
                            "pitches":list(np.round(pitches,3))}
                           }

            prediction.update(dict_to_write)

            start_idx+=frame_per_segment
            loop_idx+=1


        return prediction

    def predict_time_signature(self, x) -> dict:
        raise NotImplementedError

