import numpy as np

from assets.config.config import PreProcConfig, TransformConfig, ExtractorConfig

SEQ_DIM = 0
FEAT_DIM = 1


def get_input_shape(pre_proc_config: PreProcConfig, transform_config: TransformConfig, extractor_config: ExtractorConfig) -> tuple:

    PARAMETERS = {
        'chromagram': 'n_chroma',
        'cqt': 'n_bins',
        'tempogram': 'window_size',
        'spectrogram': 'numofband'
    }

    # abbreviated calculation
    seq_len = int((extractor_config.max_length_limit_ms // 1000 * pre_proc_config.audio_info.sampling_rate) // \
                  transform_config.transform_info[0].param['hop_size'] + 1)

    base_config = extractor_config.cnn_config[0]

    for t_info in transform_config.transform_info:
        if base_config.type == t_info.transform_type:
            feats = t_info.param[PARAMETERS[base_config.type]]

            for s in base_config.strides:
                feats /= s[FEAT_DIM]
            break
    return [seq_len, int(feats)]


def fill_input_arr(seq_len: int, input_arr):
    tmp_arr = np.zeros((seq_len, input_arr.shape[FEAT_DIM]))

    # IF the length of input audio is longer than threshold, exclude that audio
    if input_arr.shape[SEQ_DIM] > seq_len:
        raise ValueError(f'Invalid length of audio seq, It must be shorter than threshold')

    tmp_arr[:input_arr.shape[SEQ_DIM]] = input_arr
    return np.expand_dims(tmp_arr, axis=0).astype('float32')

def _c_to_r(_input, q_index):
    def softmax(value):
        v = value - np.max(value)
        return np.exp(v) / np.sum(np.exp(v))

    v = 1. / q_index
    q_depth = np.arange(q_index) * v
    _in = softmax(_input) * q_depth
    return np.sum(_in)