from __future__ import annotations

import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from typing import List

from src.audio import SpecFileInfo, Spectrogram
from src.pipeline import SpecMetadata
from src.extraction.dataset import MusicDetectionDataset

from src.extraction.models.music_detection import MusicDetectionNetwork
from src.extraction.data import Network, Trainer

from src.extraction.analysis import Analysis
from src.extraction.data import Extractor
from src.utils.data import custom_mkdir, SummaryResult
from assets.config.config import PreProcConfig, TransformConfig, ExtractorConfig, AnalysisExtractorConfig, MusicDetectionConfig
from src.extraction.utils import get_input_shape, fill_input_arr, _c_to_r

__all__ = [
    'AudioMetaExtractor',
    'AudioAnalysisExtractor',
    'MusicDetection'
]


class AudioMetaExtractor(Extractor):
    """
    Extractor extracts audio metadata from a given audio file
    """
    DATA_KEY = 'dataset'

    NETWORK_PATH = Path("assets/network")
    DEFAULT_NETWORK_FILE_NAME = Path("final_epoch.pth")

    DEFAULT_CORE_TEST_SET = Path('assets/core_test_set')

    SEQ_DIM = 0
    FEAT_DIM = 1

    REGRESSION_MODEL_PATH = NETWORK_PATH.joinpath('regression')
    CLASSIFICATION_MODEL_PATH = NETWORK_PATH.joinpath('classification')

    MULTI_AUDIO_META = 'multi_audio_meta'

    def __init__(self, x_in_dir: Path, y_in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 extractor_config: dict, num_workers: int = 1):
        self.x_in_dir = x_in_dir
        self.y_in_dir = y_in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)
        self.extractor_config = ExtractorConfig.from_dict(extractor_config)
        self.input_shape = get_input_shape(self.pre_proc_config, self.transform_config, self.extractor_config)

        self.core_test_set = self.DEFAULT_CORE_TEST_SET.joinpath(self.extractor_config.result_type)

        self.pipeline_version = Path(self.out_dir.parts[-2])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """
        # building the each network with Abstract class (Network class)
        """
        self.audio_meta = [feat for feat in self.extractor_config.task_config.task_type.single_task]
        if len(self.extractor_config.task_config.task_type.multi_task):
            self.audio_meta.append(self.MULTI_AUDIO_META)

        self.audio_meta_model = {feat: self._build_network(feat)
                                 for feat in self.audio_meta}

        """
        # calling the each trainer with Abstract class (Trainer class)
        """
        self.trainer = {feat: Trainer.TRAINER[feat](self.audio_meta_model[feat], self.pre_proc_config,
                                                    self.transform_config, self.extractor_config)
                              for feat in self.audio_meta}

        self._build(self.REGRESSION_MODEL_PATH if self.extractor_config.task_config.task_type.type[0] == 'r'
                    else self.CLASSIFICATION_MODEL_PATH)

    @classmethod
    def alias(cls):
        return 'audio_meta'

    def extract(self, save_format: str = '{}.json'):
        spec_metadata = self._load_meta()
        for meta in self.audio_meta:
            self.audio_meta_model[meta].eval()

        idx_arr = self.create_index_arr(spec_metadata[0].db_sz)
        seq_len = self.input_shape[0]

        _dir = 'classification/' if self.extractor_config.task_config.task_type.type[0] == 'c' else ''
        inference_path = self.out_dir.joinpath(_dir + self.extractor_config.result_type).joinpath('inference')
        custom_mkdir(inference_path)
        
        for idx in idx_arr:
            try:
                x = {
                        meta.spec_infos[idx].transform_info.transform_type: fill_input_arr(seq_len, self._load_input_arr(meta, idx))
                        for meta in spec_metadata
                    }

                audio_m = self.inference(x)
                res = {'trackInfo': spec_metadata[0].spec_infos[idx].file_info.meta.audio_file_meta.trackinfo.to_dict()}
                res.update({'features': audio_m})

                trackId = res['trackInfo']['trackId']
                with open(inference_path.joinpath(save_format.format(trackId)), 'w') as f:
                    json.dump(res, f, indent=4, ensure_ascii=False)

            except ValueError:
                continue
        audio_features = self.extractor_config.task_config.task_type.single_task + \
                         self.extractor_config.task_config.task_type.multi_task
        SummaryResult.RESULT[self.extractor_config.result_type](
            inference_path,
            self.core_test_set.joinpath('labels'),
            inference_path.parent,
            audio_features if self.extractor_config.result_type[0] == 'p' \
            else self.extractor_config.task_config.task_type.multi_task
        ).summary()

        return True

    def inference(self, x):
        with torch.no_grad():
            res = {}
            for meta in self.audio_meta_model:
                if meta == self.MULTI_AUDIO_META:
                    r = self.audio_meta_model[meta]({key: torch.from_numpy(x[key]).to(self.device)
                                                     for key in x.keys()}).detach().cpu()[0]
                    for i, feat in enumerate(self.extractor_config.task_config.task_type.multi_task):
                        res[feat] = float(r[i]) if self.extractor_config.task_config.task_type.type[0] == 'r' \
                            else float(_c_to_r(r[i], self.extractor_config.task_config.q_index))
                else:
                    r = self.audio_meta_model[meta](torch.from_numpy(x['spectrogram']).
                                                    to(self.device)).detach().cpu()[0]
                    res[meta] = float(r)
            return res

    def _load_meta(self):
        return [SpecMetadata.load_from_file(self.core_test_set.joinpath(_type.type))
                                            for _type in self.extractor_config.cnn_config]

    def _load_input_arr(self, meta, idx:int):
        PARAMETERS = {
            'chromagram': 1.,
            'cqt': 40.,
            'tempogram': 1.,
            'spectrogram': 40.
        }

        try:
            _path = Path(meta.spec_infos[idx].path).parts[-2:]
            meta.spec_infos[idx].path = self.core_test_set.joinpath(_path[0] + '/' + _path[1])

            input_arr = Spectrogram.load_with_file_info(meta.spec_infos[idx]).arr / \
                        PARAMETERS[meta.spec_infos[idx].transform_info.transform_type]
        except:
            raise ValueError(f'Failed load audio data..')

        return input_arr

    def create_index_arr(self, length: int):
        return np.arange(length)

    def _build_network(self, feat: str):
        """
        # build a new audio meta network
        """
        return Network.NETWORK[feat](self.extractor_config, self.input_shape).to(self.device)

    def _build(self, path: Path) -> int:
        for feat in self.audio_meta:
            if feat == self.MULTI_AUDIO_META:
                final_path = path.joinpath(self.pipeline_version).joinpath(self.DEFAULT_NETWORK_FILE_NAME)
            else:
                final_path = self.NETWORK_PATH.joinpath(feat).joinpath(self.pipeline_version).joinpath(
                    self.DEFAULT_NETWORK_FILE_NAME)

            if os.path.exists(final_path):
                self.load_network(final_path, feat)
            else:
                if feat == self.MULTI_AUDIO_META:
                    self.trainer[feat].train(input_path=self.x_in_dir, label_path=self.y_in_dir)
                else:
                    self.trainer[feat].train()
                custom_mkdir(final_path.parent)
                self.trainer[feat].save_network(final_path)

    def load_network(self, net_path: Path, feat: str):
        """
        # load the pre-trained network
        """
        self.audio_meta_model[feat] = torch.jit.load(net_path, map_location=self.device)
        if torch.cuda.is_available():
            self.audio_meta_model[feat] = nn.DataParallel(self.audio_meta_model[feat]).to(self.device)


class AudioAnalysisExtractor(Extractor):
    """
    Extractor extracts audio metadata from a given audio file
    """
    DATA_KEY = 'dataset'

    def __init__(self, x_in_dir: Path, y_in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 extractor_config: dict, num_workers: int = 1):
        self.x_in_dir = x_in_dir
        self.y_in_dir = y_in_dir
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)
        self.extractor_config = AnalysisExtractorConfig.from_dict(extractor_config)
    
        self.analyzer = Analysis()
        self.pipeline_version = Path(self.out_dir.parts[-2])


    @classmethod
    def alias(cls):
        return 'audio_analysis'

    def extract(self, save_format: str = '{}.json'):

        spec_metadata = self._load_meta()

        idx_arr = self.create_index_arr(len(spec_metadata[0].spec_infos))
        # FIXME : bug fix required, db_sz doesn't match with actual number of transformed data

        inference_path = self.out_dir.joinpath('inference')
        custom_mkdir(inference_path)

        for idx in idx_arr:
            try:
                res = {'trackInfo': spec_metadata[0].spec_infos[idx].file_info.meta.audio_file_meta.trackinfo.to_dict()}
                res.update({'feature':{}})
                res.update({'anal':{}})
                for transform_idx, _anal in enumerate(self.extractor_config.analysis_info):

                    x = Spectrogram.load_with_file_info(spec_metadata[transform_idx].spec_infos[idx]).arr

                    audio_feature=_anal.audio_feature

                    # for signal_processing based approach, feature-dependent parameters are required
                    # Thus, all parameters (preprocessing, transform for each meta) are merged

                    anal_param = _anal.param
                    transform_param=spec_metadata[transform_idx].spec_infos[idx].transform_info.param
                    print('transform_param:')
                    print(transform_param)
                    anal_param.update(transform_param)
                    anal_param.update(self.pre_proc_config.audio_info.to_dict())
                    pred=self.analyzer(audio_feature,x,anal_param)
                    if audio_feature=='pitch':
                        res['anal'].update(pred)
                    else:
                        res['feature'].update(pred)

                trackId = res['trackInfo']['trackId']
                with open(inference_path.joinpath(save_format.format(trackId)), 'w') as f:
                    json.dump(res, f, indent=4, ensure_ascii=False)

            except ValueError:
                continue


        return True

    def _load_meta(self):
        meta=[]
        for _type in self.extractor_config.analysis_info:
            meta.append(SpecMetadata.load_from_file(self.x_in_dir.joinpath(_type.target_transform)))

        return meta


    def create_index_arr(self, length: int):
        return np.arange(length)



class MusicDetection(Extractor):
    """
    Extractor extracts audio metadata from a given audio file
    """
    DATA_KEY = 'dataset'
    DEFAULT_PATH = Path("assets/network")
    DEFAULT_NETWORK_FILE_NAME = Path("music_detection_final.pt")

    def __init__(self, x_in_dir: Path, y_in_dir: Path, out_dir: Path, pre_proc_config: dict, transform_config: dict,
                 extractor_config: dict, num_workers: int = 1):
        self.x_in_dir = x_in_dir
        self.test_y_in_dir = y_in_dir
        self.out_dir = out_dir


        self.num_workers = num_workers
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)
        self.extractor_config = MusicDetectionConfig.from_dict(extractor_config)
        
        network_path = self.DEFAULT_PATH.joinpath(self.out_dir.name).joinpath(self.DEFAULT_NETWORK_FILE_NAME)
        self.device = "cuda" if torch.cuda.is_available() and self.extractor_config.num_gpus > 0 else "cpu"

        if os.path.exists(network_path):
            self.model = self.load_network(network_path)
        else:
            custom_mkdir(network_path.parent)
            train_meta_list, test_meta_list = self._load_meta(is_train=True)
            self.train(train_meta_list, str(network_path))
            self.test(test_meta_list)

        custom_mkdir(self.out_dir)

    @classmethod
    def alias(cls):
        return 'music_detection'

    def extract(self, threshold: float = .5, hop_size: float = .5, save_format: str = '{}.json'):
        #meta_list, _ = self._load_meta()
        _, meta_list = self._load_meta(is_train=True)

        for i, meta in enumerate(meta_list[0]):
            time_table = [[]]
            input_batches = self.preprocess(Spectrogram.load_with_file_info(meta).arr.T)
            preds = self.inference(input_batches)
            preds_binary = (preds > threshold).astype(int)
            frame_time = np.arange(start=hop_size, stop=len(preds)*hop_size, step=hop_size)
            if preds_binary[0] == 1:
                time_table[0].append(0)

            for i, time in enumerate(frame_time):
                if preds_binary[i-1] != preds_binary[i]:
                    time_table[0].append(time)

            if len(time_table) % 2 != 0:
                time_table[0].append(len(preds)*hop_size)

            results = {
                    'musicTimeTable': time_table,
                    'name': meta.file_info.meta.audio_file_meta.name,
                    'fileInfo': meta.file_info.meta.audio_file_meta.mddi.to_dict()
                }

            with open(self.out_dir.joinpath(save_format.format(meta.file_info.meta.audio_file_meta.name)), 'w') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)


    def train(self, meta_list: List[SpecFileInfo] , network_path: str):
        train_loader, val_loader = self.create_loaders(meta_list)

        sample_batch = next(iter(val_loader))
        self.model = self.get_model(sample_batch, network_path)
        print(self.model)

        pl.Trainer(max_epochs=self.extractor_config.num_epochs,
            gpus=self.extractor_config.num_gpus,
            check_val_every_n_epoch=self.extractor_config.check_val_every_n_epoch
        ).fit(self.model, train_loader, val_loader)


    def get_model(self, sample_batch, network_path:str):
        # Ex) sample_batch.shape : torch.Size([batch, 1, 64, 101])
        dim = sample_batch.get('spectrogram').shape
        return MusicDetectionNetwork(self.extractor_config, n_bins=dim[-2], n_frames=dim[-1], network_path=network_path)

    def load_network(self, network_path:str):
        return torch.jit.load(network_path, map_location=self.device)

    def create_loaders(self, meta_list: List[SpecFileInfo], ratio: float = 0.8):
        inputs = {tf_type.type: [] for tf_type in self.extractor_config.cnn_config}
        labels = []

        for idx in np.arange(len(meta_list[0])):
            labels.append(meta_list[0][idx].file_info.meta.audio_file_meta.mddi.type)
            for i, tf_type in enumerate(self.extractor_config.cnn_config):
                spec = Spectrogram.load_with_file_info(meta_list[i][idx]).arr
                inputs[tf_type.type].append(spec.astype('float32'))

       
        boundary = int(len(meta_list[0]) * ratio)
        train_data = [{key:inputs[key][:boundary] for key in inputs.keys()},
                      labels[:boundary], self.extractor_config.hop_length]
        validation_data = [{key:inputs[key][boundary:] for key in inputs.keys()},
                           labels[boundary:], self.extractor_config.hop_length]

        train_loader = DataLoader(MusicDetectionDataset(*train_data),
            batch_size=self.extractor_config.batch_size,
            shuffle=True,
            num_workers=self.extractor_config.num_workers,
            drop_last=False)
        val_loader = DataLoader(MusicDetectionDataset(*validation_data),
            batch_size=self.extractor_config.batch_size,
            shuffle=True,
            num_workers=self.extractor_config.num_workers,
            drop_last=False)

        return train_loader, val_loader


    def _load_meta(self, is_train: bool = False):
        meta_list = []
        test_meta_list = []
        for _type in self.extractor_config.cnn_config:
            file_infos = []
            test_file_infos = []
            metadata = SpecMetadata.load_from_file(self.x_in_dir.joinpath(_type.type)).spec_infos
            for m in metadata:
                test_file_infos.append(m) if m.file_info.meta.audio_file_meta.mddi.type == 'testset' \
                        else file_infos.append(m) if is_train else file_infos.append(m)
                
            meta_list.append(file_infos)
            test_meta_list.append(test_file_infos)

        return meta_list, test_meta_list

    def preprocess(self, input):
        FEAT_DIM = 0
        SEQ_DIM = 1

        num_batches = int(input.shape[SEQ_DIM]/ self.extractor_config.hop_length) + 1
        pad_size = int((num_batches * self.extractor_config.hop_length - input.shape[SEQ_DIM]) / 2)
        input_padded = np.pad(input, ((0, 0), (pad_size, pad_size+1)), 'constant', constant_values=0)
        
        input_batches = np.empty((num_batches, input.shape[FEAT_DIM], self.extractor_config.hop_length))
        for i in range(num_batches):
            input_batches[i, :, :] = input_padded[:, i:i+self.extractor_config.hop_length]
        input_batches = torch.from_numpy(input_batches).type(torch.float32).unsqueeze(1)

        return self.divide_into_batches(input_batches)

    def inference(self, input_batches):
        self.model.to(self.device)
        self.model.eval()
        preds = np.array([])
        with torch.no_grad():
            for x in input_batches:
                out = self.model(x.to(self.device))[0].cpu().view(-1).detach().numpy()
                preds = np.concatenate((preds, out), axis=0)

        return preds


    def test(self, meta_list: List[SpecFileInfo], hop_size: float = .5,
                      thresholds: np.ndarray = np.linspace(.1, .9, 9)):
        with open(self.test_y_in_dir.joinpath('vid_music_label_refined.csv')) as f:
            label_dict = {sample[0]: sample[1:] for sample in [row for row in csv.reader(f)][1:]}

        num_correct_total = 0
        len_total = 0
        for i, meta in enumerate(meta_list[0]):
            accuracies = []
            input_batches = self.preprocess(Spectrogram.load_with_file_info(meta).arr.T)
            preds = self.inference(input_batches)

            # (frame, )
            labels = self.transform_labels_to_frames(
                label_dict[meta.file_info.meta.audio_file_meta.name],
                length=len(preds),
                feature_size=1,
                hop_size=hop_size)

            # raw prediction, binary prediction, filtered prediction, and
            # accuracy for range of thresholds are plotted
            accuracy_best = 0.
            for idx, threshold in enumerate(thresholds):
                preds_binary = (preds > threshold) .astype(float)    
                preds_filtered = self.filter_by_median(
                    preds_binary,
                    hop_size=hop_size)
                num_correct = (preds_filtered == labels).sum()
                if threshold == .5:  # overall accuracy calculated with .5 threshold
                    num_correct_total += num_correct
                    len_total += len(preds_binary)
                accuracy = num_correct / len(preds_binary)
                if accuracy_best < accuracy:
                    threshold_best = threshold
                    accuracy_best = accuracy
                accuracies.append(accuracy)
 
        overall_acc = num_correct_total / len_total
        print(f'Overall accuracy for: {overall_acc}')

        return overall_acc


    def filter_by_median(self, preds_binary: np.ndarray,
                     hop_size: float,
                     filter_size_time: float = 5.0) -> np.ndarray:

        filter_size = int(filter_size_time / hop_size)
        if filter_size % 2 == 0:  # median filter size must always be an odd number
            filter_size = filter_size + 1
        pad_size = filter_size // 2

        preds_binary_padded = np.pad(preds_binary, (pad_size, pad_size + 1),
                                 mode='constant', constant_values=np.NaN)
        result = np.empty(preds_binary.shape)
        for i in range(len(preds_binary)):
            filtered_pred = np.nanmedian(preds_binary_padded[i: i + filter_size])
            result[i] = filtered_pred if filtered_pred > 0 else 0

        return result


    def transform_labels_to_frames(self, sample: list, length: int, hop_size: float, feature_size: float) -> np.ndarray:
        labels_by_frame = []
        center_idx = int(feature_size / 2)
        # index 0 contains file id
        if sample[0] == '0':
            labels_by_frame.append(0)
        else:
            labels_by_frame.append(int((int(sample[0]) - center_idx) // hop_size))
        for i in range(1, len(sample)):
            labels_by_frame.append(int((int(sample[i]) - center_idx) // hop_size))

        labels = np.zeros(length)
        for i in range(0, len(labels_by_frame), 2):
            start = labels_by_frame[i]
            end = labels_by_frame[i + 1]
            labels[start: end] = 1

        return labels

    
    def divide_into_batches(self, x: torch.Tensor, batch_size: int = 512) -> list:
        batches = []
        num_batches = x.shape[0] // batch_size
        if x.shape[0] % batch_size:
            num_batches += 1
        for i in range(num_batches):
            batch = x[i * batch_size: (i + 1) * batch_size]
            batches.append(batch)

        return batches




