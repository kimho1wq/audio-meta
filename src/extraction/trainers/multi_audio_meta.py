from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader

from src.audio import Spectrogram
from src.pipeline import SpecMetadata
from assets.config.config import PreProcConfig, TransformConfig, ExtractorConfig

from src.extraction.data import Trainer
from src.extraction.utils import get_input_shape
from src.extraction.dataset import AudioMetaCustomDataset

__all__ = [
    'MultiAudioMetaTrainer'
]

class MultiAudioMetaTrainer(Trainer):
    SEQ_DIM = 0
    FEAT_DIM = 1

    X = 0
    Y = 1

    FEATURE = 0
    FINETUNE = 1

    MAX_NUM_WORKERS = 10

    def __init__(self, model, pre_proc_config: PreProcConfig, transform_config: TransformConfig, extractor_config: ExtractorConfig):
        self.pre_proc_config = pre_proc_config
        self.transform_config = transform_config

        self.cnn_config = extractor_config.cnn_config
        self.rnn_config = extractor_config.rnn_config
        self.task_config = extractor_config.task_config

        self.audio_features = extractor_config.task_config.task_type.multi_task
        self.max_length_limit_ms = extractor_config.max_length_limit_ms

        self.input_shape = get_input_shape(pre_proc_config, transform_config, extractor_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = model

    @classmethod
    def alias(cls):
        return 'multi_audio_meta'

    def eval(self):
        self.network.eval()

    def train(self, input_path: Path, label_path: Path):
        lr = 0.0001
        batch_size = 128
        num_workers = self.MAX_NUM_WORKERS
        patience = 10
        epochs = 10000
        ratio = 0.8

        self.network.rnn_cell.flatten_parameters()

        if torch.cuda.is_available():
            model = nn.DataParallel(self.network).to(self.device)
        else:
            model = self.network.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = nn.MSELoss() if self.task_config.task_type.type[0] == 'r' else nn.CrossEntropyLoss()

        # data load..
        train_loader, test_loader = self.load_data(input_path, label_path, ratio)

        train_loader = AudioMetaCustomDataset(*[train_loader, self.audio_features, self.input_shape, self.task_config])
        test_loader = AudioMetaCustomDataset(*[test_loader, self.audio_features, self.input_shape, self.task_config])

        train_dataloader = DataLoader(train_loader,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      drop_last=True)
        test_dataloader = DataLoader(test_loader,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     drop_last=True)

        min_loss = 10000.
        count = 0
        PRINT_NUM = 10
        his = {'val_loss': [], 'train_loss': []}

        print(f'Training start!\n'
              f'# of train data : {len(train_dataloader)}\n'
              f'# of validation data : {len(test_dataloader)}\n')

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            # Training phase..
            for i, data in enumerate(train_dataloader):
                # get the inputs; data is a list of [inputs, labels]
                x, y = {x:data[self.X][x].to(self.device) for x in data[self.X].keys()}, data[self.Y].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                y_hat = model(x=x)

                # calc loss
                # classification
                if self.task_config.task_type.type[0] == 'c':
                    loss = 0
                    for feat in range(len(self.audio_features)):
                        loss = loss + criterion(y_hat[:, feat, :], torch.max(y[:, feat, :], 1)[1])
                # regression
                else:
                    loss = criterion(y_hat, y)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % PRINT_NUM == PRINT_NUM - 1:  # print every 20 mini-batches
                    print(f'Train [{epoch + 1}, {i + 1}] : [total loss - {running_loss / PRINT_NUM:.5f}]')
                    his['train_loss'].append(running_loss / PRINT_NUM)
                    running_loss = 0.0

            with torch.no_grad():
                # Validation phase..
                running_loss = 0.
                for i, data in enumerate(test_dataloader):
                    # get the inputs; data is a list of [inputs, labels]
                    x, y = {x: data[self.X][x].to(self.device) for x in data[self.X].keys()}, data[self.Y].to(self.device)

                    # forward
                    y_hat = model(x=x)

                    # loss
                    if self.task_config.task_type.type[0] == 'c':
                        loss = 0
                        for feat in range(len(self.audio_features)):
                            loss = loss + criterion(y_hat[:, feat, :], torch.max(y[:, feat, :], 1)[1])

                        if i == 0:
                            res = np.asarray(y_hat.to('cpu'))
                            label = np.asarray(y.to('cpu'))
                        else:
                            res = np.append(res, np.asarray(y_hat.to('cpu')), axis=0)
                            label = np.append(label, np.asarray(y.to('cpu')), axis=0)
                    else:
                        loss = criterion(y_hat, y)

                    # print statistics
                    running_loss += loss.item()

                print(f'\n---------------------------------------------\n'
                      f'Val [{epoch + 1}] : [total loss - {running_loss / len(test_dataloader):.5f}, min_loss - {min_loss:.5f},'
                      f' patience - {count} / {patience}'
                      f'---------------------------------------------\n')
                his['val_loss'].append(running_loss / len(test_dataloader))
                if self.task_config.task_type.type[0] =='c':
                    self._entropy(res)
                    print(f'accuracy : {self._accuracy(res, label)}\n')

            scheduler.step()

            # Patience check..
            if running_loss < min_loss or epoch < 50:
                min_loss = running_loss
                count = 0
            else:
                count += 1

            if count == patience:
                break

        self.network = model

    def load_data(self, x_path: Path, y_path: Path, ratio: float):
        import os

        meta_list = []
        count = 0
        for ti in self.transform_config.transform_info:
            if os.path.exists(x_path.joinpath(ti.transform_type).joinpath('new_metadata.json')):
                count += 1

        _exist = True if count == len(self.transform_config.transform_info) else False
        filename = 'new_metadata.json' if _exist else 'metadata.json'
        for ti in self.transform_config.transform_info:
            meta_list.append(SpecMetadata.load_from_file(x_path.joinpath(ti.transform_type), filename))

        # sorted by uid
        for i, a_m in enumerate(meta_list):
            meta_list[i].spec_infos = sorted(a_m.spec_infos, key=lambda x: x.file_info.meta.audio_file_meta.trackinfo.trackId)

        inputs, labels = self.create_input_arr(meta_list, y_path, _exist)

        if not _exist:
            for i, ti in enumerate(self.transform_config.transform_info):
                meta_list[i].save_to_file(x_path.joinpath(ti.transform_type), 'new_metadata.json')

        boundary = int(len(labels) * ratio)
        train, train_l = inputs[:boundary], labels[:boundary]
        validation, validation_l = inputs[boundary:], labels[boundary:]

        return [train, train_l], [validation, validation_l]

    def create_input_arr(self, meta_list: SpecMetadata, label_path: Path, _exist: bool):
        aud_meta_list = [a_m.spec_infos for a_m in meta_list]
        idx_list = np.arange(len(aud_meta_list[0]))

        inputs_list = []
        label_list = []
        flag_list = []
        default_format = 'labels/{}.json'

        for idx in idx_list:
            FLAG = True
            inputs = {}
            for i, tf_type in enumerate(self.cnn_config):
                if not _exist:
                    spec = Spectrogram.load_with_file_info(aud_meta_list[i][idx]).arr

                    # IF the length of input audio is longer than threshold, exclude that audio
                    if spec.shape[self.SEQ_DIM] > self.input_shape[self.SEQ_DIM]:
                        FLAG = False
                        break
                inputs[tf_type.type] = aud_meta_list[i][idx]

            flag_list.append(FLAG)
            if FLAG:
                label_list.append(label_path.joinpath(default_format.format(Path(aud_meta_list[0][idx].path).stem)))
                inputs_list.append(inputs)

        if not _exist:
            for idx in range(len(meta_list)):
                meta_list[idx].spec_infos = [m for i, m in enumerate(meta_list[idx].spec_infos[:len(flag_list)]) if flag_list[i]]
                meta_list[idx].db_sz = len(meta_list[idx].spec_infos)

        return inputs_list, label_list

    def _entropy(self, x: np.ndarray):
        def softmax(value):
            v = value - np.max(value, axis=1)[:, None]
            return np.exp(v) / np.sum(np.exp(v), axis=1)[:, None]

        top_k = 100
        eps = 1.e-20
        tmp_x = x + eps

        print(f'histogram of top-100 uncertainty (q_depth - {self.task_config.q_index}) :')
        for i, feat in enumerate(self.audio_features):
            inputs = tmp_x[:, i, :]
            inputs = softmax(inputs)

            e = -np.sum(inputs * np.log2(inputs), axis=1)

            m_u = np.argpartition(e, top_k)[:top_k]
            hist = np.zeros(self.task_config.q_index)
            for j in range(top_k):
                hist[np.argmax(inputs[m_u[j]])] += 1

            print(f'{feat} : {hist}')
        print(f'')

    def _accuracy(self, x: np.ndarray, y: np.ndarray):
        res = {feat: 0 for feat in self.audio_features}
        for i, feat in enumerate(self.audio_features):
            inputs = x[:, i, :]
            label = y[:, i, :]

            o = np.argmax(inputs, axis=1)
            l = np.argmax(label, axis=1)
            res[feat] = np.mean(o == l)
        return res

