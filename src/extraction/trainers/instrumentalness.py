from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader

from assets.config.config import PreProcConfig, TransformConfig, ExtractorConfig
from src.extraction.data import Trainer
from src.extraction.utils import get_input_shape
from src.extraction.dataset import InstrumentalnessCustomDataset

__all__ = [
    'InstrumentalnessTrainer'
]

class InstrumentalnessTrainer(Trainer):
    SEQ_DIM = 0
    FEAT_DIM = 1

    X = 0
    Y = 1

    POSITIVE = 1
    NEGATIVE = 0

    MAX_NUM_WORKERS = 5

    INPUT_PATH = Path('assets/instrumentalness/spectrogram')
    LABEL_PATH = Path('assets/instrumentalness/label')

    def __init__(self, model, pre_proc_config: PreProcConfig, transform_config: TransformConfig, extractor_config: ExtractorConfig):
        self.pre_proc_config = pre_proc_config
        self.transform_config = transform_config

        self.cnn_config = extractor_config.cnn_config
        self.rnn_config = extractor_config.rnn_config
        self.task_config = extractor_config.task_config

        self.max_length_limit_ms = extractor_config.max_length_limit_ms

        self.input_shape = get_input_shape(pre_proc_config, transform_config, extractor_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = model

    @classmethod
    def alias(cls):
        return 'instrumentalness'

    def eval(self):
        self.network.eval()

    def train(self):
        lr = 0.0001
        batch_size = 128
        num_workers = self.MAX_NUM_WORKERS
        patience = 5
        epochs = 10000
        ratio = 0.8

        self.network.rnn_cell.flatten_parameters()

        if torch.cuda.is_available():
            model = nn.DataParallel(self.network).to(self.device)
        else:
            model = self.network.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = nn.BCELoss()

        # data load..
        train_loader, test_loader = self.load_data(self.INPUT_PATH, self.LABEL_PATH, ratio)

        train_loader = InstrumentalnessCustomDataset(train_loader, self.input_shape)
        test_loader = InstrumentalnessCustomDataset(test_loader, self.input_shape)

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
                x, y = data[self.X].to(self.device), data[self.Y].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                y_hat = model(x=x)

                # calc loss
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
                    x, y = data[self.X].to(self.device), data[self.Y].to(self.device)

                    # forward
                    y_hat = model(x=x)

                    # loss
                    loss = criterion(y_hat, y)

                    # print statistics
                    running_loss += loss.item()

                print(f'\n---------------------------------------------\n'
                      f'Val [{epoch + 1}] : [total loss - {running_loss / len(test_dataloader):.5f}, min_loss - {min_loss:.5f},'
                      f' patience - {count} / {patience}'
                      f'---------------------------------------------\n')
                his['val_loss'].append(running_loss / len(test_dataloader))
            scheduler.step()

            # Patience check..
            if running_loss < min_loss:
                min_loss = running_loss
                count = 0
                self.network = model
            else:
                count += 1

            if count == patience:
                break


    def load_data(self, x_path: Path, y_path: Path, ratio: float):
        import os
        import numpy as np

        np.random.seed(0)
        x_list = os.listdir(x_path)

        dataset = {1: [], 0: []}
        for x in x_list:
            with open(y_path.joinpath(Path(x).stem + '.txt'), 'r') as f:
                key = int(f.read())
                dataset[key].append(x)

        np.random.shuffle(dataset[self.POSITIVE])
        np.random.shuffle(dataset[self.NEGATIVE])

        train = dataset[self.POSITIVE][:int(len(dataset[self.POSITIVE]) * ratio)] + \
                dataset[self.NEGATIVE][:int(len(dataset[self.NEGATIVE]) * ratio)]
        validation = dataset[self.POSITIVE][int(len(dataset[self.POSITIVE]) * ratio):] + \
                     dataset[self.NEGATIVE][int(len(dataset[self.NEGATIVE]) * ratio):]

        train = [[x_path.joinpath(x), y_path.joinpath(Path(x).stem + '.txt')] for x in train]
        validation = [[x_path.joinpath(x), y_path.joinpath(Path(x).stem + '.txt')] for x in validation]

        return train, validation
