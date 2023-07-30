from typing import Dict
import pytorch_lightning as pl

import torch
import torch.nn as nn
from assets.config.config import MusicDetectionConfig

__all__ = [
    'MusicDetectionNetwork'
]

class MusicDetectionNetwork(pl.LightningModule):
    def __init__(self, extractor_config: MusicDetectionConfig, n_bins: int, n_frames: int, network_path: str):
        super().__init__()
        self.extractor_config = extractor_config
        self.network_path = network_path
        self.bce_loss = nn.BCELoss()
        self.layer1 = nn.Sequential(
            # 1 * 60 * 126
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_bins = n_bins // 2
        n_frames = n_frames // 2
        self.layer2 = nn.Sequential(
            # 32 * 30 * 63
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_bins = n_bins // 2
        n_frames = n_frames // 2
        self.layer3 = nn.Sequential(
            # 64 * 15 * 31
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_bins = n_bins // 2
        n_frames = n_frames // 2
        self.layer4 = nn.Sequential(
            # 128 * 7 * 15
            nn.Dropout(.4),
            nn.Linear(in_features=128 * n_bins * n_frames, out_features=2048),
            nn.ReLU(),
            nn.Dropout(.4),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )
        self.weights_init()

    def forward(self, out):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        # dummy output for speech prediction
        return out, None

    def training_step(self, train_batch, batch_idx):
        loss, music_num_correct = self.shared_step(train_batch)

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, music_num_correct = self.shared_step(val_batch)

        return {'val_loss': loss, 'music_num_correct': music_num_correct}

    def shared_step(self, batch: Dict[str, torch.Tensor]):
        data = batch.get('spectrogram')
        music_label = batch.get('music_label')

        music_pred, _ = self.forward(data)
        music_preds = torch.round(music_pred).view(-1)
        music_num_correct = float((music_preds == music_label.view(-1)).sum()) / music_label.shape[0]

        loss = self.bce_loss(music_pred, music_label)

        return loss, music_num_correct

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()

        music_num_correct_list = [x['music_num_correct'] for x in outputs]
        avg_music_accuracies = sum(music_num_correct_list) / len(music_num_correct_list)

        torch.jit.save(self.to_torchscript(), self.network_path)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters()
                                ,lr=self.extractor_config.learning_rate
                                ,weight_decay=self.extractor_config.weight_decay)

    def detect_music(self, x):
        return self(x)[0]

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
