from __future__ import annotations

import os
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Type


__all__ = [
    'DictConvertable',
    'DataInfo',
    'custom_mkdir',
    'SummaryResult'
]

class DictConvertable(ABC):
    """
    Any data that can be represented as dictionary
    """

    @classmethod
    @abstractmethod
    def from_dict(cls, f_dict: dict):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class DataInfo(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, f_dict: dict):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


def custom_mkdir(path: Path):
    import os
    try:
        os.makedirs(path)
    except FileNotFoundError:
        raise ValueError(f'No such file or directory: {str(path)} \n')


class SummaryResult(ABC):
    RESULT = {}
    ALIASES = {}

    AUDIO_META_DICT = {
        'acousticness': 'AC',
        'danceability': 'DA',
        'energy': 'EN',
        'instrumentalness': 'IN',
        'liveness': 'LI',
        'valence': 'VA'
    }

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.RESULT[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    def __str__(self):
        return f'{self.alias()}_summary'

    def __call__(self):
        return self.summary()

    @classmethod
    def call_from_alias(cls, alias: str) -> Type[SummaryResult]:
        return cls.RESULT[alias]

    @abstractmethod
    def summary(self):
        raise NotImplementedError


class PlottingGraph(SummaryResult):
    def __init__(self, x_path: Path, y_path: Path, output_path: Path, meta_list: list):
        self.x_path = x_path
        self.y_path = y_path
        self.output_path = output_path

        # searching results..
        self.results_list = os.listdir(x_path)
        self.meta_list = meta_list

    @classmethod
    def alias(cls):
        return 'plotting'

    def summary(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        from math import pi
        import json

        for r in self.results_list:
            # loading data..
            with open(self.x_path.joinpath(r), 'r') as f:
                result = json.load(f)
            with open(self.y_path.joinpath(r), 'r') as f:
                label = json.load(f)

            for idx, res in enumerate([result, label]):
                key = 'features' if idx == 0 else 'feats'
                name = '_pm' if idx == 0 else '_label'

                # color depending on the datatype (proposed method or label)
                color = "steelblue" if idx == 0 else 'mediumseagreen'

                feats = {'song': ['']}
                feats.update({self.AUDIO_META_DICT[meta]: [res[key][meta]] for meta in self.meta_list})

                df = pd.DataFrame(feats)

                labels = df.columns[1:]
                num_labels = len(labels)
                angles = [x / float(num_labels) * (2 * pi) for x in range(num_labels)]
                angles += angles[:1]

                fig = plt.figure(figsize=(5, 5))

                # Window size, figsize-tuple x 100 pixel...
                fig.set_facecolor('white')  # background color
                for i, row in df.iterrows():
                    data = df.iloc[i].drop('song').tolist()
                    data += data[:1]

                    # generating the graphs
                    ax = plt.subplot(1, 1, i + 1,
                                     polar=True)  # args : nrows, ncols, index, **kwargs ax.set_theta_offset(pi / 2)
                    ax.set_theta_direction(-1)
                    plt.xticks(angles[:-1], labels, fontsize=13)
                    ax.tick_params(axis='x', which='major', pad=15)
                    ax.set_rlabel_position(0)
                    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=8)
                    plt.ylim(0, 1)

                    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
                    ax.fill(angles, data, color=color, alpha=0.4)
                    plt.title(row.song, size=20)
                    plt.tight_layout(pad=5)

                    plt.savefig(self.output_path.joinpath(result['trackInfo']['trackTitle'] + name + '.png'), dpi=360)
            plt.close('all')


class CalculatingAccuracy(SummaryResult):

    THRESHOLD = [0.05, 0.1, 0.15, 0.2]

    def __init__(self, x_path: Path, y_path: Path, output_path: Path, meta_list: list):
        self.x_path = x_path
        self.y_path = y_path
        self.output_path = output_path

        # searching results..
        self.results_list = os.listdir(x_path)
        self.meta_list = meta_list

    @classmethod
    def alias(cls):
        return 'accuracy'

    def summary(self):
        import json
        import numpy as np
        import tqdm
        from scipy.spatial.distance import cosine

        with open(self.output_path.joinpath('results.txt'), 'wt') as fout:
            fout.write(f'# of files : {len(self.results_list)}\n\n')
            for t in self.THRESHOLD:
                fout.write(f'Threshold : {t}\n')
                summary = {self.AUDIO_META_DICT[meta]: [] for meta in self.meta_list}
                c = []

                for r in tqdm.tqdm(self.results_list, desc=f'Threshold : {t}'):
                    o, l = np.zeros(len(self.meta_list)), np.zeros(len(self.meta_list))

                    # loading data..
                    with open(self.x_path.joinpath(r), 'r') as f:
                        result = json.load(f)
                    with open(self.y_path.joinpath(r), 'r') as f:
                        label = json.load(f)

                    for i, meta in enumerate(self.meta_list):
                        summary[self.AUDIO_META_DICT[meta]].append(abs(result['features'][meta] - label['feats'][meta]) < t)
                        o[i], l[i] = result['features'][meta], label['feats'][meta]
                    c.append(cosine(o, l) < t)

                for key in summary.keys():
                    fout.write(f'{key} : {np.mean(summary[key])}\n')
                fout.write(f'cosine dis : {np.mean(c)}\n\n')

class Uncertainty(SummaryResult):

    TOP_K = [100, 500, 1000, 5000, 10000]

    def __init__(self, x_path: Path, y_path: Path, output_path: Path, q_depth: int, result_type: str):
        self.x_path = x_path
        self.y_path = y_path
        self.output_path = output_path
        self.result_type = result_type
        self.auto_label = [1 / q_depth * i for i in range(q_depth)]

        # searching results..
        self.results_list = os.listdir(x_path)

    @classmethod
    def alias(cls):
        return 'entropy'

    def summary(self):
        import json
        import tqdm
        import bisect

        with open(self.output_path.joinpath('e_results.txt'), 'wt') as fout:
            fout.write(f'# of files : {len(self.results_list)}\n\n')

            entropy = []

            for r in tqdm.tqdm(self.results_list, desc=f'Calculating entropy :'):
                # loading data..
                with open(self.x_path.joinpath(r), 'r') as f:
                    result = json.load(f)
                entropy.append([r, self._entropy(np.asarray(result[self.result_type]))])

            entropy = sorted(entropy, key=lambda x: x[1])

            for k in self.TOP_K:
                fout.write(f'Top {k}\n')

                c = []
                c_1 = []
                hist_s = np.zeros(len(self.auto_label))
                hist_pm = np.zeros(len(self.auto_label))
                e = []
                for r in tqdm.tqdm(entropy[:k], desc=f'Finding Top-{k} uncertainty data :'):
                    # loading data..
                    with open(self.x_path.joinpath(r[0]), 'r') as f:
                        result = json.load(f)
                    with open(self.y_path.joinpath(r[0]), 'r') as f:
                        label = json.load(f)

                    e.append(r[1])

                    s = int(bisect.bisect_right(self.auto_label, label['feats'][self.result_type])) - 1
                    p = np.argmax(np.asarray(result[self.result_type]))
                    c.append(s == p)
                    c_1.append(abs(s - p) < 2)

                    hist_s[s] += 1
                    hist_pm[p] += 1

                fout.write(f'mean of uncertainty: {np.mean(e)}\n')
                fout.write(f'min uncertainty in top-k: {e[0]}\n')
                fout.write(f'max uncertainty in top-k: {e[k - 1]}\n')
                fout.write(f'histogram of the spotify data within the top-k: {hist_s}\n')
                fout.write(f'histogram of the output within the top-k: {hist_pm}\n')
                fout.write(f'accuracy (margin 0) : {np.mean(c)}\n')
                fout.write(f'accuracy (margin 1) : {np.mean(c_1)}\n\n')

    def _entropy(self, x: np.ndarray):
        eps = 1.e-20
        return -np.sum(x * np.log2(x + eps))
