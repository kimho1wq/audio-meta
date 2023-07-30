from __future__ import annotations

import numpy as np

from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Type


__all__ = [
    'AudioTransformer'
]

class AudioTransformer(ABC):
    TRANSFORMER = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.TRANSFORMER[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_transformer'

    def __call__(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        return self.transform(y, sr)

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str) -> Type[AudioTransformer]:
        return cls.TRANSFORMER[alias]

    @abstractmethod
    def transform(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        raise NotImplementedError
