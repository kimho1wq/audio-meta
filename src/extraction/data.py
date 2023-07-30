from abc import ABC, abstractmethod
import torch.nn as nn
import torch

from pathlib import Path


__all__ = [
    'Extractor',
    'Trainer',
    'Network'
]


class Extractor(ABC):
    EXTRACTOR = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.EXTRACTOR[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_extractor'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.EXTRACTOR[alias]

    @abstractmethod
    def extract(self):
        raise NotImplementedError

class Trainer(ABC):
    TRAINER = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.TRAINER[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_trainer'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.TRAINER[alias]

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def save_network(self, net_path: Path):
        """
        # save the trained network
        """
        if torch.cuda.is_available():
            torch.jit.save(torch.jit.script(self.network.module), net_path)
        else:
            torch.jit.save(torch.jit.script(self.network), net_path)


class Network(nn.Module):
    NETWORK = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.NETWORK[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_network'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.NETWORK[alias]
