from abc import ABC, abstractmethod


__all__ = [
    'TrackInfo',
    'AudioFile',
    'DBCollector'
]


class TrackInfo(ABC):
    INFO = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.INFO[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_info'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.INFO[alias]

    def to_dict(self) -> dict:
        return self.to_dict()


class AudioFile(ABC):
    AUDIOFILE = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.AUDIOFILE[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_info'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.AUDIOFILE[alias]


class DBCollector(ABC):
    COLLECTOR = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.COLLECTOR[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_collector'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.COLLECTOR[alias]

    @abstractmethod
    def operate(self):
        raise NotImplementedError