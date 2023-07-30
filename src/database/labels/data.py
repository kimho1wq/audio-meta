from abc import ABC, abstractmethod


__all__ = [
    'LabelGenerator',
    'DataInfo',
]


class DataInfo(ABC):
    DATAINFO = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.DATAINFO[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        raise NotImplementedError

    def __str__(self):
        return f'{self.alias()}'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.DATAINFO[alias]

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError


class LabelGenerator(ABC):
    LABELGENERATOR = {}
    ALIASES = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls.LABELGENERATOR[cls.alias()] = cls
        cls.ALIASES[cls.__name__] = cls.alias()

    def __str__(self):
        return f'{self.alias()}_generator'

    @classmethod
    @abstractmethod
    def alias(cls):
        pass

    @abstractmethod
    def get_metadata_type(self):
        pass

    @classmethod
    def call_from_alias(cls, alias: str):
        return cls.LABELGENERATOR[alias]

    @abstractmethod
    def generate(self):
        raise NotImplementedError