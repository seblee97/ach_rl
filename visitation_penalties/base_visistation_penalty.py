import abc
from experiments import ach_config


class BaseVisitationPenalty(abc.ABC):
    def __init__(self, config: ach_config.AchConfig):
        pass

    @abc.abstractmethod
    def __call__(self, episode: int):
        """compute penalty at episode"""
        pass