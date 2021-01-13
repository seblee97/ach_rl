import abc
from experiments import ach_config


class BaseVisitationPenalty(abc.ABC):
    """Base class for visitation penalties.

    Generally calls to this 'penalty' should follow
    the potential-based reward structure of
    Ng, Harada, Russell (1999).
    """

    def __init__(self, config: ach_config.AchConfig):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """compute penalty at episode"""
        pass
