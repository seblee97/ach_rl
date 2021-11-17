import abc
from typing import Any
from typing import Dict


class BaseLearningRateScaler(abc.ABC):
    """Base class for adaptive learning rate scaling."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, episode: int, lr_scaling_info: Dict[str, Any]):
        pass
