import abc
from typing import Any
from typing import Dict


class BaseEpsilonComputer(abc.ABC):
    """Base class for adaptive epsilon computations."""

    @abc.abstractmethod
    def __call__(self, episode: int, epsilon_info: Dict[str, Any]):
        pass
