import abc
from typing import Any
from typing import Dict
from typing import Optional


class BaseEpsilonComputer(abc.ABC):
    """Base class for adaptive epsilon computations."""

    @abc.abstractmethod
    def __call__(
        self,
        epsilon_info: Dict[str, Any],
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        pass
