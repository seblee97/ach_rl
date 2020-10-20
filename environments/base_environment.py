import abc
from typing import Tuple


class BaseEnvironment(abc.ABC):
    """Base class for RL environments.

    Abstract methods:
        step: takes action produces reward and next state.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent."""
        pass
