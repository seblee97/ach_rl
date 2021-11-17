from typing import Any
from typing import Dict
from typing import Union

from ach_rl.epsilon_computers import base_epsilon_computer


class ConstantEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Fixed, hard-coded epsilon."""

    def __init__(self, epsilon: Union[float, int]):
        self._epsilon = epsilon

    def __call__(self, episode: int, epsilon_info: Dict[str, Any]):
        return self._epsilon
