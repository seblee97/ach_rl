from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from ach_rl.epsilon_computers import base_epsilon_computer


class ConstantEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Fixed, hard-coded epsilon."""

    def __init__(self, epsilon: Union[float, int]):
        self._epsilon = epsilon

    def __call__(
        self,
        epsilon_info: Dict[str, Any],
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        return self._epsilon
