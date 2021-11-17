from typing import Any
from typing import Dict
from typing import Union

from ach_rl.epsilon_computers import base_epsilon_computer


class LinearDecayEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Deterministically decaying epsilon (linear)."""

    def __init__(
        self,
        initial_value: Union[int, float],
        final_value: Union[int, float],
        anneal_duration: int,
    ):
        self._initial_value = initial_value
        self._anneal_duration = anneal_duration
        self._step_size = (initial_value - final_value) / anneal_duration

    def __call__(self, episode: int, epsilon_info: Dict[str, Any]):
        if episode < self._anneal_duration:
            return self._initial_value - episode * self._step_size
        else:
            return self._final_value
