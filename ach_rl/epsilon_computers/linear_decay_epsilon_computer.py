from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from ach_rl import constants
from ach_rl.epsilon_computers import base_epsilon_computer


class LinearDecayEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Deterministically decaying epsilon (linear)."""

    def __init__(
        self,
        initial_value: Union[int, float],
        final_value: Union[int, float],
        anneal_duration: int,
        decay_timeframe: str = "step",
    ):
        self._initial_value = initial_value
        self._anneal_duration = anneal_duration
        self._step_size = (initial_value - final_value) / anneal_duration
        self._decay_timeframe = decay_timeframe

    def __call__(
        self,
        epsilon_info: Dict[str, Any],
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        if self._decay_timeframe == constants.STEP:
            assert (
                step is not None
            ), "Linear epsilon decay based on step, must provide step in call."
            increment = step
        elif self._decay_timeframe == constants.EPISODE:
            assert (
                episode is not None
            ), "Linear epsilon decay based on episode, must provide episode in call."
            increment = episode
        if increment < self._anneal_duration:
            return self._initial_value - increment * self._step_size
        else:
            return self._final_value
