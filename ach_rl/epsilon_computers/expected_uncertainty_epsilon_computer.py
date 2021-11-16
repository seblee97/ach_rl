from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from ach_rl import constants
from ach_rl.epsilon_computers import base_epsilon_computer


class ExpectedUncertaintyEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Epsilon tuned to 'expected' uncertainty over an ensemble."""

    def __init__(self, action_function: str, minimum_value: Union[int, float]):
        self._minimum_value = minimum_value
        self._action_function = action_function

    def _compute_epsilon(self, episode: int, epsilon_info: Dict[str, Any]):
        if self._action_function == constants.Constants.MAX:
            computed_epsilon = (
                1 - 1 / epsilon_info[constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY]
            )
        elif self._action_function == constants.Constants.MEAN:
            computed_epsilon = (
                1 - 1 / epsilon_info[constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY]
            )

        return np.amax([computed_epsilon, self._minimum_value])
