import collections
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from ach_rl import constants
from ach_rl.epsilon_computers import base_epsilon_computer


class UnexpectedUncertaintyEpsilonComputer(base_epsilon_computer.BaseEpsilonComputer):
    """Epsilon tuned to 'unexpected' uncertainty over an ensemble."""

    def __init__(
        self,
        moving_average_window: int,
        action_function: str,
        minimum_value: Union[int, float],
    ):
        self._moving_average_window = moving_average_window
        self._action_function = action_function
        self._minimum_value = minimum_value

        self._policy_entropy_history = {}

    def _compute_epsilon(self, episode: int, epsilon_info: Dict[str, Any]):
        state = epsilon_info[constants.Constants.STATE]
        if state not in self._policy_entropy_history:
            self._policy_entropy_history[state] = collections.deque(
                maxlen=self._moving_average_window
            )
            # fill with single element to avoid NaN.
            self._policy_entropy_history[state].append(0)

        previous_moving_average = np.mean(self._policy_entropy_history[state])

        self._policy_entropy_history[state].append(
            epsilon_info[constants.Constants.NORMALISED_POLICY_ENTROPY]
        )

        new_moving_average = np.mean(self._policy_entropy_history[state])
        moving_average_difference = new_moving_average - previous_moving_average

        computed_epsilon = abs(moving_average_difference)

        return np.amax([computed_epsilon, self._minimum_value])
