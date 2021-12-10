import collections
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import torch
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
        self._tensor_id_mapping = {}

    def __call__(
        self,
        epsilon_info: Dict[str, Any],
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):

        state = epsilon_info[constants.STATE]
        state_hash = hash(state.cpu().numpy().tobytes())

        if state_hash not in self._policy_entropy_history:
            self._policy_entropy_history[state_hash] = collections.deque(
                maxlen=self._moving_average_window
            )
            # fill with single element to avoid NaN.
            self._policy_entropy_history[state_hash].append(0)

        previous_moving_average = np.mean(self._policy_entropy_history[state_hash])

        self._policy_entropy_history[state_hash].append(
            epsilon_info[
                f"{constants.CURRENT}_{constants.NORMALISED_POLICY_ENTROPY}"
            ].item()
        )

        new_moving_average = np.mean(self._policy_entropy_history[state_hash])
        moving_average_difference = new_moving_average - previous_moving_average

        computed_epsilon = abs(moving_average_difference)

        return np.amax([computed_epsilon, self._minimum_value])
