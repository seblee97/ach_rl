import collections
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class ReducingVarianceWindowPenalty(base_visitation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to number of steps of reducing uncertainty over an ensemble."""

    def __init__(
        self,
        expected_multiplicative_factor: Union[float, int],
        unexpected_multiplicative_factor: Union[float, int],
        action_function: str,
        moving_average_window: int,
    ):
        self._expected_multiplicative_factor = expected_multiplicative_factor
        self._unexpected_multiplicative_factor = unexpected_multiplicative_factor
        self._action_function = action_function
        self._moving_average_window = moving_average_window

        self._uncertainty_history = {}

    def __call__(self, episode: int, penalty_info: Dict[str, Any]):
        state = penalty_info[constants.STATE]
        if state not in self._uncertainty_history:
            self._uncertainty_history[state] = collections.deque(
                maxlen=self._moving_average_window
            )
            # fill with single element to avoid NaN.
            self._uncertainty_history[state].append(0)

        previous_moving_average = np.mean(self._uncertainty_history[state])

        if self._action_function == constants.MAX:
            uncertainty = penalty_info[constants.CURRENT_STATE_MAX_UNCERTAINTY]
        elif self._action_function == constants.MEAN:
            uncertainty = penalty_info[constants.CURRENT_STATE_MEAN_UNCERTAINTY]
        elif self._action_function == constants.SELECT:
            uncertainty = penalty_info[constants.CURRENT_STATE_SELECT_UNCERTAINTY]

        self._uncertainty_history[state].append(uncertainty)

        new_moving_average = np.mean(self._uncertainty_history[state])
        moving_average_difference = new_moving_average - previous_moving_average
        unexpected_uncertainty = np.max([0, moving_average_difference])

        penalty = (
            self._unexpected_multiplicative_factor * unexpected_uncertainty
            + self._expected_multiplicative_factor * uncertainty
        )

        return penalty
