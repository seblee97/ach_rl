import collections
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class ReducingEntropyWindowPenalty(
    base_visitation_penalty.BaseVisitationPenaltyComputer
):
    """Visitation penalty tuned to number of steps of reducing entropy over an ensemble."""

    def __init__(
        self,
        expected_multiplicative_factor: Union[float, int],
        unexpected_multiplicative_factor: Union[float, int],
        moving_average_window: int,
    ):
        self._expected_multiplicative_factor = expected_multiplicative_factor
        self._unexpected_multiplicative_factor = unexpected_multiplicative_factor
        self._moving_average_window = moving_average_window

        self._entropy_history = {}

    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        state = penalty_info[constants.STATE]
        if state not in self._entropy_history:
            self._entropy_history[state] = collections.deque(
                maxlen=self._moving_average_window
            )
            # fill with single element to avoid NaN.
            self._entropy_history[state].append(0)

        previous_moving_average = np.mean(self._entropy_history[state])

        entropy = penalty_info[constants.CURRENT_STATE_POLICY_ENTROPY]

        self._entropy_history[state].append(entropy)

        new_moving_average = np.mean(self._entropy_history[state])
        moving_average_difference = new_moving_average - previous_moving_average
        unexpected_uncertainty = np.max([0, moving_average_difference])

        penalty = (
            self._unexpected_multiplicative_factor * unexpected_uncertainty
            + self._expected_multiplicative_factor * entropy
        )

        return penalty
