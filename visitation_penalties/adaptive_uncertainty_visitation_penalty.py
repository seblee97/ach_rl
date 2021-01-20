from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import constants
from visitation_penalties import base_visistation_penalty


class AdaptiveUncertaintyPenalty(base_visistation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to uncertainty over an ensemble."""

    def __init__(self, multiplicative_factor: Union[float, int], action_function: str):
        self._state_action_values: List[Dict[Tuple[int], List[float]]]

        self._multiplicative_factor = multiplicative_factor
        self._action_function = action_function

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def __call__(
        self, episode: int, state: Tuple, action: int, next_state: Tuple
    ) -> float:
        current_state_values = [s[state] for s in self._state_action_values]

        if self._action_function == constants.Constants.MAX:
            # this option means uncertainty is only function of state, not action
            uncertainty = np.std([np.max(s) for s in current_state_values])
        elif self._action_function == constants.Constants.MEAN:
            uncertainty = np.mean(np.std(current_state_values, axis=0))
        elif self._action_function == constants.Constants.SELECT:
            uncertainty = np.std([s[action] for s in current_state_values])

        penalty = self._multiplicative_factor * uncertainty

        penalty_info = {constants.Constants.UNCERTAINTY: uncertainty}

        return penalty, penalty_info
