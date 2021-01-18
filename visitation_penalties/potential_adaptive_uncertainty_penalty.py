from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import constants
from visitation_penalties import base_visistation_penalty


class PotentialAdaptiveUncertaintyPenalty(
    base_visistation_penalty.BaseVisitationPenalty
):
    """Visitation penalty tuned to uncertainty,
    in the potential-based reward shaping formulation of Ng, Harada, Russell (1999)."""

    def __init__(
        self,
        gamma: float,
        multiplicative_factor: Union[int, float],
        action_function: str,
    ):
        self._state_action_values: List[Dict[Tuple[int], float]]

        self._gamma = gamma
        self._multiplicative_factor = multiplicative_factor
        self._action_function = action_function

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values):
        self._state_action_values = state_action_values

    def __call__(self, state: Tuple, action: int, next_state: Tuple) -> float:
        current_state_values = [s[state] for s in self._state_action_values]
        next_state_values = [s[next_state] for s in self._state_action_values]

        if self._action_function == constants.Constants.MAX:
            # this option means uncertainty is only function of state, not action
            current_state_uncertainty = np.std(
                [np.max(s) for s in current_state_values]
            )
            next_state_uncertainty = np.std([np.max(s) for s in next_state_values])
        elif self._action_function == constants.Constants.MEAN:
            current_state_action_variances = np.std(current_state_values, axis=0)
            next_state_action_variances = np.std(next_state_values, axis=0)

            current_state_uncertainty = np.mean(current_state_action_variances)
            next_state_uncertainty = np.mean(next_state_action_variances)
        elif self._action_function == constants.Constants.SELECT:
            current_state_uncertainty = np.std(
                [s[action] for s in current_state_values]
            )
            next_state_uncertainty = np.std([s[action] for s in next_state_values])

        penalty = self._multiplicative_factor * (
            self._gamma * next_state_uncertainty - current_state_uncertainty
        )

        return penalty
