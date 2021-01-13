from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

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
        max_over_actions: bool,
    ):
        self._state_action_values: List[Dict[Tuple[int], float]]

        self._gamma = gamma
        self._multiplicative_factor = multiplicative_factor
        self._max_over_actions = max_over_actions

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values):
        self._state_action_values = state_action_values

    def __call__(self, state: Tuple, action: int, next_state: Tuple) -> float:
        current_state_values = [s[state] for s in self._state_action_values]
        next_state_values = [s[next_state] for s in self._state_action_values]

        if self._max_over_actions:
            # this option means uncertainty is only function of state, not action
            current_state_action_values = [np.max(s) for s in current_state_values]
            next_state_action_values = [np.max(s) for s in next_state_values]
        else:
            current_state_action_values = [s[action] for s in current_state_values]
            next_state_action_values = [s[action] for s in next_state_values]

        current_state_uncertainty = np.std(current_state_action_values)
        next_state_uncertainty = np.std(next_state_action_values)

        penalty = self._multiplicative_factor * (
            self._gamma * next_state_uncertainty - current_state_uncertainty
        )

        return penalty
