from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from visitation_penalties import base_visistation_penalty


class AdaptiveUncertaintyPenalty(base_visistation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to uncertainty over an ensemble."""

    def __init__(
        self, multiplicative_factor: Union[float, int], max_over_actions: bool
    ):
        self._state_action_values: List[Dict[Tuple[int], List[float]]]

        self._multiplicative_factor = multiplicative_factor
        self._max_over_actions = max_over_actions

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def __call__(self, state: Tuple, action: int) -> float:
        current_state_values = [s[state] for s in self._state_action_values]

        if self._max_over_actions:
            # this option means uncertainty is only function of state, not action
            current_state_action_values = [np.max(s) for s in current_state_values]
        else:
            current_state_action_values = [s[action] for s in current_state_values]

        uncertainty = self._multiplicative_factor * np.std(current_state_action_values)

        return uncertainty
