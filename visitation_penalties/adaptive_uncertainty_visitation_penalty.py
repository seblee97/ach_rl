from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from experiments import ach_config
from visitation_penalties import base_visistation_penalty


class AdaptiveUncertaintyPenalty(base_visistation_penalty.BaseVisitationPenalty):
    def __init__(self, config: ach_config.AchConfig):
        self._state_action_values: List[Dict[Tuple[int], float]]

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values):
        self._state_action_values = state_action_values

    def __call__(self, state: Tuple, action: int) -> float:
        current_state_action_values = [
            s[state][action] for s in self._state_action_values
        ]
        uncertainty = np.std(current_state_action_values)

        return uncertainty
