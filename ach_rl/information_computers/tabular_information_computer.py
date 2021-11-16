from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from ach_rl.information_computers import base_information_computer


class TabularInformationComputer(base_information_computer.BaseInformationComputer):
    def __init__(self):
        self._state_action_values: List[Dict[Tuple[int], List[float]]]
        super().__init__()

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def _compute_state_values(self, state):
        return np.array([s[state] for s in self._state_action_values])
