from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from visitation_penalties import base_visitation_penalty


class TabularVisitationPenalty(base_visitation_penalty.BaseVisitationPenalty):
    def __init__(
        self, penalty_computer: base_visitation_penalty.BaseVisitationPenaltyComputer
    ):
        super().__init__(penalty_computer)

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def _compute_state_values(self, state):
        return np.array([s[state] for s in self._state_action_values])
