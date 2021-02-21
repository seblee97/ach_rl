from typing import Any
from typing import Dict
from typing import List

import constants
import numpy as np
from visitation_penalties import base_visitation_penalty


class HardCodedPenalty(base_visitation_penalty.BaseVisitationPenaltyComputer):
    """Hard-coded penalties."""

    def __init__(self, hard_coded_penalties: List[List]):
        self._switch_episodes = iter([int(i[0]) for i in hard_coded_penalties])
        self._penalties = iter([float(i[1]) for i in hard_coded_penalties])

        # default 0
        self._current_penalty = 0.0
        self._next_switch_episode = next(self._switch_episodes)

    def _get_next_penalty_set(self):
        try:
            self._next_switch_episode = next(self._switch_episodes)
        except StopIteration:
            self._next_switch_episode = np.inf
        try:
            self._current_penalty = next(self._penalties)
        except StopIteration:
            pass

    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]) -> float:
        if episode == self._next_switch_episode:
            self._get_next_penalty_set()

        batch_dimension = penalty_info[
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY
        ].shape[0]

        return self._current_penalty * np.ones(batch_dimension)
