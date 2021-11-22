from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class HardCodedPenalty(base_visitation_penalty.BaseVisitationPenalty):
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

    def __call__(
        self,
        episode: int,
        penalty_info: Dict[str, Any],
        batch_dimension: Optional[int] = None,
    ) -> float:
        if episode == self._next_switch_episode:
            self._get_next_penalty_set()

        if batch_dimension is not None:
            return self._current_penalty * np.ones(batch_dimension)
        else:
            return self._current_penalty

        # candidate_reference_measures = [
        #     k for k in penalty_info.keys() if constants.UNCERTAINTY in k
        # ]
        # if candidate_reference_measures:
        #     reference_measure = penalty_info[candidate_reference_measures[0]]
        # else:
        #     reference_measure = None

        # if reference_measure is None or isinstance(reference_measure, float):
        #     return self._current_penalty
        # elif isinstance(reference_measure, np.ndarray):
        #     batch_dimension = reference_measure.shape[0]
        #     return self._current_penalty * np.ones(batch_dimension)
