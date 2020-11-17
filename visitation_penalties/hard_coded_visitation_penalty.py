import numpy as np

from experiments import ach_config
from visitation_penalties import base_visistation_penalty


class HardCodedPenalty(base_visistation_penalty.BaseVisitationPenalty):
    def __init__(self, config: ach_config.AchConfig):
        hard_coded_penalties = config.schedule

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

    def __call__(self, episode: int) -> float:
        if episode == self._next_switch_episode:
            self._get_next_penalty_set()
        return self._current_penalty
