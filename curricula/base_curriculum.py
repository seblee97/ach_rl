import abc
from typing import List

import numpy as np


class BaseCurriculum(abc.ABC):
    def __init__(self, transition_episodes: List[int]):
        self._transition_episodes = iter(transition_episodes)
        self._next_transition_episode = next(self._transition_episodes)

    @property
    def next_transition_episode(self) -> int:
        return self._next_transition_episode

    @abc.abstractmethod
    def _transition(self):
        """Transition environment to next phase."""
        pass

    def __next__(self):
        self._transition()
        try:
            self._next_transition_episode = next(self._transition_episodes)
        except StopIteration:
            self._next_transition_episode = np.inf
