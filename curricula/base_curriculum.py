import abc
from typing import List

import numpy as np


class BaseCurriculum(abc.ABC):
    """Base class for curricula,
    which determine different stages of an environment."""

    def __init__(self, transition_episodes: List[int]) -> None:
        """Class constructor.

        Args:
            transition_episodes: indices of episodes at which transitions occur.
        """
        self._transition_episodes = iter(transition_episodes)
        self._next_transition_episode: int
        self._get_next_transition_episode()

    @property
    def next_transition_episode(self) -> int:
        return self._next_transition_episode

    @abc.abstractmethod
    def _transition(self) -> None:
        """Transition environment to next phase."""
        pass

    def _get_next_transition_episode(self) -> None:
        """From list of transition episodes, obtain the current 'next' index
        at which a switch in environment occurs.
        """
        try:
            self._next_transition_episode = next(self._transition_episodes)
        except StopIteration:
            self._next_transition_episode = np.inf

    def __next__(self) -> None:
        """Initiate change in environment."""
        self._transition()
        self._get_next_transition_episode()
