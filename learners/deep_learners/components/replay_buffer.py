import numpy as np
import random
import constants
from collections import deque
from typing import Tuple
from collections import namedtuple
from utils import custom_objects


class ReplayBuffer:
    def __init__(self, replay_size: int, state_dim: Tuple):

        self._states_buffer = np.zeros((replay_size,) + state_dim, dtype=np.int32)
        self._actions_buffer = np.zeros(replay_size, dtype=np.int32)
        self._rewards_buffer = np.zeros(replay_size)
        self._next_states_buffer = np.zeros((replay_size,) + state_dim, dtype=np.int32)
        self._active_buffer = np.zeros(replay_size, dtype=bool)

        self._insertion_index = 0
        self._replay_size = replay_size

    @property
    def states(self) -> np.ndarray:
        return self._states_buffer

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        active: bool,
    ):

        insertion_index = self._insertion_index % self._replay_size

        self._states_buffer[insertion_index] = state
        self._actions_buffer[insertion_index] = action
        self._rewards_buffer[insertion_index] = reward
        self._next_states_buffer[insertion_index] = next_state
        self._active_buffer[insertion_index] = active

        self._insertion_index += 1

    def sample(self, batch_size: int) -> namedtuple:

        current_num_entries = min(self._insertion_index, self._replay_size)
        sample_indices = np.random.choice(current_num_entries, size=batch_size)

        states_sample = self._states_buffer[sample_indices]
        actions_sample = self._actions_buffer[sample_indices]
        rewards_sample = self._rewards_buffer[sample_indices]
        next_states_sample = self._next_states_buffer[sample_indices]
        active_sample = self._active_buffer[sample_indices]

        return custom_objects.Transition(
            state_encoding=states_sample,
            action=actions_sample,
            reward=rewards_sample,
            next_state_encoding=next_states_sample,
            active=active_sample,
        )
