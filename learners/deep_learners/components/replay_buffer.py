import numpy as np
from typing import Tuple
import collections
from collections import namedtuple
from utils import custom_objects


class ReplayBuffer:
    """Object for experience replay,
    used to store experience tuples for off-policy learning."""

    def __init__(self, replay_size: int, state_dim: Tuple) -> None:
        """Class constructor.

        Args:
            replay_size: size of buffer.
            state_dim: dimension of state.
        """
        self._buffer = collections.deque(maxlen=replay_size)
        self._replay_size = replay_size

    def add(self, transition: custom_objects.Transition) -> None:
        """Add tuple to buffer.

        Args:
            state: pre-state.
            action: action.
            reward: reward.
            next_state: post-state.
            active: whether episode is over.
        """
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> namedtuple:
        """Sample experience from buffer.

        Args:
            batch_size: size of sample.

        Return:
            sample: experience sample.
        """
        sample_indices = np.random.choice(
            len(self._buffer), size=batch_size, replace=False
        )
        (
            states_sample,
            actions_sample,
            rewards_sample,
            next_states_sample,
            active_sample,
        ) = zip(*[self._buffer[idx] for idx in sample_indices])

        return (
            np.array(states_sample),
            np.array(actions_sample),
            np.array(rewards_sample, dtype=np.float32),
            np.array(next_states_sample),
            np.array(active_sample, dtype=np.uint8),
        )


class OldReplayBuffer:
    """Object for experience replay,
    used to store experience tuples for off-policy learning."""

    def __init__(self, replay_size: int, state_dim: Tuple) -> None:
        """Class constructor.

        Args:
            replay_size: size of buffer.
            state_dim: dimension of state.
        """
        self._states_buffer = np.zeros((replay_size,) + state_dim, dtype=np.float32)
        self._actions_buffer = np.zeros(replay_size, dtype=np.int8)
        self._rewards_buffer = np.zeros(replay_size, dtype=np.float32)
        self._next_states_buffer = np.zeros(
            (replay_size,) + state_dim, dtype=np.float32
        )
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
    ) -> None:
        """Add tuple to buffer.

        Args:
            state: pre-state.
            action: action.
            reward: reward.
            next_state: post-state.
            active: whether episode is over.
        """
        insertion_index = self._insertion_index % self._replay_size

        self._states_buffer[insertion_index] = state
        self._actions_buffer[insertion_index] = action
        self._rewards_buffer[insertion_index] = reward
        self._next_states_buffer[insertion_index] = next_state
        self._active_buffer[insertion_index] = active

        self._insertion_index += 1

    def sample(self, batch_size: int) -> namedtuple:
        """Sample experience from buffer.

        Args:
            batch_size: size of sample.

        Return:
            sample: experience sample.
        """
        current_num_entries = min(self._insertion_index, self._replay_size)
        sample_indices = np.random.choice(
            current_num_entries, size=batch_size, replace=False
        )

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
