from collections import namedtuple
from typing import Tuple
from typing import Union

import numpy as np
from utils import custom_objects


class ReplayBuffer:
    """Object for experience replay,
    used to store experience tuples for off-policy learning."""

    def __init__(
        self,
        replay_size: int,
        state_dim: Tuple,
        mask_length: Union[int, None] = None,
        penalties: bool = False,
    ) -> None:
        """Class constructor.

        Args:
            replay_size: size of buffer.
            state_dim: dimension of state.
        """
        self._replay_size = replay_size
        self._state_dim = state_dim
        self._mask_length = mask_length
        self._penalties = penalties
        self._initialise_buffers()

        self._insertion_index = 0

    def _initialise_buffers(self):
        self._states_buffer = np.zeros(
            (self._replay_size,) + self._state_dim, dtype=np.float32
        )
        self._actions_buffer = np.zeros(self._replay_size, dtype=np.int8)
        self._rewards_buffer = np.zeros(self._replay_size, dtype=np.float32)
        self._next_states_buffer = np.zeros(
            (self._replay_size,) + self._state_dim, dtype=np.float32
        )
        self._active_buffer = np.zeros(self._replay_size, dtype=bool)
        if self._mask_length is not None:
            self._mask_buffer = np.zeros(
                (self._replay_size, self._mask_length), dtype=np.int8
            )
        else:
            self._mask_buffer = None
        if self._penalties:
            self._penalty_buffer = np.zeros(self._replay_size, dtype=np.float32)

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
        mask: Union[np.ndarray, None] = None,
        penalty: Union[np.ndarray, None] = None,
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
        print(insertion_index)

        self._states_buffer[insertion_index] = state
        self._actions_buffer[insertion_index] = action
        self._rewards_buffer[insertion_index] = reward
        self._next_states_buffer[insertion_index] = next_state
        self._active_buffer[insertion_index] = active
        if mask is not None:
            self._mask_buffer[insertion_index] = mask
        if penalty is not None:
            self._penalty_buffer[insertion_index] = penalty
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

        if self._mask_buffer is not None:
            mask_sample = self._mask_buffer[sample_indices]

            if self._penalties:
                penalty_sample = self._penalty_buffer[sample_indices]
                return custom_objects.MaskedPenaltyTransition(
                    state_encoding=states_sample,
                    action=actions_sample,
                    reward=rewards_sample,
                    next_state_encoding=next_states_sample,
                    active=active_sample,
                    mask=mask_sample,
                    penalty=penalty_sample,
                )
            else:
                return custom_objects.MaskedTransition(
                    state_encoding=states_sample,
                    action=actions_sample,
                    reward=rewards_sample,
                    next_state_encoding=next_states_sample,
                    active=active_sample,
                    mask=mask_sample,
                )

        else:
            if self._penalties:
                penalty_sample = self._penalty_buffer[sample_indices]
                return custom_objects.PenaltyTransition(
                    state_encoding=states_sample,
                    action=actions_sample,
                    reward=rewards_sample,
                    next_state_encoding=next_states_sample,
                    active=active_sample,
                    penalty=penalty_sample,
                )
            else:
                return custom_objects.Transition(
                    state_encoding=states_sample,
                    action=actions_sample,
                    reward=rewards_sample,
                    next_state_encoding=next_states_sample,
                    active=active_sample,
                )
