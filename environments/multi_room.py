import copy
import itertools
import random
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import constants
import collections

from environments import base_environment


class MultiRoom(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms. Multiple rewards."""

    ACTION_SPACE = [0, 1, 2, 3]

    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN

    DELTAS = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(
        self,
        ascii_map_path: str,
        episode_timeout: Optional[int] = None,
    ):

        (
            self._map,
            self._starting_xy,
            self._key_positions,
            self._door_positions,
            reward_positions,
        ) = self._parse_map(ascii_map_path)

        state_indices = np.where(self._map == 0)
        wall_indices = np.where(self._map == 1)
        positional_state_space = list(zip(state_indices[1], state_indices[0]))
        key_possession_state_space = list(
            itertools.product([0, 1], repeat=len(self._key_positions))
        )
        self._state_space = [
            i[0] + i[1]
            for i in itertools.product(
                positional_state_space, key_possession_state_space
            )
        ]

        self._walls = list(zip(wall_indices[1], wall_indices[0]))

        self._rewards = {
            tuple(reward_position): 1.0 for reward_position in reward_positions
        }
        self._total_rewards = sum(self._rewards.values())

        self._episode_timeout = episode_timeout or np.inf
        self._visitation_counts = -1 * copy.deepcopy(self._map)

        self._episode_history: List[List[int]]
        self._agent_position: np.ndarray
        self._rewards_received: List[float]
        self._keys_state: np.ndarray
        self._key_collection_times: Dict[int, int]

    def _parse_map(
        self, map_file_path: str
    ) -> Tuple[np.ndarray, List, List, List, List]:

        start_positions = []
        key_positions = []
        door_positions = []
        reward_positions = []
        map_rows = []

        MAPPING = {
            constants.Constants.WALL_CHARACTER: 1,
            constants.Constants.START_CHARACTER: 0,
            constants.Constants.DOOR_CHARACTER: 0,
            constants.Constants.OPEN_CHARACTER: 0,
            constants.Constants.KEY_CHARACTER: 0,
            constants.Constants.REWARD_CHARACTER: 0,
        }

        with open(map_file_path) as f:
            map_lines = f.read().splitlines()

            # flip indices for x, y referencing
            for i, line in enumerate(map_lines[::-1]):
                map_row = [MAPPING[char] for char in line]
                map_rows.append(map_row)
                if constants.Constants.REWARD_CHARACTER in line:
                    reward_positions.append(
                        (line.index(constants.Constants.REWARD_CHARACTER), i)
                    )
                if constants.Constants.KEY_CHARACTER in line:
                    key_positions.append(
                        (line.index(constants.Constants.KEY_CHARACTER), i)
                    )
                if constants.Constants.START_CHARACTER in line:
                    start_positions.append(
                        (line.index(constants.Constants.START_CHARACTER), i)
                    )
                if constants.Constants.DOOR_CHARACTER in line:
                    door_positions.append(
                        (line.index(constants.Constants.DOOR_CHARACTER), i)
                    )

        assert all(
            len(i) == len(map_rows[0]) for i in map_rows
        ), "ASCII map must specify rectangular grid."

        assert (
            len(start_positions) == 1
        ), "maximally one start position 'S' should be specified in ASCII map."

        multi_room_grid = np.array(map_rows, dtype=float)

        return (
            multi_room_grid,
            start_positions[0],
            key_positions,
            door_positions,
            reward_positions,
        )

    @property
    def starting_xy(self) -> Tuple[int, int]:
        return self._starting_xy

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._state_space

    @property
    def walls(self) -> List[Tuple]:
        return self._walls

    @property
    def episode_step_count(self) -> int:
        return self._episode_step_count

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._visitation_counts

    @property
    def episode_history(self) -> np.ndarray:
        return np.array(self._episode_history)

    def plot_episode_history(self) -> np.ndarray:
        # flip size so indexing is consistent with axis dimensions
        heatmap = np.ones(self._map.shape + (3,))

        # make walls black
        heatmap[self._map == 1] = np.zeros(3)

        # show reward in red
        for reward in self._rewards.keys():
            heatmap[reward[::-1]] = [1.0, 0.0, 0.0]

        # show door in maroon
        for door in self._door_positions:
            heatmap[tuple(door[::-1])] = [0.5, 0.0, 0.0]

        # show key in yellow
        for key_index, key_position in enumerate(self._key_positions):
            heatmap[tuple(key_position[::-1])] = [1.0, 1.0, 0.0]

        state_rgb_images = []

        # show agent in silver
        for step, state in enumerate(self._episode_history):
            state_image = copy.deepcopy(heatmap)
            for key_index, key_position in enumerate(self._key_positions):
                if self._keys_state[key_index]:
                    if step > self._key_collection_times[key_index]:
                        heatmap[tuple(key_position[::-1])] = np.ones(3)
            state_image[tuple(state[::-1])] = 0.5 * np.ones(3)
            state_rgb_images.append(state_image)

        return state_rgb_images

    def show_grid(self) -> np.ndarray:
        """Generate 2d array of current state of environment."""
        grid_state = copy.deepcopy(self._map)
        for reward in self._rewards.keys():
            grid_state[reward] = np.inf
        grid_state[tuple(self._agent_position)] = -1
        return grid_state

    def _move_agent(self, delta: np.ndarray) -> None:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._walls
        locked_door = tuple(provisional_new_position) in self._door_positions

        if locked_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._keys_state[door_index]:
                locked_door = False

        if not moving_into_wall and not locked_door:
            self._agent_position = provisional_new_position

        if tuple(self._agent_position) in self._key_positions:
            key_index = self._key_positions.index(tuple(self._agent_position))
            if not self._keys_state[key_index]:
                self._keys_state[key_index] = 1
                self._key_collection_times[key_index] = self._episode_step_count

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        self._move_agent(delta=self.DELTAS[action])

        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1

        reward = self._compute_reward()
        self._active = self._remain_active(reward=reward)

        self._episode_history.append(copy.deepcopy(tuple(self._agent_position)))

        new_state = tuple(self._agent_position) + tuple(self._keys_state)

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
        ):
            reward = self._rewards.get(tuple(self._agent_position))
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0
        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count == self._episode_timeout,
            reward == self._total_rewards,
        ]
        return not any(conditions)

    def reset_environment(self, train: bool) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        self._active = True
        self._episode_step_count = 0
        self._training = train
        self._agent_position = np.array(self._starting_xy)
        self._episode_history = [copy.deepcopy(tuple(self._agent_position))]
        self._rewards_received = []
        self._keys_state = np.zeros(len(self._key_positions), dtype=int)
        self._key_collection_times = {}

        initial_state = tuple(self._agent_position) + tuple(self._keys_state)

        return initial_state
