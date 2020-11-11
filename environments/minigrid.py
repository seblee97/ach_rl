import copy
import random
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from environments import base_environment


class MiniGrid(base_environment.BaseEnvironment):
    """Grid world environment. Single reward"""

    def __init__(
        self,
        size: Tuple[int, int],
        num_rewards: int,
        reward_magnitudes: List[float],
        starting_xy: Optional[Tuple[int, int]] = None,
        reward_xy: Optional[List[Tuple[int, int]]] = None,
        repeat_rewards: Optional[bool] = False,
        episode_timeout: Optional[int] = None,
    ):
        self._size = size
        self._num_rewards = num_rewards
        self._starting_xy = starting_xy

        reward_xy = reward_xy or self._get_random_rewards()
        self._rewards = {
            reward_position: reward_magnitude
            for reward_position, reward_magnitude in zip(reward_xy, reward_magnitudes)
        }
        self._total_rewards = sum(self._rewards.values())
        self._repeat_rewards = repeat_rewards

        self._episode_timeout = episode_timeout or np.inf

        self._action_space = [0, 1, 2, 3]
        self._state_space = list(np.ndindex(size))
        self._visitation_counts = np.zeros(self._size)

        self._agent_position: List[int]
        self._training: bool
        self._active: bool
        self._episode_step_count: int
        self._rewards_received: List[float]

    @property
    def starting_xy(self) -> Tuple[int, int]:
        return self._starting_xy

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self._action_space

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._state_space

    @property
    def active(self) -> bool:
        return self._active

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
        heatmap = np.zeros(self._size)
        for reward in self._rewards.keys():
            heatmap[reward[0]][reward[1]] = np.inf
        for state in self._episode_history:
            heatmap[state[0]][state[1]] += 1
        return heatmap

    def show_grid(self) -> np.ndarray:
        """Generate 2d array of current state of environment."""
        grid_state = np.zeros(self._size)
        for reward in self._rewards.keys():
            grid_state[reward[0]][reward[1]] = np.inf
        grid_state[self._agent_position[0]][self._agent_position[1]] = -1
        return grid_state

    def random_coordinate(self) -> Tuple[int, int]:
        """Generate random set of coordinates within grid dimensions.

        Returns:
            random_coordinates: (x, y) where x, y are random valid coordinates in grid.
        """
        x = random.randint(0, self._size[0] - 1)
        y = random.randint(0, self._size[1] - 1)
        return (x, y)

    def _get_random_rewards(self) -> List[Tuple[int, int]]:
        """Generate list of random reward locations without repeats."""
        reward_locations = []
        for _ in range(self._num_rewards):
            while True:
                random_coordinate = self.random_coordinate()
                if random_coordinate not in reward_locations:
                    reward_locations.append(random_coordinate)
                    break
        return reward_locations

    def _move_agent_left(self) -> None:
        """Move agent one position to the left. If already at left-most point, no-op."""
        if self._agent_position[0] == 0:
            pass
        else:
            self._agent_position[0] -= 1

    def _move_agent_up(self) -> None:
        """Move agent one position upwards. If already at upper-most point, no-op."""
        if self._agent_position[1] == self._size[1] - 1:
            pass
        else:
            self._agent_position[1] += 1

    def _move_agent_right(self) -> None:
        """Move agent one position upwards. If already at right-most point, no-op."""
        if self._agent_position[0] == self._size[0] - 1:
            pass
        else:
            self._agent_position[0] += 1

    def _move_agent_down(self) -> None:
        """Move agent one position downwards. If already at bottom-most point, no-op."""
        if self._agent_position[1] == 0:
            pass
        else:
            self._agent_position[1] -= 1

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            action in self._action_space
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        if action == 0:
            self._move_agent_left()
        elif action == 1:
            self._move_agent_up()
        elif action == 2:
            self._move_agent_right()
        elif action == 3:
            self._move_agent_down()

        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[0]][
                self._agent_position[1]
            ] += 1

        reward = self._compute_reward()
        self._active = self._remain_active(reward=reward)

        self._episode_history.append(copy.deepcopy(self._agent_position))

        return reward, tuple(self._agent_position)

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
            reward == self._total_rewards and not self._repeat_rewards,
        ]
        return not any(conditions)

    def reset_environment(self, train: bool) -> None:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        self._active = True
        self._episode_step_count = 0
        self._training = train
        if self._starting_xy is not None:
            self._agent_position = list(self._starting_xy)
        else:
            self._agent_position = list(self.random_coordinate())
        self._episode_history = [copy.deepcopy(self._agent_position)]
        self._rewards_received = []
