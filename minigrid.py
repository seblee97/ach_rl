import random
from typing import Optional, Tuple, List

import numpy as np


class MiniGrid:
    """Grid world environment. Single reward"""

    def __init__(
        self,
        size: Tuple[int, int],
        starting_xy: Optional[List[int]] = None,
        reward_xy: Optional[Tuple[int, int]] = None,
    ):
        self._size = size
        self._starting_xy = starting_xy or list(self.random_coordinate())
        self._reward_xy = reward_xy or self.random_coordinate()

        self._agent_position: List[int]
        self._reset_environment()

        self._action_space = [0, 1, 2, 3]
        self._state_space = list(np.ndindex(size))

    @property
    def starting_xy(self) -> List[int]:
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

    def random_coordinate(self) -> Tuple[int, int]:
        """Generate random set of coordinates within grid dimensions.

        Returns:
            random_coordinates: (x, y) where x, y are random valid coordinates in grid.
        """
        x = random.randint(0, self._size[0] - 1)
        y = random.randint(0, self._size[1] - 1)
        return (x, y)

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

        if self._check_for_reward():
            return 1.0, tuple(self._agent_position)
        else:
            return 0.0, tuple(self._agent_position)

    def _check_for_reward(self) -> bool:
        """Check for reward, i.e. whether agent position is equal to reward position.
        If reward is found, reset environment for new episode.
        """
        if self._agent_position == list(self._reward_xy):
            self._reset_environment()
            return True
        else:
            return False

    def _reset_environment(self) -> None:
        """Reset environment. Bring agent back to starting position."""
        self._agent_position = self._starting_xy
