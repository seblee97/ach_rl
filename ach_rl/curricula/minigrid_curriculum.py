from typing import List
from typing import Optional
from typing import Tuple

from ach_rl import constants
from ach_rl.curricula import base_curriculum
from ach_rl.environments import minigrid


class MinigridCurriculum(minigrid.MiniGrid, base_curriculum.BaseCurriculum):
    """Curriculum wrapper for the MiniGrid environment."""

    def __init__(
        self,
        transition_episodes: List[int],
        environment_changes: List[List],
        size: Tuple[int, int],
        num_rewards: int,
        reward_magnitudes: List[float],
        starting_xy: Optional[Tuple[int, int]] = None,
        reward_xy: Optional[List[Tuple[int, int]]] = None,
        repeat_rewards: Optional[bool] = False,
        episode_timeout: Optional[int] = None,
    ) -> None:
        """Class constructor.

        Args:
            transition_episodes: indices of episodes at which transitions occur.
            environment_changes: list of changes to be made to the environment
                at each switch.
            size: size of environment.
            num_rewards: number of distinct rewards in the environment.
            reward_magnitudes: size of reward for each reward.
            starting_xy: the coordinate position at which agent starts
                each episode (random if not provided).
            reward_xy: list of coordinates at which rewards can be found.
            repeat_rewards: list of booleans indicating whether a reward
                is only collectible once.
            episode_timeout: number of episodes beyond which environment resets.
        """
        minigrid.MiniGrid.__init__(
            self,
            size=size,
            num_rewards=num_rewards,
            reward_magnitudes=reward_magnitudes,
            starting_xy=starting_xy,
            reward_xy=reward_xy,
            repeat_rewards=repeat_rewards,
            episode_timeout=episode_timeout,
        )
        base_curriculum.BaseCurriculum.__init__(
            self, transition_episodes=transition_episodes
        )

        self._environment_changes = iter(environment_changes)
        self._next_change = next(self._environment_changes)

    def _transition(self):
        """Transition environment to next phase."""
        if self._next_change[0] == constants.CHANGE_STARTING_POSITION:
            new_starting_xy = self._next_change[1]
            self._change_starting_position(new_starting_xy=new_starting_xy)
        else:
            raise ValueError(
                f"Environment change key {self._next_change[0]} not recognised."
            )
        try:
            self._next_change = next(self._environment_changes)
        except StopIteration:
            self._next_change = None

    def _change_starting_position(self, new_starting_xy: List[int]):
        """Change the start position of the agent in each episode."""
        self._starting_xy = tuple(new_starting_xy)
