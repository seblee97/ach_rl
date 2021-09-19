from typing import List
from typing import Optional
from typing import Tuple

import constants
from curricula import base_curriculum
from environments import multi_room


class MultiroomCurriculum(multi_room.MultiRoom, base_curriculum.BaseCurriculum):
    """Curriculum wrapper for the Multiroom environment."""

    def __init__(
        self,
        transition_episodes: List[int],
        environment_changes: List[List],
        ascii_map_path: str,
        reward_specifications: List,
        representation: str,
        frame_stack: Optional[int] = 1,
        episode_timeout: Optional[int] = None,
        json_map_path: Optional[str] = None,
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
        multi_room.MultiRoom.__init__(
            self,
            ascii_map_path=ascii_map_path,
            reward_specifications=reward_specifications,
            representation=representation,
            frame_stack=frame_stack,
            episode_timeout=episode_timeout,
            json_map_path=json_map_path
        )
        base_curriculum.BaseCurriculum.__init__(
            self, transition_episodes=transition_episodes
        )   

        self._environment_changes = iter(environment_changes)
        self._next_change = next(self._environment_changes)

    def _transition(self):
        """Transition environment to next phase."""
        if self._next_change[0] == constants.Constants.CHANGE_STARTING_POSITION:
            new_starting_xy = self._next_change[1]
            self._change_starting_position(new_starting_xy=new_starting_xy)
        elif self._next_change[0] == constants.Constants.CHANGE_REWARD_POSITIONS:
            new_positions = [tuple(p) for p in self._next_change[1]]
            self._change_reward_positions(new_positions=new_positions)
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

    def _change_reward_positions(self, new_positions: List[List[int]]):
        # ensure new reward positions are accessible (i.e. not in wall or outside grid)
        for new_position in new_positions:
            assert new_position not in self._walls, "new position can not be in wall."
            assert new_position in self._positional_state_space, "new position must be in positional state space."

        self._rewards = self._get_reward_specification(
            reward_positions=new_positions,
            reward_specifications=self._reward_specifications,
        )
