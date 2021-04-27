from typing import List
from typing import Optional
from typing import Tuple

import gym
import numpy as np
from environments import base_environment
from environments import env_wrappers


class MujocoEnv(base_environment.BaseEnvironment):

    """Wrapper for the gym mujoco environments"""

    def __init__(
        self,
        mujoco_env_name: str,
        episode_timeout: Optional[int] = None,
    ) -> None:
        """Class constructor.

        Args:
            mujoco_env_name: gym mujoco environment name e.g. 'Ant-v2'.
            frame_skip: number of frames to repeat each action.
            frame_stack: number of frames (independent of skip) to stack
                in state construction.
            episode_timeout: number of steps before automatic termination of episode.
        """
        self._mujoco_env_name = mujoco_env_name
        self._episode_timeout = episode_timeout or np.inf

        self._env = self._make_env()

        self._episode_history: List[np.ndarray]

    @property
    def action_space(self) -> List[int]:
        return np.array(
            [
                bound
                for bound in zip(
                    self._env.action_space.low, self._env.action_space.high
                )
            ]
        )

    def get_random_action(self):
        return self._env.action_space.sample()

    def _make_env(self):
        env = gym.make(self._mujoco_env_name)
        return env

    @property
    def state_dimension(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def episode_history(self) -> np.ndarray:
        return self._episode_history

    def plot_episode_history(self) -> np.ndarray:
        return self._episode_history

    def step(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """Take step in environment according to action of agent.

        Args:
            action: index of action to take.

        Returns:
            reward: float indicating reward.
            next_state: new state of agent.
        """
        state, reward, done, _ = self._env.step(action)

        self._episode_step_count += 1
        self._episode_history.append(state)

        self._active = not done
        if self._episode_step_count == self._episode_timeout:
            self._active = False

        return reward, state

    def reset_environment(self, train: bool) -> np.ndarray:
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        state = self._env.reset()

        self._active = True
        self._episode_step_count = 0
        self._rewards_received = []
        self._training = train
        self._episode_history = []

        return state
