from typing import List
from typing import Optional
from typing import Tuple

import gym
import numpy as np

from environments import base_environment
from environments import env_wrappers


class AtariEnv(base_environment.BaseEnvironment):

    ACTION_MEANING = {
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",
        4: "LEFT",
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT",
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE",
    }

    """Wrapper for the gym Atari environments"""

    def __init__(
        self,
        atari_env_name: str,
        frame_skip: int,
        frame_stack: int,
        episode_timeout: Optional[int] = None,
    ) -> None:
        """Class constructor.

        Args:
            atari_env_name: gym atari environment name e.g. 'Pong-v0'.
            frame_skip: number of frames to repeat each action.
            frame_stack: number of frames (independent of skip) to stack
                in state construction.
            episode_timeout: number of steps before automatic termination of episode.
        """
        self._atari_env_name = atari_env_name
        self._frame_skip = frame_skip
        self._frame_stack = frame_stack
        self._episode_timeout = episode_timeout or np.inf

        self._env = self._make_env()

        self._action_space = list(range(self._env.action_space.n))

        self._episode_history: List[np.ndarray]

    def _make_env(self):
        env = gym.make(self._atari_env_name)
        env = env_wrappers.MaxAndSkipEnv(env, skip=self._frame_skip)
        env = env_wrappers.FireResetEnv(env)
        env = env_wrappers.ProcessFrame84(env)
        env = env_wrappers.ImageToPyTorch(env)
        env = env_wrappers.BufferWrapper(env, n_steps=self._frame_stack)
        env = env_wrappers.ScaledFloatFrame(env)
        return env

    @property
    def action_space(self) -> List[int]:
        return self._action_space

    @property
    def state_dimension(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def episode_history(self) -> np.ndarray:
        return self._episode_history

    def plot_episode_history(self) -> np.ndarray:
        return self._episode_history

    def step(self, action: int) -> Tuple[float, np.ndarray]:
        """Take step in environment according to action of agent.

        Args:
            action: index of action to take.

        Returns:
            reward: float indicating reward.
            next_state: new state of agent.
        """
        squeezed_state, reward, done, _ = self._env.step(action)

        self._episode_step_count += 1
        self._episode_history.append(squeezed_state[-1])

        self._active = not done
        if self._episode_step_count == self._episode_timeout:
            self._active = False

        state = np.expand_dims(squeezed_state, 0)

        return reward, state

    def reset_environment(self, train: bool) -> np.ndarray:
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        squeezed_state = self._env.reset()

        self._active = True
        self._episode_step_count = 0
        self._rewards_received = []
        self._training = train
        self._episode_history = []

        state = np.expand_dims(squeezed_state, 0)

        return state
