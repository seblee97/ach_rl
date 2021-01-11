from collections import deque
from typing import List
from typing import Optional
from typing import Tuple

import gym
import numpy as np

import constants
from environments import base_environment
from utils import pre_processing_functions


class PreProcessor:
    """Class to take pre-process states for atari envrionment.

    Supports
        - Down-sampling
        - Gray-scaling
        - Taking max over set of frames.
    """

    def __init__(self, pre_processing: List[List]) -> None:
        """Class constructor.

        Sets up functions to be called in pre-processing.

        Args:
            pre_processing: list of pre-processing steps to take.
        """

        self._pre_process_functions = []

        for pre_processing_step in pre_processing:

            pre_processing_type = list(pre_processing_step.keys())[0]
            pre_processing_info = list(pre_processing_step.values())[0]

            if pre_processing_type == constants.Constants.DOWN_SAMPLE:
                width = pre_processing_info.get(constants.Constants.WIDTH)
                height = pre_processing_info.get(constants.Constants.HEIGHT)
                transformation = pre_processing_functions.DownSample(
                    width=width, height=height
                )
            elif pre_processing_type == constants.Constants.GRAY_SCALE:
                transformation = pre_processing_functions.GrayScale()
            elif pre_processing_type == constants.Constants.MAX_OVER:
                number_frames = pre_processing_info.get(constants.Constants.NUM_FRAMES)
                transformation = pre_processing_functions.MaxOver(
                    number_frames=number_frames
                )
            else:
                raise ValueError(
                    f"Pre-processing step {pre_processing_type} not recognised."
                )
            self._pre_process_functions.append(transformation)

    def __call__(self, unprocessed_state: np.ndarray) -> np.ndarray:
        """Call to each successive pre-processing function.

        Args:
            unprocessed_state: state before pre-processing.

        Returns:
            x: pre-processed state.
        """
        x = unprocessed_state
        for pre_process_function in self._pre_process_functions:
            x = pre_process_function(x)
        return x


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
        pre_processing: List[List],
        frame_skip: int,
        frame_stack: int,
        episode_timeout: Optional[int] = None,
    ) -> None:
        """Class constructor.

        Args:
            atari_env_name: gym atari environment name e.g. 'Pong-v0'.
            pre_processing: list of pre-processing steps to apply to state.
            frame_skip: number of frames to repeat each action.
            frame_stack: number of frames (independent of skip) to stack
                in state construction.
            episode_timeout: number of steps before automatic termination of episode.
        """
        # frame skip set to 1 here, dealt with later.
        try:
            self._env = gym.make(atari_env_name, frameskip=0)
        except TypeError:
            self._env = gym.make(atari_env_name)
        self._action_space = list(range(self._env.action_space.n))

        self._frame_skip = frame_skip
        self._frame_stack = frame_stack
        self._episode_timeout = episode_timeout or np.inf

        self._preprocessor = PreProcessor(pre_processing=pre_processing)

        self._frame_set: deque
        self._reward_set: deque

        self._episode_history: List[np.ndarray]

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
        frames, rewards = self._frame_skip_wrap(action=action)
        processed_frames = self._preprocessor(frames)
        self._episode_history.append(processed_frames)
        self._frame_set.append(processed_frames)
        self._reward_set.append(np.sum(rewards))

        squeezed_stacked_frame = np.stack(self._frame_set)

        # add batch dimension
        stacked_frame = np.expand_dims(squeezed_stacked_frame, 0)

        cum_reward = np.sum(self._reward_set)

        self._episode_step_count += 1

        if self._episode_step_count == self._episode_timeout:
            self._active = False

        return cum_reward, stacked_frame

    def _frame_skip_wrap(self, action: int) -> Tuple[List, List]:
        """Method to collect skipped states.

        Args:
            action: index of action to take through frames.

        Returns:
            states: list of states for each frame.
            rewards: list of rewards associated with each state.
        """
        states = []
        rewards = []
        for _ in range(self._frame_skip):
            state, reward, done, _ = self._env.step(action)
            states.append(state)
            rewards.append(reward)
            self._active = not done
        return states, rewards

    def reset_environment(self, train: bool) -> np.ndarray:
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        _ = self._env.reset()

        self._active = True
        self._episode_step_count = 0
        self._rewards_received = []
        self._training = train
        self._episode_history = []

        self._frame_set = deque(maxlen=self._frame_stack)
        self._reward_set = deque(maxlen=self._frame_stack)

        # frame stack and frame-skip with no-op
        for _ in range(self._frame_stack):
            frames, rewards = self._frame_skip_wrap(action=0)
            processed_frames = self._preprocessor(frames)
            self._episode_history.append(processed_frames)
            self._frame_set.append(processed_frames)
            self._reward_set.append(np.sum(rewards))

        squeezed_stacked_frame = np.stack(self._frame_set)

        # add batch dimension
        stacked_frame = np.expand_dims(squeezed_stacked_frame, 0)

        return stacked_frame
