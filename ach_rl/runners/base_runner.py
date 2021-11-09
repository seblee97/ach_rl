import abc
import os
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from ach_rl import constants
from ach_rl.experiments import ach_config
from ach_rl.runners import setup_runner


class BaseRunner(setup_runner.SetupRunner):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig, unique_id: str) -> None:

        super().__init__(config=config, unique_id=unique_id)

        self._apply_curriculum = config.apply_curriculum
        self._print_frequency = config.print_frequency or np.inf
        self._checkpoint_frequency = config.checkpoint_frequency
        self._model_checkpoint_frequency = config.model_checkpoint_frequency
        self._test_frequency = config.test_frequency
        self._test_types = config.testing
        self._num_episodes = config.num_episodes

        self._array_logging = self._setup_logging_frequencies(config.arrays)
        self._array_folder_path = os.path.join(self._checkpoint_path, constants.ARRAYS)
        os.makedirs(name=self._array_folder_path, exist_ok=True)
        for tag in self._array_logging.keys():
            os.makedirs(name=os.path.join(self._array_folder_path, tag), exist_ok=True)
        self._scalar_logging = self._setup_logging_frequencies(config.scalars)
        self._visualisations = self._setup_logging_frequencies(config.visualisations)
        self._visualisations_folder_path = os.path.join(
            self._checkpoint_path, constants.VISUALISATIONS
        )
        os.makedirs(name=self._visualisations_folder_path, exist_ok=True)
        self._post_visualisations = config.post_visualisations

        config.save_configuration(folder_path=config.checkpoint_path)

    def _setup_logging_frequencies(
        self, logging_list: List[Tuple[Union[List, str], int]]
    ) -> Dict[str, int]:
        """Parse logging list from config into mapping from
        tag to log to frequency with which it should be logged.

        Args:
            logging_list: un-parsed list of lists consisting of attributes
                to log and how frequently they should be logged.

        Returns:
            logging_frequencies: mapping from tags to log frequencies.
        """
        logging_frequencies = {}
        if logging_list is not None:
            for i in logging_list:
                if isinstance(i[0], list):
                    logging_frequencies[i[0][0]] = i[1]
                elif isinstance(i[0], str):
                    logging_frequencies[i[0]] = i[1]
                else:
                    raise ValueError("Log list incorrectly formatted.")
        return logging_frequencies

    @abc.abstractmethod
    def _setup_learner(self, config: ach_config.AchConfig):
        """Instantiate learner specified in configuration."""
        pass

    @abc.abstractmethod
    def _pre_episode_log(self, episode: int):
        """Define logging functionality for prior to each training episode."""
        pass

    def _array_log_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not array tag should be logged at this episode."""
        if tag in self._array_logging:
            if episode % self._array_logging[tag] == 0:
                return True
        return False

    def _scalar_log_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not scalar tag should be logged at this episode."""
        if tag in self._scalar_logging:
            if episode % self._scalar_logging[tag] == 0:
                return True
        return False

    def _visualisation_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not visualistation should be produced at this episode."""
        if tag in self._visualisations:
            if episode % self._visualisations[tag] == 0:
                return True
        return False

    def _write_scalar(
        self,
        tag: str,
        episode: int,
        scalar: Union[float, int],
        df_tag: Optional[str] = None,
    ):
        """If specified, log scalar."""
        if self._scalar_log_iteration(tag=tag, episode=episode):
            df_tag = df_tag or tag
            self._data_logger.write_scalar(tag=df_tag, step=episode, scalar=scalar)

    def _write_array(
        self,
        tag: str,
        episode: int,
        array: np.ndarray,
    ) -> None:
        if self._array_log_iteration(tag=tag, episode=episode):
            file_path = os.path.join(self._array_folder_path, tag, f"{tag}_{episode}")
            np.save(file_path, array)

    def train(self) -> None:
        """Perform training (and validation) on given number of episodes."""
        train_reward: float = 0
        train_step_count: float = np.inf
        episode_duration: float = 0

        self._logger.info("Starting Training...")

        for i in range(self._num_episodes):

            episode_start_time = time.time()

            if i % self._print_frequency == 0:
                if i != 0:
                    self._logger.info(f"    Latest Episode Duration {episode_duration}")
                    self._logger.info(f"    Latest Train Reward: {train_reward}")
                    self._logger.info(f"    Latest Train Length: {train_step_count}")
                self._logger.info(f"Episode {i + 1}/{self._num_episodes}: ")

            if i % self._checkpoint_frequency == 0 and i != 0:
                self._data_logger.checkpoint()
            if self._model_checkpoint_frequency is not None:
                if i % self._model_checkpoint_frequency == 0 and i != 0:
                    self._learner.checkpoint(
                        checkpoint_path=os.path.join(
                            self._checkpoint_path,
                            f"{constants.MODEL_CHECKPOINT}_{i}",
                        )
                    )

            self._pre_episode_log(i)
            self._test_episode(episode=i)

            if self._apply_curriculum:
                if i == self._environment.next_transition_episode:
                    next(self._environment)

            train_reward, train_step_count = self._train_episode(episode=i)

            self._write_scalar(
                tag=constants.TRAIN_EPISODE_LENGTH,
                episode=i,
                scalar=train_step_count,
            )
            self._write_scalar(
                tag=constants.TRAIN_EPISODE_REWARD,
                episode=i,
                scalar=train_reward,
            )

            episode_duration = time.time() - episode_start_time

        if constants.VISITATION_COUNT_HEATMAP in self._visualisations:
            self._data_logger.plot_array_data(
                name=constants.VISITATION_COUNT_HEATMAP,
                data=self._environment.visitation_counts,
            )

        self._data_logger.checkpoint()

    @abc.abstractmethod
    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        pass

    @abc.abstractmethod
    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner."""
        pass

    def _test_episode(self, episode: int) -> None:
        """Perform test rollouts, once with target policy,
        and---if specified---once with non-repeat target.

        Args:
            episode: current episode number.
        """
        if episode % self._test_frequency == 0:
            self._learner.eval()

            with torch.no_grad():
                if self._test_types is not None:
                    if constants.GREEDY in self._test_types:
                        self._greedy_test_episode(episode=episode)
                    if constants.NO_REP in self._test_types:
                        self._non_repeat_test_episode(episode=episode)

                self._run_specific_tests(episode=episode)

            self._learner.train()

    def _greedy_test_episode(
        self,
        episode: int,
        action_selection_method: Optional[Callable] = None,
        action_selection_method_args: Optional[Dict] = {},
        tag_: Optional[str] = "",
        output: bool = False,
    ) -> None:
        """Perform 'test' rollout with target policy

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        action_selection_method = (
            action_selection_method or self._learner.select_target_action
        )

        episode_reward = 0

        state = self._environment.reset_environment(train=False)

        while self._environment.active:
            action = action_selection_method(
                state=state, **action_selection_method_args
            )
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._data_logger.write_scalar(
            tag=constants.TEST_EPISODE_LENGTH + tag_,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._data_logger.write_scalar(
            tag=constants.TEST_EPISODE_REWARD + tag_,
            step=episode,
            scalar=episode_reward,
        )

        if episode != 0:
            if self._visualisation_iteration(
                constants.INDIVIDUAL_TEST_RUN + tag_, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.INDIVIDUAL_TEST_RUN + tag_}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

        if output:
            return episode_reward, self._environment.episode_step_count

    def _non_repeat_test_episode(
        self,
        episode: int,
        action_selection_method: Optional[Callable] = None,
        action_selection_method_args: Optional[Dict] = {},
        tag_: Optional[str] = "",
    ) -> None:
        """Perform 'test' rollout with target policy, except actions are never
        repeated in same state. Instead next best action is chosen.
        This is to break loops in test rollouts.

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        action_selection_method = (
            action_selection_method or self._learner.select_target_action
        )

        episode_reward = 0

        states_visited = {}

        state = self._environment.reset_environment(train=False)
        states_visited[state] = []

        while self._environment.active:
            action = action_selection_method(
                state=state,
                excluded_actions=states_visited.get(state, []),
                **action_selection_method_args,
            )
            if state not in states_visited:
                states_visited[state] = []
            states_visited[state].append(action)
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._data_logger.write_scalar_df(
            tag=constants.NO_REPEAT_TEST_EPISODE_LENGTH,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._data_logger.write_scalar_df(
            tag=constants.NO_REPEAT_TEST_EPISODE_REWARD,
            step=episode,
            scalar=episode_reward,
        )

        if episode != 0:
            if self._visualisation_iteration(
                constants.INDIVIDUAL_NO_REP_TEST_RUN, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.INDIVIDUAL_NO_REP_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def post_process(self) -> None:
        """Solidify any data and make plots."""
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
        self._post_visualisation()

    @abc.abstractmethod
    def _post_visualisation(self):
        """Create visualisations at end of training."""
        pass
