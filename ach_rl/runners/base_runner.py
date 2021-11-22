import abc
import os
import time
from typing import Any
from typing import Callable
from typing import Dict
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

        config.save_configuration(folder_path=config.checkpoint_path)

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

    def _print_info(
        self,
        episode: int,
        episode_duration: float,
        train_reward: float,
        train_step_count: int,
    ) -> None:
        """Print/log latest information.

        Args:
            episode: current episode count.
            episode_duration: time taken for latest episode to complete.
            train_reward: reward received by agent in last episode (train).
            train_step_count: number of steps taken in last episode (train).
        """
        if episode % self._print_frequency == 0:
            if episode != 0:
                self._logger.info(f"    Latest Episode Duration {episode_duration}")
                self._logger.info(f"    Latest Train Reward: {train_reward}")
                self._logger.info(f"    Latest Train Length: {train_step_count}")
            self._logger.info(f"Episode {episode + 1}/{self._num_episodes}: ")

    def _checkpoint(self, episode: int, final: bool = False) -> None:
        """Checkpoint data and models.

        Args:
            episode: current episode count.
            final: whether this is the last checkpoint. If true,
                checkpoint is performed regardless of stated checkpoint period.
        """
        condition_data = episode % self._checkpoint_frequency == 0 and episode != 0
        if condition_data or final:
            self._data_logger.checkpoint()
        if self._model_checkpoint_frequency is not None:
            condition_model = (
                episode % self._model_checkpoint_frequency == 0 and episode != 0
            )
            if condition_model or final:
                self._learner.checkpoint(
                    checkpoint_path=os.path.join(
                        self._checkpoint_path,
                        f"{constants.MODEL_CHECKPOINT}_{episode + 1}",
                    )
                )

    def _log_episode(self, episode: int, logging_dict: Dict[str, float]) -> None:
        """Write scalars for all quantities collected in logging dictionary.

        Args:
            episode: current episode.
            logging_dict: dictionary of items to be logged collected during training.
        """
        for tag, scalar in logging_dict.items():
            self._write_scalar(tag=tag, episode=episode, scalar=scalar)

    def train(self) -> None:
        """Perform training (and validation) on given number of episodes."""
        # train_reward: float = 0
        # train_step_count: float = np.inf
        # episode_duration: float = 0

        self._logger.info("Starting Training...")

        # self._pre_train_logging()

        for i in range(self._num_episodes):

            episode_start_time = time.time()

            self._checkpoint(episode=i)

            self._test_episode(episode=i)

            episode_logging_dict = self._train_episode(episode=i)

            self._log_episode(episode=i + 1, logging_dict=episode_logging_dict)

            episode_duration = time.time() - episode_start_time

            self._print_info(
                episode=i,
                episode_duration=episode_duration,
                train_reward=episode_logging_dict[constants.TRAIN_EPISODE_REWARD],
                train_step_count=episode_logging_dict[constants.TRAIN_EPISODE_LENGTH],
            )

            if i != 0 and self._visualisation_iteration(
                constants.INDIVIDUAL_TRAIN_RUN, i
            ):
                self._environment.visualise_episode_history(
                    save_path=os.path.join(
                        self._rollout_folder_path,
                        f"{constants.INDIVIDUAL_TRAIN_RUN}_{i}.mp4",
                    )
                )

        if constants.VISITATION_COUNT_HEATMAP in self._visualisations:
            self._data_logger.plot_array_data(
                name=constants.VISITATION_COUNT_HEATMAP,
                data=self._environment.visitation_counts,
            )

        self._checkpoint(episode=-1, final=True)

    @abc.abstractmethod
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            logging_dict: dictionary of items to log (e.g. episode reward).
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

        self._logger.info(
            "Performing greedy test episode with action selection method: "
            f"{action_selection_method}"
        )

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
