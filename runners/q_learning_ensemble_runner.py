import copy
import multiprocessing
import os
from typing import Tuple
from typing import Union

import numpy as np

import constants
from environments import base_environment
from experiments import ach_config
from learners import base_learner
from learners.ensemble_learners import ensemble_learner
from learners.ensemble_learners.majority_vote_ensemble_learner import (
    MajorityVoteEnsemble,
)
from learners.ensemble_learners.mean_greedy_ensemble_learner import MeanGreedyEnsemble
from learners.ensemble_learners.sample_greedy_ensemble_learner import (
    SampleGreedyEnsemble,
)
from learners.tabular_learners import q_learner
from visitation_penalties.adaptive_uncertainty_visitation_penalty import (
    AdaptiveUncertaintyPenalty,
)
from visitation_penalties.adaptive_arriving_uncertainty_visitation_penalty import (
    AdaptiveArrivingUncertaintyPenalty,
)
from visitation_penalties.potential_adaptive_uncertainty_penalty import (
    PotentialAdaptiveUncertaintyPenalty,
)
from visitation_penalties.hard_coded_visitation_penalty import HardCodedPenalty
from runners import base_runner
from utils import cycle_counter


class EnsembleQLearningRunner(base_runner.BaseRunner):
    """Runner for Q-learning ensemble."""

    def __init__(self, config: ach_config.AchConfig):
        self._num_learners = config.num_learners
        self._targets = config.targets
        super().__init__(config=config)

        self._parallelise_ensemble = config.parallelise_ensemble
        if self._parallelise_ensemble:
            num_cores = min(len(self._learner.ensemble), multiprocessing.cpu_count())
            self._pool = multiprocessing.Pool(processes=num_cores)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learners = [
            self._get_individual_q_learner(config=config)
            for _ in range(self._num_learners)
        ]
        learner = ensemble_learner.EnsembleLearner(learner_ensemble=learners)
        return learner

    def _get_individual_q_learner(self, config: ach_config.AchConfig):
        """Setup a single q-learner."""
        return q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target="",
            initialisation_strategy=config.initialisation,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
        )

    def _pre_episode_log(self, episode: int):
        """Logging pre-episode. Includes value-function, individual run."""
        visualisation_configurations = [
            (constants.Constants.MAX_VALUES_PDF, True, False),
            (constants.Constants.QUIVER_VALUES_PDF, False, True),
            (constants.Constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(constants.Constants.VALUE_FUNCTION, episode):
            averaged_state_action_values = self._learner.state_action_values
            # tuple of save_path_tag, plot_max_values (bool), quiver (bool)
            for visualisation_configuration in visualisation_configurations:
                self._environment.plot_value_function(
                    values=averaged_state_action_values,
                    save_path=os.path.join(
                        self._checkpoint_path,
                        f"{episode}_{visualisation_configuration[0]}",
                    ),
                    plot_max_values=visualisation_configuration[1],
                    quiver=visualisation_configuration[2],
                )

        if self._visualisation_iteration(
            constants.Constants.INDIVIDUAL_VALUE_FUNCTIONS, episode
        ):
            all_state_action_values = (
                self._learner.individual_learner_state_action_values
            )
            for i, individual_state_action_values in enumerate(all_state_action_values):
                for visualisation_configuration in visualisation_configurations:
                    self._environment.plot_value_function(
                        values=individual_state_action_values,
                        save_path=os.path.join(
                            self._checkpoint_path,
                            f"{episode}_{i}_{visualisation_configuration[0]}",
                        ),
                        plot_max_values=visualisation_configuration[1],
                        quiver=visualisation_configuration[2],
                    )

        if self._visualisation_iteration(
            constants.Constants.VALUE_FUNCTION_STD, episode
        ):
            state_action_values_std = self._learner.state_action_values_std
            self._environment.plot_value_function(
                values=state_action_values_std,
                save_path=os.path.join(
                    self._checkpoint_path,
                    f"{episode}_{constants.Constants.VALUE_FUNCTION_STD_PDF}",
                ),
                plot_max_values=False,
                quiver=False,
            )

        if episode != 0:
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_TRAIN_RUN, episode
            ):
                self._logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TRAIN_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop (per learner in ensemble).

        Args:
            episode: index of episode

        Returns:
            episode_reward: mean scalar reward accumulated over ensemble episodes.
            num_steps: mean number of steps taken for ensemble episodes.
        """
        self._visitation_penalty.state_action_values = (
            self._learner.individual_learner_state_action_values
        )

        if self._parallelise_ensemble:
            train_fn = self._parallelised_train_episode
        else:
            train_fn = self._serial_train_episode

        (
            ensemble_rewards,
            ensemble_step_counts,
            ensemble_mean_penalties,
            ensemble_mean_penalty_infos,
        ) = train_fn(episode=episode)

        mean_reward = np.mean(ensemble_rewards)
        mean_step_count = np.mean(ensemble_step_counts)

        std_reward = np.std(ensemble_rewards)
        std_step_count = np.std(ensemble_step_counts)

        # close and join pool after last step
        if self._parallelise_ensemble:
            if episode == self._num_episodes - 1:
                self._pool.close()
                self._pool.join()

        # log data from individual runners in ensemble
        for i in range(len(self._learner.ensemble)):
            self._write_scalar(
                tag=f"{constants.Constants.TRAIN_EPISODE_REWARD}_{constants.Constants.ENSEMBLE_RUNNER}",
                episode=episode,
                scalar=ensemble_rewards[i],
                df_tag=f"{constants.Constants.TRAIN_EPISODE_REWARD}_{constants.Constants.ENSEMBLE_RUNNER}_{i}",
            )
            self._write_scalar(
                tag=f"{constants.Constants.TRAIN_EPISODE_LENGTH}_{constants.Constants.ENSEMBLE_RUNNER}",
                episode=episode,
                scalar=ensemble_step_counts[i],
                df_tag=f"{constants.Constants.TRAIN_EPISODE_LENGTH}_{constants.Constants.ENSEMBLE_RUNNER}_{i}",
            )

        # averages over ensemble
        self._write_scalar(
            tag=constants.Constants.ENSEMBLE_EPISODE_REWARD_STD,
            episode=episode,
            scalar=std_reward,
        )
        self._write_scalar(
            tag=constants.Constants.ENSEMBLE_EPISODE_LENGTH_STD,
            episode=episode,
            scalar=std_step_count,
        )
        self._write_scalar(
            tag=constants.Constants.MEAN_VISITATION_PENALTY,
            episode=episode,
            scalar=np.mean(ensemble_mean_penalties),
        )
        for penalty_info, ensemble_penalty_info in ensemble_mean_penalty_infos.items():
            self._write_scalar(
                tag=constants.Constants.MEAN_PENALTY_INFO,
                episode=episode,
                scalar=np.mean(ensemble_penalty_info),
                df_tag=penalty_info,
            )

        return mean_reward, mean_step_count

    def _serial_train_episode(
        self,
        episode: int,
    ) -> Tuple[float, int]:
        """Perform the train episode for each learner in ensemble serially.

        Args:
            episode: index of episode
        """
        ensemble_episode_rewards = []
        ensemble_episode_step_counts = []
        mean_penalties = []
        mean_penalty_infos = {}

        for learner in self._learner.ensemble:
            (
                episode_reward,
                episode_count,
                mean_penalty,
                mean_penalty_info,
            ) = self._single_train_episode(
                environment=self._environment,
                learner=learner,
                episode=episode,
            )

            ensemble_episode_rewards.append(episode_reward)
            ensemble_episode_step_counts.append(episode_count)
            mean_penalties.append(mean_penalty)
            for info_key, mean_info in mean_penalty_info.items():
                if info_key not in mean_penalty_infos:
                    mean_penalty_infos[info_key] = []
                mean_penalty_infos[info_key].append(mean_info)

        return (
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_infos,
        )

    def _parallelised_train_episode(
        self,
        episode: int,
    ) -> Tuple[float, int]:
        """Perform the train episode for each learner in ensemble in parallel.

        Args:
            episode: index of episode
        """
        processes_arguments = [
            (
                copy.deepcopy(self._environment),
                learner,
                episode,
            )
            for learner in self._learner.ensemble
        ]
        processes_results = self._pool.starmap(
            self._single_train_episode, processes_arguments
        )
        (
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_info,
        ) = zip(*processes_results)

        return (
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_info,
        )

    def _single_train_episode(
        self,
        environment: base_environment.BaseEnvironment,
        learner: base_learner.BaseLearner,
        episode: int,
    ) -> Union[None, Tuple[float, int]]:
        """Single learner train episode.

        Args:
            environment: environment in which to rollout episode.
            learner: learner to place in environment.
            episode: index of episode.

        Returns:
            episode_reward: single episode, single learner episode reward
            num_steps: single episode, single learner episode duration
        """
        episode_reward = 0

        penalties = []
        penalty_infos = {}

        state = environment.reset_environment(train=True)

        while environment.active:
            action = learner.select_behaviour_action(state)
            reward, next_state = environment.step(action)

            penalty, penalty_info = self._visitation_penalty(
                episode=episode, state=state, action=action, next_state=next_state
            )

            penalties.append(penalty)
            for info_key, info in penalty_info.items():
                if info_key not in penalty_infos.keys():
                    penalty_infos[info_key] = []
                penalty_infos[info_key].append(info)

            learner.step(
                state,
                action,
                reward,
                next_state,
                environment.active,
                penalty,
            )
            state = next_state
            episode_reward += reward

        mean_penalties = np.mean(penalties)
        mean_penalty_info = {k: np.mean(v) for k, v in penalty_infos.items()}

        return (
            episode_reward,
            environment.episode_step_count,
            mean_penalties,
            mean_penalty_info,
        )

    def _get_visitation_penalty(self, episode: int, state, action: int, next_state):
        if isinstance(self._visitation_penalty, AdaptiveUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(state=state, action=action)
        elif isinstance(self._visitation_penalty, HardCodedPenalty):
            penalty, penalty_info = self._visitation_penalty(episode=episode)
        elif isinstance(self._visitation_penalty, PotentialAdaptiveUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(
                state=state, action=action, next_state=next_state
            )
        elif isinstance(self._visitation_penalty, AdaptiveArrivingUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(next_state=next_state)
        return penalty, penalty_info

    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner.

        Here, there are various methods for performing inference.
        """
        greedy_sample = constants.Constants.GREEDY_SAMPLE
        greedy_mean = constants.Constants.GREEDY_MEAN
        greedy_vote = constants.Constants.GREEDY_VOTE

        no_rep_greedy_sample = "_".join(
            [constants.Constants.NO_REP, constants.Constants.GREEDY_SAMPLE]
        )
        no_rep_greedy_mean = "_".join(
            [constants.Constants.NO_REP, constants.Constants.GREEDY_MEAN]
        )
        no_rep_greedy_vote = "_".join(
            [constants.Constants.NO_REP, constants.Constants.GREEDY_VOTE]
        )

        if greedy_sample in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=SampleGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_sample}",
            )
        if greedy_mean in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=MeanGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_mean}",
            )
        if greedy_vote in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=MajorityVoteEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_vote}",
            )
        if no_rep_greedy_sample in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=SampleGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_sample}",
            )
        if no_rep_greedy_mean in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=MeanGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_mean}",
            )
        if no_rep_greedy_vote in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=MajorityVoteEnsemble.select_target_action,
                action_selection_method_args={
                    constants.Constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_vote}",
            )
