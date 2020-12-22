import copy
import multiprocessing
from multiprocessing import managers
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

import constants
from environments import base_environment
from experiments import ach_config
from learners import base_learner
from learners.ensemble_learners import majority_vote_ensemble_learner
from learners.ensemble_learners import mean_greedy_ensemble_learner
from learners.ensemble_learners import sample_greedy_ensemble_learner
from learners.tabular_learners import q_learner
from runners import base_runner
from visitation_penalties import base_visistation_penalty

from utils import decorators

import time


class EnsembleQLearningRunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig):
        self._num_learners = config.num_learners
        super().__init__(config=config)

        self._multiplicative_factor = config.multiplicative_factor

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
        if config.target == constants.Constants.GREEDY_SAMPLE:
            learner = sample_greedy_ensemble_learner.SampleGreedyEnsemble(
                learner_ensemble=learners
            )
        elif config.target == constants.Constants.GREEDY_MEAN:
            learner = mean_greedy_ensemble_learner.MeanGreedyEnsemble(
                learner_ensemble=learners
            )
        elif config.target == constants.Constants.GREEDY_VOTE:
            learner = majority_vote_ensemble_learner.MajorityVoteEnsemble(
                learner_ensemble=learners
            )
        return learner

    def _get_individual_q_learner(self, config: ach_config.AchConfig):
        return q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target=config.target,
            initialisation_strategy=config.initialisation,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
        )

    def _pre_episode_log(self, episode: int):
        if episode != 0:
            if constants.Constants.INDIVIDUAL_TRAIN_RUN in self._plot_logging:
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
        self._visitation_penalty.state_action_values = [
            learner.state_action_values for learner in self._learner.ensemble
        ]

        if self._parallelise_ensemble:
            train_fn = self._parallelised_train_episode
        else:
            train_fn = self._serial_train_episode

        ensemble_rewards, ensemble_step_counts, ensemble_mean_penalties = train_fn(
            episode=episode
        )

        mean_reward = np.mean(ensemble_rewards)
        mean_step_count = np.mean(ensemble_step_counts)

        std_reward = np.std(ensemble_rewards)
        std_step_count = np.std(ensemble_step_counts)

        # close and join pool after last step
        if self._parallelise_ensemble:
            if episode == self._num_episodes - 1:
                self._pool.close()
                self._pool.join()

        if episode % self._train_log_frequency == 0:
            # log data from individual runners in ensemble
            for i in range(len(self._learner.ensemble)):
                self._logger.write_scalar_df(
                    tag=f"{constants.Constants.TRAIN_EPISODE_REWARD}_{constants.Constants.ENSEMBLE_RUNNER}_{i}",
                    step=episode,
                    scalar=ensemble_rewards[i],
                )
                self._logger.write_scalar_df(
                    tag=f"{constants.Constants.TRAIN_EPISODE_LENGTH}_{constants.Constants.ENSEMBLE_RUNNER}_{i}",
                    step=episode,
                    scalar=ensemble_step_counts[i],
                )
            self._logger.write_scalar_df(
                tag=constants.Constants.ENSEMBLE_EPISODE_REWARD_STD,
                step=episode,
                scalar=std_reward,
            )
            self._logger.write_scalar_df(
                tag=constants.Constants.ENSEMBLE_EPISODE_LENGTH_STD,
                step=episode,
                scalar=std_step_count,
            )
        self._logger.write_scalar_df(
            tag=constants.Constants.MEAN_VISITATION_PENALTY,
            step=episode,
            scalar=np.mean(ensemble_mean_penalties),
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

        for learner in self._learner.ensemble:
            episode_reward, episode_count, mean_penalty = self._single_train_episode(
                environment=self._environment,
                learner=learner,
                episode=episode,
            )

            ensemble_episode_rewards.append(episode_reward)
            ensemble_episode_step_counts.append(episode_count)
            mean_penalties.append(mean_penalty)

        return ensemble_episode_rewards, ensemble_episode_step_counts, mean_penalties

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
        ensemble_episode_rewards, ensemble_episode_step_counts, mean_penalties = zip(
            *processes_results
        )

        return ensemble_episode_rewards, ensemble_episode_step_counts, mean_penalties

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

        state = environment.reset_environment(train=True)

        while environment.active:
            action = learner.select_behaviour_action(state)
            reward, new_state = environment.step(action)
            penalty = self._multiplicative_factor * self._visitation_penalty(
                state=state, action=action
            )
            penalties.append(penalty)
            learner.step(
                state,
                action,
                reward,
                new_state,
                environment.active,
                penalty,
            )
            state = new_state
            episode_reward += reward

        return episode_reward, environment.episode_step_count, np.mean(penalties)
