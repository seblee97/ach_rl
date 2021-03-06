import copy
import itertools
import multiprocessing
import os
import re
from typing import Tuple
from typing import Union
from typing import Dict

import constants
import numpy as np
from environments import base_environment
from experiments import ach_config
from learners import base_learner
from learners.ensemble_learners import tabular_ensemble_learner
from learners.ensemble_learners.majority_vote_ensemble_learner import \
    MajorityVoteEnsemble
from learners.ensemble_learners.mean_greedy_ensemble_learner import \
    MeanGreedyEnsemble
from learners.ensemble_learners.sample_greedy_ensemble_learner import \
    SampleGreedyEnsemble
from learners.tabular_learners import q_learner
from runners import base_runner
from utils import cycle_counter
from utils import experiment_utils
from visitation_penalties.adaptive_arriving_uncertainty_visitation_penalty import \
    AdaptiveArrivingUncertaintyPenalty
from visitation_penalties.adaptive_uncertainty_visitation_penalty import \
    AdaptiveUncertaintyPenalty
from visitation_penalties.hard_coded_visitation_penalty import HardCodedPenalty
from visitation_penalties.potential_adaptive_uncertainty_penalty import \
    PotentialAdaptiveUncertaintyPenalty


class EnsembleQLearningRunner(base_runner.BaseRunner):
    """Runner for Q-learning ensemble."""

    def __init__(self, config: ach_config.AchConfig):
        self._num_learners = config.num_learners
        self._targets = config.targets
        super().__init__(config=config)

        self._parallelise_ensemble = config.parallelise_ensemble
        if self._parallelise_ensemble:
            num_cores = multiprocessing.cpu_count()
            self._pool = multiprocessing.Pool(processes=num_cores)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        initialisation_strategy = self._get_initialisation_strategy(config)
        if config.copy_learner_initialisation:
            single_learner = self._get_individual_q_learner(
                config=config, initialisation_strategy=initialisation_strategy
            )
            learners = [
                copy.deepcopy(single_learner) for _ in range(self._num_learners)
            ]
        else:
            learners = [
                self._get_individual_q_learner(
                    config=config,
                    initialisation_strategy=initialisation_strategy
                )
                for _ in range(self._num_learners)
            ]
        learner = tabular_ensemble_learner.TabularEnsembleLearner(
            learner_ensemble=learners
        )
        return learner

    def _get_initialisation_strategy(self, config: ach_config.AchConfig):
        if config.initialisation == constants.Constants.RANDOM_UNIFORM:
            initialisation_strategy = {
                constants.Constants.RANDOM_UNIFORM: {
                    constants.Constants.LOWER_BOUND: config.lower_bound,
                    constants.Constants.UPPER_BOUND: config.upper_bound
                }
            }
        elif config.initialisation == constants.Constants.RANDOM_NORMAL:
            initialisation_strategy = {
                constants.Constants.RANDOM_NORMAL: {
                    constants.Constants.MEAN: config.mean,
                    constants.Constants.VARIANCE: config.variance
                }
            }
        else:
            initialisation_strategy == {config.initialisation}
        return initialisation_strategy

    def _get_individual_q_learner(self, config: ach_config.AchConfig, initialisation_strategy: Dict):
        """Setup a single q-learner."""
        return q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target="",
            initialisation_strategy=initialisation_strategy,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
        )

    def _pre_episode_log(self, episode: int):
        """Logging pre-episode. Includes value-function, individual run."""
        self._logger.info(f"Episode {episode}: pre-episode data logging...")
        self._pre_episode_visualisations(episode=episode)
        self._pre_episode_array_logging(episode=episode)

    def _pre_episode_visualisations(self, episode: int):
        visualisation_configurations = [
            (constants.Constants.MAX_VALUES_PDF, True, False),
            (constants.Constants.QUIVER_VALUES_PDF, False, True),
            (constants.Constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(constants.Constants.VALUE_FUNCTION, episode):
            averaged_state_action_values = self._learner.state_action_values
            # tuple of save_path_tag, plot_max_values (bool), quiver (bool)
            if self._parallelise_ensemble:

                self._logger.info("Parallel value function visualisation...")
                processes_arguments = [
                    (
                        averaged_state_action_values,
                        visualisation_configuration[1],
                        visualisation_configuration[2],
                        constants.Constants.MAX,
                        os.path.join(
                            self._visualisations_folder_path,
                            f"{episode}_{visualisation_configuration[0]}",
                        ),
                    )
                    for visualisation_configuration in visualisation_configurations
                ]
                self._pool.starmap(
                    self._environment.plot_value_function, processes_arguments
                )
            else:
                for visualisation_configuration in visualisation_configurations:
                    self._logger.info(
                        "Serial value function visualisation: "
                        f"{visualisation_configuration[0]}"
                    )
                    self._environment.plot_value_function(
                        values=averaged_state_action_values,
                        save_path=os.path.join(
                            self._visualisations_folder_path,
                            f"{episode}_{visualisation_configuration[0]}",
                        ),
                        plot_max_values=visualisation_configuration[1],
                        quiver=visualisation_configuration[2],
                        over_actions=constants.Constants.MAX,
                    )

        if self._visualisation_iteration(
            constants.Constants.INDIVIDUAL_VALUE_FUNCTIONS, episode
        ):
            all_state_action_values = (
                self._learner.individual_learner_state_action_values
            )
            learner_visual_configuration_combos = list(
                itertools.product(
                    np.arange(len(all_state_action_values)),
                    visualisation_configurations,
                )
            )
            if self._parallelise_ensemble:
                self._logger.info("Parallel individual value function visualisation...")
                processes_arguments = [
                    (
                        all_state_action_values[combo[0]],
                        combo[1][1],
                        combo[1][2],
                        constants.Constants.MAX,
                        os.path.join(
                            self._visualisations_folder_path,
                            f"{episode}_{combo[0]}_{combo[1][0]}",
                        ),
                    )
                    for combo in learner_visual_configuration_combos
                ]
                self._pool.starmap(
                    self._environment.plot_value_function, processes_arguments
                )
            else:
                for i, individual_state_action_values in enumerate(
                    all_state_action_values
                ):
                    for visualisation_configuration in visualisation_configurations:
                        self._logger.info(
                            "Serial individual value function visualisation: "
                            f"learner {i}, {visualisation_configuration[0]}"
                        )
                        self._environment.plot_value_function(
                            values=individual_state_action_values,
                            save_path=os.path.join(
                                self._visualisations_folder_path,
                                f"{episode}_{i}_{visualisation_configuration[0]}",
                            ),
                            plot_max_values=visualisation_configuration[1],
                            quiver=visualisation_configuration[2],
                            over_actions=constants.Constants.MAX,
                        )

        if self._visualisation_iteration(
            constants.Constants.VALUE_FUNCTION_STD, episode
        ):
            self._logger.info("Standard deviation value function visualisation...")
            state_action_values_std = self._learner.state_action_values_std
            self._environment.plot_value_function(
                values=state_action_values_std,
                save_path=os.path.join(
                    self._visualisations_folder_path,
                    f"{episode}_{constants.Constants.VALUE_FUNCTION_STD_PDF}",
                ),
                plot_max_values=True,
                quiver=False,
                over_actions=constants.Constants.MEAN,
            )

        if episode != 0:
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_TRAIN_RUN, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TRAIN_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def _pre_episode_array_logging(self, episode: int):
        self._write_array(
            tag=constants.Constants.VALUE_FUNCTION,
            episode=episode,
            array=self._learner.state_action_values,
        )
        self._write_array(
            tag=constants.Constants.VALUE_FUNCTION_STD,
            episode=episode,
            array=self._learner.state_action_values_std,
        )
        self._write_array(
            tag=constants.Constants.POLICY_ENTROPY,
            episode=episode,
            array=self._learner.policy_entropy,
        )
        self._write_array(
            tag=constants.Constants.VISITATION_COUNTS,
            episode=episode,
            array=self._learner.state_visitation_counts,
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
            self._logger.info("Training ensemble of learners in parallel...")
        else:
            train_fn = self._serial_train_episode
            self._logger.info("Training ensemble of learners in serial...")

        # get proxy for rng state
        rng_state = int(np.mean(np.random.get_state()[1])) % (10000)
        self._logger.info(rng_state)

        (
            ensemble_rewards,
            ensemble_step_counts,
            ensemble_mean_penalties,
            ensemble_mean_penalty_infos,
            ensemble_std_penalty_infos,
        ) = train_fn(episode=episode, rng_state=rng_state)

        # set again, to ensure serial/parallel consistency
        experiment_utils.set_random_seeds(rng_state)

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
                df_tag=f"{penalty_info}_mean",
            )
        for penalty_info, ensemble_penalty_info in ensemble_std_penalty_infos.items():
            self._write_scalar(
                tag=constants.Constants.STD_PENALTY_INFO,
                episode=episode,
                scalar=np.mean(ensemble_penalty_info),
                df_tag=f"{penalty_info}_std",
            )

        return mean_reward, mean_step_count

    def _serial_train_episode(
        self,
        episode: int,
        rng_state: int,
    ) -> Tuple[float, int]:
        """Perform the train episode for each learner in ensemble serially.

        Args:
            episode: index of episode
        """
        ensemble_episode_rewards = []
        ensemble_episode_step_counts = []
        mean_penalties = []
        mean_penalty_infos = {}
        std_penalty_infos = {}

        for i, learner in enumerate(self._learner.ensemble):
            self._logger.info(f"Training learner {i}/{len(self._learner.ensemble)}...")
            (
                _,
                episode_reward,
                episode_count,
                mean_penalty,
                mean_penalty_info,
                std_penalty_info,
            ) = self._single_train_episode(
                environment=copy.deepcopy(self._environment),
                learner=learner,
                visitation_penalty=self._visitation_penalty,
                episode=episode,
                learner_seed=i * rng_state,
            )

            ensemble_episode_rewards.append(episode_reward)
            ensemble_episode_step_counts.append(episode_count)
            mean_penalties.append(mean_penalty)
            for info_key, mean_info in mean_penalty_info.items():
                if info_key not in mean_penalty_infos:
                    mean_penalty_infos[info_key] = []
                mean_penalty_infos[info_key].append(mean_info)
            for info_key, std_info in std_penalty_info.items():
                if info_key not in std_penalty_infos:
                    std_penalty_infos[info_key] = []
                std_penalty_infos[info_key].append(std_info)

        return (
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_infos,
            std_penalty_infos,
        )

    def _parallelised_train_episode(
        self,
        episode: int,
        rng_state: int,
    ) -> Tuple[float, int]:
        """Perform the train episode for each learner in ensemble in parallel.

        Args:
            episode: index of episode
        """
        processes_arguments = [
            (
                copy.deepcopy(self._environment),
                learner,
                self._visitation_penalty,
                episode,
                i * rng_state,
            )
            for i, learner in enumerate(self._learner.ensemble)
        ]
        processes_results = self._pool.starmap(
            self._single_train_episode, processes_arguments
        )
        (
            learners,
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_info,
            std_penalty_info,
        ) = zip(*processes_results)

        self._learner.ensemble = list(learners)

        mean_penalty_infos = {}
        std_penalty_infos = {}
        for per_learner_mean_penalty_info in mean_penalty_info:
            for info_key, mean_info in per_learner_mean_penalty_info.items():
                if info_key not in mean_penalty_infos:
                    mean_penalty_infos[info_key] = []
                mean_penalty_infos[info_key].append(mean_info)
        for per_learner_std_penalty_info in std_penalty_info:
            for info_key, std_info in per_learner_std_penalty_info.items():
                if info_key not in std_penalty_infos:
                    std_penalty_infos[info_key] = []
                std_penalty_infos[info_key].append(std_info)

        return (
            ensemble_episode_rewards,
            ensemble_episode_step_counts,
            mean_penalties,
            mean_penalty_infos,
            std_penalty_infos,
        )

    @staticmethod
    def _single_train_episode(
        environment: base_environment.BaseEnvironment,
        learner: base_learner.BaseLearner,
        visitation_penalty,
        episode: int,
        learner_seed: int,
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
        # in parallel processing each sub-process inherits rng of parent.
        # to guarantee independently random sub-processes, we reset seeds here
        # based on learner id.

        experiment_utils.set_random_seeds(learner_seed)

        episode_reward = 0

        penalties = []
        penalty_infos = {}

        state = environment.reset_environment(train=True)

        while environment.active:
            action = learner.select_behaviour_action(state)
            reward, next_state = environment.step(action)

            penalty, penalty_info = visitation_penalty(
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
        std_penalty_info = {k: np.std(v) for k, v in penalty_infos.items()}

        return (
            learner,
            episode_reward,
            environment.episode_step_count,
            mean_penalties,
            mean_penalty_info,
            std_penalty_info,
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

    def _post_visualisation(self):
        post_visualisations_path = os.path.join(
            self._checkpoint_path, constants.Constants.POST_VISUALISATIONS
        )

        value_function_visualisations = {
            constants.Constants.VALUE_FUNCTION: constants.Constants.MAX,
            constants.Constants.VALUE_FUNCTION_STD: constants.Constants.MEAN,
            constants.Constants.POLICY_ENTROPY: constants.Constants.MEAN,
            constants.Constants.VISITATION_COUNTS: constants.Constants.MEAN,
        }

        for tag, over_actions in value_function_visualisations.items():
            if tag in self._post_visualisations:
                os.makedirs(post_visualisations_path, exist_ok=True)
                array_path = os.path.join(self._array_folder_path, tag)
                all_value_paths = sorted(
                    os.listdir(array_path),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                # dictionary has been saved, so need to call [()]
                all_values = [
                    np.load(os.path.join(array_path, f), allow_pickle=True)[()]
                    for f in all_value_paths
                ]
                self._environment.animate_value_function(
                    all_values=all_values,
                    save_path=os.path.join(post_visualisations_path, f"{tag}.gif"),
                    over_actions=over_actions,
                )
