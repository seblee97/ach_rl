import copy
import itertools
import os
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

from ach_rl import constants
from ach_rl.experiments import ach_config
from ach_rl.learners.ensemble_learners import tabular_ensemble_learner
from ach_rl.learners.ensemble_learners.majority_vote_ensemble_learner import \
    MajorityVoteEnsemble
from ach_rl.learners.ensemble_learners.mean_greedy_ensemble_learner import \
    MeanGreedyEnsemble
from ach_rl.learners.ensemble_learners.sample_greedy_ensemble_learner import \
    SampleGreedyEnsemble
from ach_rl.learners.tabular_learners import q_learner
from ach_rl.runners import base_runner
from ach_rl.utils import decorators
from ach_rl.visitation_penalties.adaptive_arriving_uncertainty_visitation_penalty import \
    AdaptiveArrivingUncertaintyPenalty
from ach_rl.visitation_penalties.adaptive_uncertainty_visitation_penalty import \
    AdaptiveUncertaintyPenalty
from ach_rl.visitation_penalties.hard_coded_visitation_penalty import \
    HardCodedPenalty
from ach_rl.visitation_penalties.potential_adaptive_uncertainty_penalty import \
    PotentialAdaptiveUncertaintyPenalty


class MaskedEnsembleQLearningRunner(base_runner.BaseRunner):
    """Runner for Q-learning ensemble, where experiences are shared between ALL 
    members of the ensemble (as opposed to completely independent learners), but 
    where updates at each step are made by a subset of the ensemble (masked)."""

    def __init__(self, config: ach_config.AchConfig, unique_id: str):

        self._num_learners = config.num_learners
        self._targets = config.targets
        self._mask_probability = config.mask_probability
        self._ensemble_behaviour = config.behaviour

        if self._ensemble_behaviour == constants.GREEDY_SAMPLE:
            self._action_selection = SampleGreedyEnsemble.select_target_action
        elif self._ensemble_behaviour == constants.GREEDY_MEAN:
            self._action_selection = MeanGreedyEnsemble.select_target_action
        elif self._ensemble_behaviour == constants.GREEDY_VOTE:
            self._action_selection = MajorityVoteEnsemble.select_target_action

        super().__init__(config=config, unique_id=unique_id)

        self._information_computer_period = config.information_computer_update_period

    def _get_data_columns(self):
        columns = [
            constants.TRAIN_EPISODE_REWARD,
            constants.TRAIN_EPISODE_LENGTH,
            # constants.MEAN_VISITATION_PENALTY,
            # f"{constants.NEXT_STATE_POLICY_ENTROPY}_{constants.MEAN}",
            # f"{constants.CURRENT_STATE_POLICY_ENTROPY}_{constants.MEAN}",
            # f"{constants.NEXT_STATE_MEAN_UNCERTAINTY}_{constants.MEAN}",
            # f"{constants.NEXT_STATE_MAX_UNCERTAINTY}_{constants.MEAN}",
            # f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_SAMPLE}",
            # f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_MEAN}",
            # f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_VOTE}",
            # f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_SAMPLE}",
            # f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_MEAN}",
            # f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_VOTE}",
            # f"{constants.CURRENT_STATE_MAX_UNCERTAINTY}_{constants.MEAN}",
            # f"{constants.CURRENT_STATE_MEAN_UNCERTAINTY}_{constants.MEAN}",
            # f"{constants.CURRENT_STATE_SELECT_UNCERTAINTY}_{constants.MEAN}",
            # constants.ENSEMBLE_EPISODE_REWARD_STD,
            # constants.ENSEMBLE_EPISODE_LENGTH_STD,
        ]

        return columns

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
            if config.variance_estimation:
                single_variance_learner = self._get_individual_q_learner(
                    config=config, initialisation_strategy=initialisation_strategy
                )
                variance_learners = [
                    copy.deepcopy(single_variance_learner)
                    for _ in range(self._num_learners)
                ]
            else:
                variance_learners = None
        else:
            learners = [
                self._get_individual_q_learner(
                    config=config, initialisation_strategy=initialisation_strategy
                )
                for _ in range(self._num_learners)
            ]
            if config.variance_estimation:
                variance_learners = [self._get_individual_q_learner(
                        config=config, initialisation_strategy=initialisation_strategy
                    )
                    for _ in range(self._num_learners)
                ]
            else:
                variance_learners = None

        learner = tabular_ensemble_learner.TabularEnsembleLearner(
            learner_ensemble=learners,
            variance_learner_ensemble=variance_learners
        )

        if config.pretrained_model_path is not None:
            learner.load_model(config.pretrained_model_path)

        return learner

    def _get_initialisation_strategy(self, config: ach_config.AchConfig):
        if config.initialisation_type == constants.RANDOM_UNIFORM:
            initialisation_strategy = {
                constants.RANDOM_UNIFORM: {
                    constants.LOWER_BOUND: config.lower_bound,
                    constants.UPPER_BOUND: config.upper_bound,
                }
            }
        elif config.initialisation_type == constants.RANDOM_NORMAL:
            initialisation_strategy = {
                constants.RANDOM_NORMAL: {
                    constants.MEAN: config.mean,
                    constants.VARIANCE: config.variance,
                }
            }
        else:
            initialisation_strategy == {config.initialisation_type}
        return initialisation_strategy

    def _get_individual_q_learner(
        self, config: ach_config.AchConfig, initialisation_strategy: Dict
    ):
        """Setup a single q-learner."""
        return q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target="",
            initialisation_strategy=initialisation_strategy,
            # epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            split_value_function=config.split_value_function,
        )

    def _pre_episode_log(self, episode: int):
        """Logging pre-episode. Includes value-function, individual run."""
        self._logger.info(f"Episode {episode + 1}: pre-episode data logging...")
        self._pre_episode_visualisations(episode=episode)
        self._pre_episode_array_logging(episode=episode)

    @decorators.timer
    def _pre_episode_visualisations(self, episode: int):
        visualisation_configurations = [
            (constants.MAX_VALUES_PDF, True, False),
            (constants.QUIVER_VALUES_PDF, False, True),
            (constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(constants.VALUE_FUNCTION, episode):
            averaged_state_action_values = self._learner.state_action_values
            # tuple of save_path_tag, plot_max_values (bool), quiver (bool)
            if self._parallelise_ensemble:

                self._logger.info("Parallel value function visualisation...")
                processes_arguments = [
                    (
                        averaged_state_action_values,
                        visualisation_configuration[1],
                        visualisation_configuration[2],
                        constants.MAX,
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
                    averaged_values = (
                        self._environment.average_values_over_positional_states(
                            self._learner.state_action_values
                        )
                    )
                    averaged_max_values = {
                        p: max(v) for p, v in averaged_values.items()
                    }
                    self._environment.plot_heatmap_over_env(
                        heatmap=averaged_max_values,
                        save_name=os.path.join(
                            self._visualisations_folder_path,
                            f"{episode}_{visualisation_configuration[0]}",
                        ),
                    )

        if self._visualisation_iteration(constants.INDIVIDUAL_VALUE_FUNCTIONS, episode):
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
                        constants.MAX,
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
                            over_actions=constants.MAX,
                        )

        if self._visualisation_iteration(constants.VALUE_FUNCTION_STD, episode):
            self._logger.info("Standard deviation value function visualisation...")
            state_action_values_std = self._learner.state_action_values_std
            self._environment.plot_value_function(
                values=state_action_values_std,
                save_path=os.path.join(
                    self._visualisations_folder_path,
                    f"{episode}_{constants.VALUE_FUNCTION_STD_PDF}",
                ),
                plot_max_values=True,
                quiver=False,
                over_actions=constants.MEAN,
            )

    def _generate_visualisations(self, episode: int):
        if episode != 0:
            if self._visualisation_iteration(constants.INDIVIDUAL_TRAIN_RUN, episode):
                self._environment.visualise_episode_history(
                    save_path=os.path.join(
                        self._rollout_folder_path,
                        f"{constants.INDIVIDUAL_TRAIN_RUN}_{episode}.mp4",
                    ),
                    history=self._latest_train_history,
                )

    @decorators.timer
    def _pre_episode_array_logging(self, episode: int):
        self._write_array(
            tag=constants.VALUE_FUNCTION,
            episode=episode,
            array=self._learner.state_action_values,
        )
        self._write_array(
            tag=constants.VALUE_FUNCTION_STD,
            episode=episode,
            array=self._learner.state_action_values_std,
        )
        self._write_array(
            tag=constants.POLICY_ENTROPY,
            episode=episode,
            array=self._learner.policy_entropy,
        )
        self._write_array(
            tag=constants.VISITATION_COUNTS,
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
        if episode % self._information_computer_period == 0:
            self._logger.info("Updating global information...")
            self._information_computer.state_action_values = (
                self._learner.individual_learner_state_action_values
            )

        (
            episode_reward,
            episode_step_count,
            mean_penalties,
            mean_epsilons,
            mean_lr_scalings,
            mean_infos,
            std_infos,
            train_episode_history,
        ) = self._single_train_episode(episode=episode)

        # add episode history to environment (for now arbitrarily choose last of ensemble)
        # self._environment.train_episode_history = train_episode_histories[-1]
        self._latest_train_history = train_episode_history

        # averages over ensemble
        ensemble_log = {
            constants.TRAIN_EPISODE_REWARD: episode_reward,
            constants.TRAIN_EPISODE_LENGTH: episode_step_count,
        }

        ensemble_average_info_log = {
            f"{info}_mean": np.mean(ensemble_info)
            for info, ensemble_info in mean_infos.items()
        }
        ensemble_std_info_log = {
            f"{info}_std": np.mean(ensemble_info)
            for info, ensemble_info in std_infos.items()
        }

        logging_dict = {
            **ensemble_log,
            **ensemble_average_info_log,
            **ensemble_std_info_log,
        }

        return logging_dict

    def _single_train_episode(
        self,
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
        epsilons = []
        lr_scalings = []

        state_infos = {}

        state = self._environment.reset_environment(train=True)

        while self._environment.active:

            current_state_info = self._information_computer.compute_state_information(
                state=state, state_label=constants.CURRENT
            )

            epsilon = self._epsilon_computer(
                episode=episode, epsilon_info=current_state_info
            )

            action = self._action_selection(learners=self._learner.ensemble, state=state)
            reward, next_state = self._environment.step(action)

            next_state_info = self._information_computer.compute_state_information(
                state=next_state, state_label=constants.NEXT
            )
            select_info = self._information_computer.compute_state_select_information(
                state=state, action=action
            )

            state_info = {**current_state_info, **select_info, **next_state_info}

            penalty = self._visitation_penalty(episode=episode, penalty_info=state_info)
            lr_scaling = self._lr_scaler(episode=episode, lr_scaling_info=state_info)

            penalties.append(penalty)
            lr_scalings.append(lr_scaling)
            epsilons.append(epsilon)

            for info_key, info in state_info.items():
                if info_key not in state_infos.keys():
                    state_infos[info_key] = []
                state_infos[info_key].append(info)

            mask = np.random.random(size=self._num_learners) < self._mask_probability

            for learner, m in zip(self._learner.ensemble, mask):
                if m:
                    learner.step(
                        state,
                        action,
                        reward,
                        next_state,
                        self._environment.active,
                        penalty,
                        lr_scaling,
                    )
            state = next_state
            episode_reward += reward

        mean_penalties = np.mean(penalties)
        mean_info = {k: np.mean(v) for k, v in state_infos.items()}
        std_info = {k: np.std(v) for k, v in state_infos.items()}

        mean_epsilons = np.mean(epsilons)
        mean_lr_scalings = np.mean(lr_scalings)

        return (
            episode_reward,
            self._environment.episode_step_count,
            mean_penalties,
            mean_epsilons,
            mean_lr_scalings,
            mean_info,
            std_info,
            self._environment.train_episode_history,
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
        greedy_sample = constants.GREEDY_SAMPLE
        greedy_mean = constants.GREEDY_MEAN
        greedy_vote = constants.GREEDY_VOTE

        no_rep_greedy_sample = "_".join([constants.NO_REP, constants.GREEDY_SAMPLE])
        no_rep_greedy_mean = "_".join([constants.NO_REP, constants.GREEDY_MEAN])
        no_rep_greedy_vote = "_".join([constants.NO_REP, constants.GREEDY_VOTE])

        if greedy_sample in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=SampleGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_sample}",
            )
        if greedy_mean in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=MeanGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_mean}",
            )
        if greedy_vote in self._targets:
            self._greedy_test_episode(
                episode=episode,
                action_selection_method=MajorityVoteEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{greedy_vote}",
            )
        if no_rep_greedy_sample in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=SampleGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_sample}",
            )
        if no_rep_greedy_mean in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=MeanGreedyEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_mean}",
            )
        if no_rep_greedy_vote in self._targets:
            self._non_repeat_test_episode(
                episode=episode,
                action_selection_method=MajorityVoteEnsemble.select_target_action,
                action_selection_method_args={
                    constants.LEARNERS: self._learner.ensemble
                },
                tag_=f"_{no_rep_greedy_vote}",
            )

    def _post_visualisation(self):
        post_visualisations_path = os.path.join(
            self._checkpoint_path, constants.POST_VISUALISATIONS
        )

        value_function_visualisations = {
            constants.VALUE_FUNCTION: constants.MAX,
            constants.VALUE_FUNCTION_STD: constants.MEAN,
            constants.POLICY_ENTROPY: constants.MEAN,
            constants.VISITATION_COUNTS: constants.MEAN,
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
