import copy
import os
import random
from typing import Tuple
from typing import Union

import constants
import numpy as np
import torch
from experiments import ach_config
from learners.deep_learners import dqn_learner
from learners.deep_learners import multi_head_dqn_learner
from learners.deep_learners.components import replay_buffer
from learners.ensemble_learners.majority_vote_ensemble_learner import \
    MajorityVoteEnsemble
from learners.ensemble_learners.mean_greedy_ensemble_learner import \
    MeanGreedyEnsemble
from learners.ensemble_learners.sample_greedy_ensemble_learner import \
    SampleGreedyEnsemble
from runners import base_runner
from visitation_penalties.adaptive_arriving_uncertainty_visitation_penalty import \
    AdaptiveArrivingUncertaintyPenalty
from visitation_penalties.adaptive_uncertainty_visitation_penalty import \
    AdaptiveUncertaintyPenalty
from visitation_penalties.hard_coded_visitation_penalty import HardCodedPenalty
from visitation_penalties.potential_adaptive_uncertainty_penalty import \
    PotentialAdaptiveUncertaintyPenalty


class DQNRunner(base_runner.BaseRunner):
    """Runner for DQN ensemble where agent has shared features and separate head layers."""

    def __init__(self, config: ach_config.AchConfig):
        self._num_learners: Union[int, None]
        self._mask_probability = Union[float, None]
        self._ensemble: bool

        self._targets = config.targets
        super().__init__(config=config)

        self._device = config.experiment_device
        self._batch_size = config.batch_size
        self._shaping_implementation = config.shaping_implementation

        if self._visitation_penalty is not None:
            self._visitation_penalty.q_network = copy.deepcopy(self._learner.q_network)
            self._visitation_penalty.target_q_network = copy.deepcopy(
                self._learner.target_q_network
            )

        self._replay_buffer = self._setup_replay_buffer(config=config)
        self._fill_replay_buffer(num_trajectories=config.num_replay_fill_trajectories)

    def _setup_replay_buffer(
        self, config: ach_config.AchConfig
    ) -> replay_buffer.ReplayBuffer:
        """Instantiate replay buffer object to store experiences."""
        state_dim = tuple(config.encoded_state_dimensions)
        replay_size = config.replay_buffer_size
        penalties = config.shaping_implementation in [
            constants.Constants.TRAIN_Q_NETWORK,
            constants.Constants.TRAIN_TARGET_NETWORK,
        ]
        return replay_buffer.ReplayBuffer(
            replay_size=replay_size,
            state_dim=state_dim,
            mask_length=self._num_learners,
            penalties=penalties,
        )

    def _get_random_mask(self):
        return np.random.choice(
            [0, 1],
            size=self._num_learners,
            p=[self._mask_probability, 1 - self._mask_probability],
        )

    def _fill_replay_buffer(self, num_trajectories: int):
        """Build up store of experiences before training begins.

        Args:
            num_trajectories: number of experience tuples to collect before training.
        """
        self._logger.info("Filling replay buffer...")

        state = self._environment.reset_environment(train=True)
        for _ in range(num_trajectories):
            action = random.choice(self._environment.action_space)
            reward, next_state = self._environment.step(action)

            if (
                self._visitation_penalty is None
                or self._shaping_implementation == constants.Constants.ACT
            ):
                penalty = None
            else:
                penalty, _ = self._visitation_penalty(
                    episode=0,
                    state=torch.from_numpy(state).to(
                        device=self._device, dtype=torch.float
                    ),
                    action=action,
                    next_state=torch.from_numpy(next_state).to(
                        device=self._device, dtype=torch.float
                    ),
                )

            if self._ensemble:
                mask = self._get_random_mask()
            else:
                mask = None

            self._replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                active=self._environment.active,
                mask=mask,
                penalty=penalty,
            )
            if not self._environment.active:
                state = self._environment.reset_environment(train=True)
            else:
                state = next_state

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        if config.type == constants.Constants.BOOTSTRAPPED_ENSEMBLE_DQN:
            self._num_learners = config.num_learners
            self._mask_probability = config.mask_probability
            self._ensemble = True
            learner = multi_head_dqn_learner.MultiHeadDQNLearner(
                action_space=self._environment.action_space,
                state_dimensions=tuple(config.encoded_state_dimensions),
                layer_specifications=config.layer_specifications,
                shared_layers=config.shared_layers,
                num_branches=self._num_learners,
                optimiser_type=config.optimiser,
                epsilon=self._epsilon_function,
                learning_rate=config.learning_rate,
                momentum=config.gradient_momentum,
                eps=config.min_squared_gradient,
                gamma=config.discount_factor,
                target_network_update_period=config.target_network_update_period,
                device=config.experiment_device,
                gradient_clipping=config.gradient_clipping,
            )
        elif config.type == constants.Constants.VANILLA_DQN:
            self._num_learners = None
            self._mask_probability = None
            self._ensemble = False
            learner = dqn_learner.DQNLearner(
                action_space=self._environment.action_space,
                state_dimensions=tuple(config.encoded_state_dimensions),
                layer_specifications=config.layer_specifications,
                optimiser_type=config.optimiser,
                epsilon=self._epsilon_function,
                learning_rate=config.learning_rate,
                momentum=config.gradient_momentum,
                eps=config.min_squared_gradient,
                gamma=config.discount_factor,
                target_network_update_period=config.target_network_update_period,
                device=config.experiment_device,
                gradient_clipping=config.gradient_clipping,
            )
        else:
            raise ValueError(f"Learner type {learner} not recognised.")
        return learner

    def _pre_episode_log(self, episode: int):
        """Logging pre-episode. Includes value-function, individual run."""
        visualisation_configurations = [
            (constants.Constants.MAX_VALUES_PDF, True, False),
            (constants.Constants.QUIVER_VALUES_PDF, False, True),
            (constants.Constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(
            constants.Constants.VALUE_FUNCTION, episode
        ) or self._array_log_iteration(constants.Constants.VALUE_FUNCTION, episode):
            # compute state action values for a tabular env
            state_action_values = {}
            for tuple_state in self._environment.state_space:
                with torch.no_grad():
                    np_state_representation = (
                        self._environment.get_state_representation(
                            tuple_state=tuple_state
                        )
                    )
                    # stacked_representation = np.repeat(
                    #     np_state_representation, self._environment.frame_stack, 0
                    # )
                    pixel_state = torch.from_numpy(
                        np_state_representation
                    ).to(device=self._device, dtype=torch.float)
                    state_action_values[tuple_state] = (
                        self._learner.q_network(pixel_state).cpu().numpy().squeeze()
                    )

        if self._visualisation_iteration(constants.Constants.VALUE_FUNCTION, episode):
            for visualisation_configuration in visualisation_configurations:
                self._logger.info(
                    "Serial value function visualisation: "
                    f"{visualisation_configuration[0]}"
                )
                self._environment.plot_value_function(
                    values=state_action_values,
                    save_path=os.path.join(
                        self._visualisations_folder_path,
                        f"{episode}_{visualisation_configuration[0]}",
                    ),
                    plot_max_values=visualisation_configuration[1],
                    quiver=visualisation_configuration[2],
                    over_actions=constants.Constants.MAX,
                )
        if self._visualisation_iteration(
            constants.Constants.VISITATION_COUNT_HEATMAP, episode
        ):
            self._environment.plot_value_function(
                values=self._learner.state_visitation_counts,
                save_path=os.path.join(
                    self._visualisations_folder_path,
                    f"{episode}_{constants.Constants.VISITATION_COUNT_HEATMAP_PDF}",
                ),
                plot_max_values=True,
                quiver=False,
                over_actions=constants.Constants.SELECT,
            )
        if episode != 0:
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_TRAIN_RUN, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TRAIN_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_TEST_RUN, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(train=False),
                )
        if self._array_log_iteration(constants.Constants.VALUE_FUNCTION, episode):
            self._write_array(
                tag=constants.Constants.VALUE_FUNCTION,
                episode=episode,
                array=state_action_values,
            )

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop (per learner in ensemble).

        Args:
            episode: index of episode

        Returns:
            episode_reward: mean scalar reward accumulated over ensemble episodes.
            num_steps: mean number of steps taken for ensemble episodes.
        """
        if self._visitation_penalty is not None:
            self._visitation_penalty.q_network = copy.deepcopy(self._learner.q_network)
            self._visitation_penalty.target_q_network = copy.deepcopy(
                self._learner.target_q_network
            )

        if self._ensemble:
            # select branch uniformly at random for rollout
            branch = random.choice(range(self._num_learners))

        episode_reward = 0
        episode_loss = 0

        acting_penalties = []
        acting_penalties_infos = {}

        sample_penalties = []
        sample_penalties_infos = {}

        state = self._environment.reset_environment(train=True)

        while self._environment.active:

            if self._ensemble:
                action = self._learner.select_behaviour_action(state, branch=branch)
            else:
                action = self._learner.select_behaviour_action(state)
            reward, next_state = self._environment.step(action)

            if self._visitation_penalty is not None:
                acting_penalty, acting_penalty_info = self._visitation_penalty(
                    episode=episode,
                    state=torch.from_numpy(state).to(
                        device=self._device, dtype=torch.float
                    ),
                    action=action,
                    next_state=torch.from_numpy(next_state).to(
                        device=self._device, dtype=torch.float
                    ),
                )

                acting_penalties.append(acting_penalty)

                for info_key, info in acting_penalty_info.items():
                    if info_key not in acting_penalties_infos.keys():
                        acting_penalties_infos[info_key] = []
                    acting_penalties_infos[info_key].append(info)

            if self._visitation_penalty is None:
                buffer_penalty = None
            else:
                if self._shaping_implementation == constants.Constants.ACT:
                    buffer_penalty = None
                else:
                    buffer_penalty = acting_penalty
            if self._ensemble:
                mask = self._get_random_mask()
            else:
                mask = None

            self._replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                active=self._environment.active,
                mask=mask,
                penalty=buffer_penalty,
            )

            experience_sample = self._replay_buffer.sample(self._batch_size)

            state_sample = torch.from_numpy(experience_sample[0]).to(
                device=self._device, dtype=torch.float
            )
            action_sample = torch.from_numpy(experience_sample[1]).to(
                device=self._device, dtype=torch.int
            )
            reward_sample = torch.from_numpy(experience_sample[2]).to(
                device=self._device, dtype=torch.float
            )
            next_state_sample = torch.from_numpy(experience_sample[3]).to(
                device=self._device, dtype=torch.float
            )
            active_sample = torch.from_numpy(experience_sample[4]).to(
                device=self._device, dtype=torch.int
            )
            if self._ensemble:
                mask_sample = torch.from_numpy(experience_sample[5]).to(
                    device=self._device, dtype=torch.int
                )

            if self._visitation_penalty is None:
                penalty = 0
            else:
                sample_penalty, sample_penalty_infos = self._visitation_penalty(
                    episode=episode,
                    state=state_sample,
                    action=experience_sample[1],
                    next_state=next_state_sample,
                )
                sample_penalties.append(np.mean(sample_penalty))
                for info_key, info in sample_penalty_infos.items():
                    if info_key not in sample_penalties_infos.keys():
                        sample_penalties_infos[info_key] = []
                    sample_penalties_infos[info_key].append(np.mean(info))

                if self._shaping_implementation in [
                    constants.Constants.TRAIN_Q_NETWORK,
                    constants.Constants.TRAIN_TARGET_NETWORK,
                ]:
                    penalty = torch.Tensor(sample_penalty).to(
                        device=self._device, dtype=torch.float
                    )
                elif self._shaping_implementation in [constants.Constants.ACT]:
                    penalty = torch.from_numpy(experience_sample[6]).to(
                        device=self._device, dtype=torch.float
                    )

            if self._ensemble:
                loss, epsilon = self._learner.step(
                    state=state_sample,
                    action=action_sample,
                    reward=reward_sample,
                    next_state=next_state_sample,
                    active=active_sample,
                    visitation_penalty=penalty,
                    mask=mask_sample,
                )
            else:
                loss, epsilon = self._learner.step(
                    state=state_sample,
                    action=action_sample,
                    reward=reward_sample,
                    next_state=next_state_sample,
                    active=active_sample,
                    visitation_penalty=penalty,
                )

            state = next_state
            episode_reward += reward
            episode_loss += loss

        mean_sample_penalty = np.mean(sample_penalties)
        mean_sample_penalty_info = {
            k: np.mean(v) for k, v in sample_penalties_infos.items()
        }
        std_sample_penalty_info = {
            k: np.std(v) for k, v in sample_penalties_infos.items()
        }
        mean_acting_penalty = np.mean(acting_penalties)
        mean_acting_penalty_info = {
            k: np.mean(v) for k, v in acting_penalties_infos.items()
        }
        std_acting_penalty_info = {
            k: np.std(v) for k, v in acting_penalties_infos.items()
        }

        episode_steps = self._environment.episode_step_count

        self._write_scalar(
            tag=constants.Constants.LOSS,
            episode=episode,
            scalar=episode_loss / episode_steps,
        )
        if self._ensemble:
            for i in range(self._num_learners):
                if i == branch:
                    loss_scalar = episode_loss / episode_steps
                    reward_scalar = episode_reward
                else:
                    loss_scalar = np.nan
                    reward_scalar = np.nan
                self._write_scalar(
                    tag=constants.Constants.BRANCH_LOSS,
                    episode=episode,
                    scalar=loss_scalar,
                    df_tag=f"branch_{i}_loss",
                )
                self._write_scalar(
                    tag=constants.Constants.BRANCH_REWARD,
                    episode=episode,
                    scalar=reward_scalar,
                    df_tag=f"branch_{i}_reward",
                )
        self._write_scalar(
            tag=constants.Constants.EPSILON, episode=episode, scalar=epsilon
        )
        self._write_scalar(
            tag=constants.Constants.MEAN_VISITATION_PENALTY,
            episode=episode,
            scalar=mean_sample_penalty,
            df_tag=f"sample_{constants.Constants.MEAN_VISITATION_PENALTY}",
        )
        self._write_scalar(
            tag=constants.Constants.MEAN_VISITATION_PENALTY,
            episode=episode,
            scalar=mean_acting_penalty,
            df_tag=f"acting_{constants.Constants.MEAN_VISITATION_PENALTY}",
        )
        for penalty_info, ensemble_penalty_info in mean_sample_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.MEAN_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=f"sample_{penalty_info}",
            )
        for penalty_info, ensemble_penalty_info in std_sample_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.STD_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=f"sample_{penalty_info}_std",
            )
        for penalty_info, ensemble_penalty_info in mean_acting_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.MEAN_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=f"acting_{penalty_info}",
            )
        for penalty_info, ensemble_penalty_info in std_acting_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.STD_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=f"acting_{penalty_info}_std",
            )

        return episode_reward, episode_steps

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
        greedy_individual = constants.Constants.GREEDY_INDIVIDUAL

        if self._targets is None:
            pass

        else:
            if greedy_individual in self._targets:
                test_episode_rewards = []
                test_episode_lengths = []
                for i in range(self._num_learners):
                    reward, length = self._greedy_test_episode(
                        episode=episode,
                        action_selection_method=self._learner.select_target_action,
                        action_selection_method_args={constants.Constants.BRANCH: i},
                        tag_=f"_{greedy_individual}_{i}",
                        output=True,
                    )
                    test_episode_rewards.append(reward)
                    test_episode_lengths.append(length)
                self._write_scalar(
                    tag=f"{constants.Constants.TEST}_{constants.Constants.ENSEMBLE_EPISODE_REWARD_MEAN}",
                    episode=episode,
                    scalar=np.mean(test_episode_rewards),
                )
                self._write_scalar(
                    tag=f"{constants.Constants.TEST}_{constants.Constants.ENSEMBLE_EPISODE_LENGTH_MEAN}",
                    episode=episode,
                    scalar=np.mean(test_episode_lengths),
                )
                self._write_scalar(
                    tag=f"{constants.Constants.TEST}_{constants.Constants.ENSEMBLE_EPISODE_REWARD_STD}",
                    episode=episode,
                    scalar=np.std(test_episode_rewards),
                )
                self._write_scalar(
                    tag=f"{constants.Constants.TEST}_{constants.Constants.ENSEMBLE_EPISODE_LENGTH_STD}",
                    episode=episode,
                    scalar=np.std(test_episode_lengths),
                )

            if greedy_sample in self._targets:
                self._greedy_test_episode(
                    episode=episode,
                    action_selection_method=self._learner.select_greedy_sample_target_action,
                    tag_=f"_{greedy_sample}",
                )
            if greedy_mean in self._targets:
                self._greedy_test_episode(
                    episode=episode,
                    action_selection_method=self._learner.select_greedy_mean_target_action,
                    tag_=f"_{greedy_mean}",
                )
            if greedy_vote in self._targets:
                self._greedy_test_episode(
                    episode=episode,
                    action_selection_method=self._learner.select_greedy_vote_target_action,
                    tag_=f"_{greedy_vote}",
                )

    def _post_visualisation(self):
        pass
