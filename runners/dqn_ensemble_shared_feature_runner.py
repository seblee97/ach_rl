import copy
import multiprocessing
import random
from typing import Tuple
from typing import Union

import constants
import numpy as np
import torch
from environments import base_environment
from experiments import ach_config
from learners import base_learner
from learners.deep_learners import dqn_learner
from learners.deep_learners import multi_head_dqn_learner
from learners.deep_learners.components import replay_buffer
from learners.ensemble_learners import deep_ensemble_learner
from learners.ensemble_learners.majority_vote_ensemble_learner import \
    MajorityVoteEnsemble
from learners.ensemble_learners.mean_greedy_ensemble_learner import \
    MeanGreedyEnsemble
from learners.ensemble_learners.sample_greedy_ensemble_learner import \
    SampleGreedyEnsemble
from runners import base_runner
from visitation_penalties import base_visistation_penalty
from visitation_penalties.adaptive_arriving_uncertainty_visitation_penalty import \
    AdaptiveArrivingUncertaintyPenalty
from visitation_penalties.adaptive_uncertainty_visitation_penalty import \
    AdaptiveUncertaintyPenalty
from visitation_penalties.hard_coded_visitation_penalty import HardCodedPenalty
from visitation_penalties.potential_adaptive_uncertainty_penalty import \
    PotentialAdaptiveUncertaintyPenalty


class EnsembleDQNSharedFeatureRunner(base_runner.BaseRunner):
    """Runner for DQN ensemble where agent has shared features and separate head layers."""

    def __init__(self, config: ach_config.AchConfig):
        self._num_learners = config.num_learners
        self._targets = config.targets
        super().__init__(config=config)

        self._device = config.experiment_device
        self._batch_size = config.batch_size
        self._mask_probability = config.mask_probability

        self._replay_buffer = self._setup_replay_buffer(config=config)
        self._fill_replay_buffer(
            num_trajectories=config.num_replay_fill_trajectories)

    def _setup_replay_buffer(
            self, config: ach_config.AchConfig) -> replay_buffer.ReplayBuffer:
        """Instantiate replay buffer object to store experiences."""
        state_dim = tuple(config.encoded_state_dimensions)
        replay_size = config.replay_buffer_size
        return replay_buffer.ReplayBuffer(replay_size=replay_size,
                                          state_dim=state_dim,
                                          mask_length=self._num_learners)

    def _get_random_mask(self):
        return np.random.choice(
            [0, 1],
            size=self._num_learners,
            p=[self._mask_probability, 1 - self._mask_probability])

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

            self._replay_buffer.add(state=state,
                                    action=action,
                                    reward=reward,
                                    next_state=next_state,
                                    active=self._environment.active,
                                    mask=self._get_random_mask())
            if not self._environment.active:
                state = self._environment.reset_environment(train=True)
            else:
                state = next_state

    def _setup_learner(self,
                       config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        return multi_head_dqn_learner.MultiHeadDQNLearner(
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
            gradient_clipping=config.gradient_clipping)

    def _pre_episode_log(self, episode: int):
        """Logging pre-episode. Includes value-function, individual run."""
        pass

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop (per learner in ensemble).

        Args:
            episode: index of episode

        Returns:
            episode_reward: mean scalar reward accumulated over ensemble episodes.
            num_steps: mean number of steps taken for ensemble episodes.
        """
        self._visitation_penalty.q_network = copy.deepcopy(
            self._learner.q_network)

        # select branch uniformly at random for rollout
        branch = random.choice(range(self._num_learners))

        episode_reward = 0
        episode_loss = 0

        penalties = []
        penalty_infos = {}

        state = self._environment.reset_environment(train=True)

        while self._environment.active:

            action = self._learner.select_behaviour_action(state, branch=branch)
            reward, next_state = self._environment.step(action)

            penalty, penalty_info = self._visitation_penalty(
                episode=episode,
                state=torch.from_numpy(state).to(device=self._device,
                                                 dtype=torch.float),
                action=action,
                next_state=torch.from_numpy(next_state).to(device=self._device,
                                                           dtype=torch.float))

            penalties.append(penalty)
            for info_key, info in penalty_info.items():
                if info_key not in penalty_infos.keys():
                    penalty_infos[info_key] = []
                penalty_infos[info_key].append(info)

            self._replay_buffer.add(state=state,
                                    action=action,
                                    reward=reward,
                                    next_state=next_state,
                                    active=self._environment.active,
                                    mask=self._get_random_mask())

            experience_sample = self._replay_buffer.sample(self._batch_size)

            loss, epsilon = self._learner.step(
                state=torch.from_numpy(experience_sample[0]).to(
                    device=self._device, dtype=torch.float),
                action=torch.from_numpy(experience_sample[1]).to(
                    device=self._device, dtype=torch.int),
                reward=torch.from_numpy(experience_sample[2]).to(
                    device=self._device, dtype=torch.float),
                next_state=torch.from_numpy(experience_sample[3]).to(
                    device=self._device, dtype=torch.float),
                active=torch.from_numpy(experience_sample[4]).to(
                    device=self._device, dtype=torch.int),
                visitation_penalty=penalty,
                mask=torch.from_numpy(experience_sample[5]).to(
                    device=self._device, dtype=torch.int))

            state = next_state
            episode_reward += reward
            episode_loss += loss

        mean_penalty = np.mean(penalties)
        mean_penalty_info = {k: np.mean(v) for k, v in penalty_infos.items()}
        std_penalty_info = {k: np.std(v) for k, v in penalty_infos.items()}
        episode_steps = self._environment.episode_step_count

        self._write_scalar(
            tag=constants.Constants.LOSS,
            episode=episode,
            scalar=episode_loss / episode_steps,
        )
        self._write_scalar(
            tag=constants.Constants.MEAN_VISITATION_PENALTY,
            episode=episode,
            scalar=mean_penalty,
        )
        for penalty_info, ensemble_penalty_info in mean_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.MEAN_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=penalty_info,
            )
        for penalty_info, ensemble_penalty_info in std_penalty_info.items():
            self._write_scalar(
                tag=constants.Constants.STD_PENALTY_INFO,
                episode=episode,
                scalar=ensemble_penalty_info,
                df_tag=f"{penalty_info}_std",
            )

        return episode_reward, episode_steps

    def _get_visitation_penalty(self, episode: int, state, action: int,
                                next_state):
        if isinstance(self._visitation_penalty, AdaptiveUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(state=state,
                                                             action=action)
        elif isinstance(self._visitation_penalty, HardCodedPenalty):
            penalty, penalty_info = self._visitation_penalty(episode=episode)
        elif isinstance(self._visitation_penalty,
                        PotentialAdaptiveUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(
                state=state, action=action, next_state=next_state)
        elif isinstance(self._visitation_penalty,
                        AdaptiveArrivingUncertaintyPenalty):
            penalty, penalty_info = self._visitation_penalty(
                next_state=next_state)
        return penalty, penalty_info

    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner.

        Here, there are various methods for performing inference.
        """
        pass
        # greedy_sample = constants.Constants.GREEDY_SAMPLE
        # greedy_mean = constants.Constants.GREEDY_MEAN
        # greedy_vote = constants.Constants.GREEDY_VOTE

        # no_rep_greedy_sample = "_".join(
        #     [constants.Constants.NO_REP, constants.Constants.GREEDY_SAMPLE])
        # no_rep_greedy_mean = "_".join(
        #     [constants.Constants.NO_REP, constants.Constants.GREEDY_MEAN])
        # no_rep_greedy_vote = "_".join(
        #     [constants.Constants.NO_REP, constants.Constants.GREEDY_VOTE])

        # if greedy_sample in self._targets:
        #     self._greedy_test_episode(
        #         episode=episode,
        #         action_selection_method=SampleGreedyEnsemble.
        #         select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{greedy_sample}",
        #     )
        # if greedy_mean in self._targets:
        #     self._greedy_test_episode(
        #         episode=episode,
        #         action_selection_method=MeanGreedyEnsemble.select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{greedy_mean}",
        #     )
        # if greedy_vote in self._targets:
        #     self._greedy_test_episode(
        #         episode=episode,
        #         action_selection_method=MajorityVoteEnsemble.
        #         select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{greedy_vote}",
        #     )
        # if no_rep_greedy_sample in self._targets:
        #     self._non_repeat_test_episode(
        #         episode=episode,
        #         action_selection_method=SampleGreedyEnsemble.
        #         select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{no_rep_greedy_sample}",
        #     )
        # if no_rep_greedy_mean in self._targets:
        #     self._non_repeat_test_episode(
        #         episode=episode,
        #         action_selection_method=MeanGreedyEnsemble.select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{no_rep_greedy_mean}",
        #     )
        # if no_rep_greedy_vote in self._targets:
        #     self._non_repeat_test_episode(
        #         episode=episode,
        #         action_selection_method=MajorityVoteEnsemble.
        #         select_target_action,
        #         action_selection_method_args={
        #             constants.Constants.LEARNERS: self._learner.ensemble
        #         },
        #         tag_=f"_{no_rep_greedy_vote}",
        #     )

    def _post_visualisation(self):
        pass
