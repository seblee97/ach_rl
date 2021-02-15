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

        self._parallelise_ensemble = config.parallelise_ensemble
        if self._parallelise_ensemble:
            num_cores = multiprocessing.cpu_count()
            self._pool = multiprocessing.Pool(processes=num_cores)

        self._device = config.experiment_device

        self._batch_size = config.batch_size
        self._share_replay_buffer = config.share_replay_buffer

        if self._share_replay_buffer:
            self._replay_buffer = self._setup_individual_replay_buffer(
                config=config)
            self._fill_replay_buffer(
                replay_buffer=self._replay_buffer,
                num_trajectories=config.num_replay_fill_trajectories)
        else:
            self._replay_buffers = [
                self._setup_individual_replay_buffer(config=config)
                for _ in range(self._num_learners)
            ]
            for rep_buffer in self._replay_buffers:
                self._fill_replay_buffer(
                    replay_buffer=rep_buffer,
                    num_trajectories=config.num_replay_fill_trajectories)

    def _setup_individual_replay_buffer(
            self, config: ach_config.AchConfig) -> replay_buffer.ReplayBuffer:
        """Instantiate replay buffer object to store experiences."""
        state_dim = tuple(config.encoded_state_dimensions)
        replay_size = config.replay_buffer_size
        return replay_buffer.ReplayBuffer(replay_size=replay_size,
                                          state_dim=state_dim)

    def _fill_replay_buffer(self, replay_buffer, num_trajectories: int):
        """Build up store of experiences before training begins.

        Args:
            num_trajectories: number of experience tuples to collect before training.
        """
        self._logger.info("Filling replay buffer...")

        state = self._environment.reset_environment(train=True)
        for _ in range(num_trajectories):
            action = random.choice(self._environment.action_space)
            reward, next_state = self._environment.step(action)
            replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                active=self._environment.active,
            )
            if not self._environment.active:
                state = self._environment.reset_environment(train=True)
            else:
                state = next_state

    def _setup_learner(self,
                       config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learner = dqn_learner.DQNLearner(
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
        learner.set_network_branch_gradients(branch_index=0)
        return learner
        # learner = deep_ensemble_learner.DeepEnsembleLearner(
        #     learner_ensemble=learners)
        # return learner

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

        if self._parallelise_ensemble:
            train_fn = self._parallelised_train_episode
        else:
            train_fn = self._serial_train_episode

        (ensemble_rewards, ensemble_step_counts, ensemble_mean_penalties,
         ensemble_mean_penalty_infos,
         ensemble_std_penalty_infos) = train_fn(episode=episode)

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
        for i in range(self._num_learners):
            self._write_scalar(
                tag=(f"{constants.Constants.TRAIN_EPISODE_REWARD}"
                     f"_{constants.Constants.ENSEMBLE_RUNNER}"),
                episode=episode,
                scalar=ensemble_rewards[i],
                df_tag=(f"{constants.Constants.TRAIN_EPISODE_REWARD}"
                        f"_{constants.Constants.ENSEMBLE_RUNNER}_{i}"),
            )
            self._write_scalar(
                tag=(f"{constants.Constants.TRAIN_EPISODE_LENGTH}"
                     f"_{constants.Constants.ENSEMBLE_RUNNER}"),
                episode=episode,
                scalar=ensemble_step_counts[i],
                df_tag=(f"{constants.Constants.TRAIN_EPISODE_LENGTH}"
                        f"_{constants.Constants.ENSEMBLE_RUNNER}_{i}"),
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
        for penalty_info, ensemble_penalty_info in ensemble_mean_penalty_infos.items(
        ):
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
        std_penalty_infos = {}

        for i in range(self._num_learners):
            if self._share_replay_buffer:
                replay_buffer = self._replay_buffer
            else:
                replay_buffer = self._replay_buffers[i]
            (_, episode_reward, episode_count, mean_penalty, mean_penalty_info,
             std_penalty_info) = self._single_train_episode(
                 environment=self._environment,
                 learner=self._learner,
                 branch=i,
                 replay_buffer=replay_buffer,
                 visitation_penalty=self._visitation_penalty,
                 device=self._device,
                 batch_size=self._batch_size,
                 episode=episode,
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

        return (ensemble_episode_rewards, ensemble_episode_step_counts,
                mean_penalties, mean_penalty_infos, std_penalty_infos)

    def _parallelised_train_episode(
        self,
        episode: int,
    ) -> Tuple[float, int]:
        """Perform the train episode for each learner in ensemble in parallel.

        Args:
            episode: index of episode
        """
        raise NotImplementedError
        # processes_arguments = [(
        #     copy.deepcopy(self._environment),
        #     learner,
        #     self._visitation_penalty,
        #     episode,
        # ) for learner in self._learner.ensemble]
        # processes_results = self._pool.starmap(self._single_train_episode,
        #                                        processes_arguments)
        # (
        #     learners,
        #     ensemble_episode_rewards,
        #     ensemble_episode_step_counts,
        #     mean_penalties,
        #     mean_penalty_info,
        # ) = zip(*processes_results)

        # self._learner.ensemble = list(learners)

        # mean_penalty_infos = {}
        # for per_learner_mean_penalty_info in mean_penalty_info:
        #     for info_key, mean_info in per_learner_mean_penalty_info.items():
        #         if info_key not in mean_penalty_infos:
        #             mean_penalty_infos[info_key] = []
        #         mean_penalty_infos[info_key].append(mean_info)

        # return (
        #     ensemble_episode_rewards,
        #     ensemble_episode_step_counts,
        #     mean_penalties,
        #     mean_penalty_infos,
        # )

    @staticmethod
    def _single_train_episode(
        environment: base_environment.BaseEnvironment,
        learner: base_learner.BaseLearner,
        branch: int,
        replay_buffer: replay_buffer.ReplayBuffer,
        visitation_penalty: base_visistation_penalty.BaseVisitationPenalty,
        device: str,
        batch_size: int,
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
        episode_loss = 0
        episode_steps = 0

        penalties = []
        penalty_infos = {}

        # track gradients for relevant branch of Q-networks
        learner.set_network_branch_gradients(branch_index=branch)

        state = environment.reset_environment(train=True)

        while environment.active:

            action = learner.select_behaviour_action(state, branch=branch)
            reward, next_state = environment.step(action)

            penalty, penalty_info = visitation_penalty(
                episode=episode,
                state=torch.from_numpy(state).to(self._device,
                                                 dtype=torch.float),
                action=action,
                next_state=torch.from_numpy(next_state).to(self._device,
                                                           dtype=torch.float))

            penalties.append(penalty)
            for info_key, info in penalty_info.items():
                if info_key not in penalty_infos.keys():
                    penalty_infos[info_key] = []
                penalty_infos[info_key].append(info)

            replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                active=environment.active,
            )

            experience_sample = replay_buffer.sample(batch_size)

            loss, epsilon = learner.step(
                state=torch.from_numpy(experience_sample[0]).to(
                    device=device, dtype=torch.float),
                action=torch.from_numpy(experience_sample[1]).to(
                    device=device, dtype=torch.int),
                reward=torch.from_numpy(experience_sample[2]).to(
                    device=device, dtype=torch.float),
                next_state=torch.from_numpy(experience_sample[3]).to(
                    device=device, dtype=torch.float),
                active=torch.from_numpy(experience_sample[4]).to(
                    device=device, dtype=torch.int),
                visitation_penalty=penalty,
                branch=branch)

            state = next_state
            episode_reward += reward
            episode_loss += loss
            episode_steps += 1

            mean_penalties = np.mean(penalties)
            mean_penalty_info = {
                k: np.mean(v) for k, v in penalty_infos.items()
            }
            std_penalty_info = {k: np.std(v) for k, v in penalty_infos.items()}

        # episode_reward = 0

        # penalties = []
        # penalty_infos = {}

        # state = environment.reset_environment(train=True)

        # while environment.active:
        #     action = learner.select_behaviour_action(state)
        #     reward, next_state = environment.step(action)

        #     penalty, penalty_info = visitation_penalty(episode=episode,
        #                                                state=state,
        #                                                action=action,
        #                                                next_state=next_state)

        #     penalties.append(penalty)
        #     for info_key, info in penalty_info.items():
        #         if info_key not in penalty_infos.keys():
        #             penalty_infos[info_key] = []
        #         penalty_infos[info_key].append(info)

        #     learner.step(
        #         state,
        #         action,
        #         reward,
        #         next_state,
        #         environment.active,
        #         penalty,
        #     )
        #     state = next_state
        #     episode_reward += reward

        # mean_penalties = np.mean(penalties)

        # mean_penalty_info = {k: np.mean(v) for k, v in penalty_infos.items()}

        return (learner, episode_reward, environment.episode_step_count,
                mean_penalties, mean_penalty_info, std_penalty_info)

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
