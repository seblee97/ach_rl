import random
from typing import Tuple

import torch

from experiments import ach_config
from learners.deep_learners import dqn_learner
from learners.deep_learners.components import replay_buffer
from runners import base_runner


class DQNRunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

        self._batch_size = config.batch_size
        self._replay_buffer = self._setup_replay_buffer(config=config)

        self._fill_replay_buffer(num_trajectories=config.num_replay_fill_trajectories)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learner = dqn_learner.DQNLearner(
            action_space=self._environment.action_space,
            state_dimensions=tuple(config.encoded_state_dimensions),
            layer_specifications=config.layer_specifications,
            optimiser_type=config.optimiser,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            target_network_update_period=config.target_network_update_period,
            device=config.experiment_device,
        )
        return learner

    def _setup_replay_buffer(self, config: ach_config.AchConfig):
        state_dim = tuple(config.encoded_state_dimensions)
        replay_size = config.replay_buffer_size
        return replay_buffer.ReplayBuffer(replay_size=replay_size, state_dim=state_dim)

    def _fill_replay_buffer(self, num_trajectories: int):

        print("Filling replay buffer...")

        state = self._environment.reset_environment(train=True)
        for _ in range(num_trajectories):
            action = random.choice(self._environment.action_space)
            reward, next_state = self._environment.step(action)
            self._replay_buffer.add(
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

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        visitation_penalty = self._visitation_penalty(episode)

        state = self._environment.reset_environment(train=True)

        while self._environment.active:

            action = self._learner.select_behaviour_action(state)
            reward, next_state = self._environment.step(action)

            self._replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                active=self._environment.active,
            )

            experience_sample = self._replay_buffer.sample(self._batch_size)

            self._learner.step(
                state=torch.from_numpy(experience_sample[0]).to(torch.float),
                action=torch.from_numpy(experience_sample[1]).to(torch.int),
                reward=torch.from_numpy(experience_sample[2]).to(torch.float),
                next_state=torch.from_numpy(experience_sample[3]).to(torch.float),
                active=torch.from_numpy(experience_sample[4]).to(torch.int),
                visitation_penalty=visitation_penalty,
            )

            state = next_state
            episode_reward += reward

        return episode_reward, self._environment.episode_step_count
