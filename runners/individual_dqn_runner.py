import random
from typing import Tuple

import constants
import torch
from experiments import ach_config
from learners.deep_learners import dqn_learner
from learners.deep_learners.components import replay_buffer
from runners import base_runner


class DQNRunner(base_runner.BaseRunner):
    """Runner for DQN model."""

    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

        self._device = config.experiment_device

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
            momentum=config.gradient_momentum,
            eps=config.min_squared_gradient,
            gamma=config.discount_factor,
            target_network_update_period=config.target_network_update_period,
            device=config.experiment_device,
            gradient_clipping=config.gradient_clipping,
        )
        return learner

    def _setup_replay_buffer(
        self, config: ach_config.AchConfig
    ) -> replay_buffer.ReplayBuffer:
        """Instantiate replay buffer object to store experiences."""
        state_dim = tuple(config.encoded_state_dimensions)
        replay_size = config.replay_buffer_size
        return replay_buffer.ReplayBuffer(replay_size=replay_size, state_dim=state_dim)

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

    def _pre_episode_log(self, episode: int):
        if episode != 0:
            if self._visualisation_iteration(constants.INDIVIDUAL_TRAIN_RUN, episode):
                self._logger.plot_array_data(
                    name=f"{constants.INDIVIDUAL_TRAIN_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0

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

            loss, epsilon = self._learner.step(
                state=torch.from_numpy(experience_sample[0]).to(
                    device=self._device, dtype=torch.float
                ),
                action=torch.from_numpy(experience_sample[1]).to(
                    device=self._device, dtype=torch.int
                ),
                reward=torch.from_numpy(experience_sample[2]).to(
                    device=self._device, dtype=torch.float
                ),
                next_state=torch.from_numpy(experience_sample[3]).to(
                    device=self._device, dtype=torch.float
                ),
                active=torch.from_numpy(experience_sample[4]).to(
                    device=self._device, dtype=torch.int
                ),
                visitation_penalty=visitation_penalty,
            )

            state = next_state
            episode_reward += reward
            episode_loss += loss
            episode_steps += 1

        if self._scalar_log_iteration(constants.AVERAGE_ACTION_VALUE, episode):
            average_action_value = self._compute_average_action_value()
            self._logger.write_scalar(
                tag=constants.AVERAGE_ACTION_VALUE,
                step=episode,
                scalar=average_action_value,
            )
        self._write_scalar(
            tag=constants.LOSS,
            episode=episode,
            scalar=episode_loss / episode_steps,
        )
        self._write_scalar(tag=constants.EPSILON, episode=episode, scalar=epsilon)

        return episode_reward, self._environment.episode_step_count

    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner."""
        pass

    def _compute_average_action_value(self) -> float:
        """Compute average value of action implied by learned state-action values.

        Returns:
            average_value: average_value.
        """
        self._learner.eval()

        with torch.no_grad():
            states = torch.from_numpy(self._replay_buffer.states)[:256].to(
                dtype=torch.float, device=self._device
            )
            action_values_over_states = self._learner.q_network(states)
            average_value = torch.mean(action_values_over_states).item()

        self._learner.train()

        return average_value
