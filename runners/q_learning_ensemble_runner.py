from experiments import ach_config

from learners.tabular_learners import q_learner
from learners import ensemble_learner
from runners import base_runner

from typing import Tuple

import numpy as np


class EnsembleQLearningRunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig):
        self._num_learners = config.num_learners
        super().__init__(config=config)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learners = [
            self._get_individual_q_learner(config=config)
            for _ in range(self._num_learners)
        ]
        learner = ensemble_learner.EnsembleLearner(learner_ensemble=learners)
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

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        ensemble_episode_rewards = []
        ensemble_episode_step_counts = []

        for learner in self._learner.ensemble:

            episode_reward = 0

            visitation_penalty = self._visitation_penalty(episode)

            self._environment.reset_environment(train=True)
            state = self._environment.agent_position

            while self._environment.active:
                action = learner.select_behaviour_action(state)
                reward, new_state = self._environment.step(action)
                learner.step(
                    state,
                    action,
                    reward,
                    new_state,
                    self._environment.active,
                    visitation_penalty,
                )
                state = new_state
                episode_reward += reward

            ensemble_episode_rewards.append(episode_reward)
            ensemble_episode_step_counts.append(self._environment.episode_step_count)

        return np.mean(episode_reward), np.mean(self._environment.episode_step_count)
