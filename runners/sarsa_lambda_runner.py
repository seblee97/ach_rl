from experiments import ach_config

from learners import sarsa_lambda
from runners import base_runner

from typing import Tuple


class SARSALambdaRunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

        self._discount_factor = config.discount_factor

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learner = sarsa_lambda.TabularSARSALambda(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            initialisation_strategy=config.initialisation,
            behaviour=config.behaviour,
            epsilon=config.epsilon,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            visitation_penalty=config.visitation_penalty,
            trace_lambda=config.trace_lambda,
        )
        return learner

    def _train_episode(self) -> Tuple[float, int]:
        """Perform single training loop.

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        self._environment.reset_environment(train=True)
        state = self._environment.agent_position
        action = self._learner.select_behaviour_action(state)

        while self._environment.active:
            reward, next_state = self._environment.step(action)
            next_action = self._learner.select_behaviour_action(next_state)
            self._learner.step(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            episode_reward += reward

        return episode_reward, self._environment.episode_step_count
