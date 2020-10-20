import ach_config

from learners import backward_sarsa_lambda
from runners import base_runner


class BackwardSARSARunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

        self._discount_factor = config.discount_factor

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learner = backward_sarsa_lambda.BackwardSARSALambda(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            initialisation_strategy=config.initialisation,
            epsilon=config.epsilon,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            visitation_penalty=config.visitation_penalty,
            trace_lambda=config.trace_lambda,
        )
        return learner

    def train(self, num_episodes: int):
        state = self._environment.agent_position
        episode_rewards = []
        episode_lengths = []
        for _ in range(num_episodes):
            self._environment.reset_environment()
            episode_reward = 0
            state = self._environment.agent_position
            while self._environment.active:
                action = self._learner.select_behaviour_action(state)
                reward, new_state = self._environment.step(action)
                next_action = self._learner.select_behaviour_action(new_state)
                self._learner.step(state, action, reward, new_state, next_action)
                state = new_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            episode_lengths.append(self._environment.episode_step_count)

        return episode_rewards, episode_lengths
