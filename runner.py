import minigrid
import q_learner

import ach_config

import constants


class Runner:
    def __init__(self, config: ach_config.AchConfig):
        self._environment = self._setup_environment(config=config)
        self._learner = self._setup_learner(config=config)

    def _setup_environment(
        self, config: ach_config.AchConfig
    ):  # TODO: envs should all have set of methods, wrap base class.
        """Initialise environment specified in configuration."""
        if config.environment == constants.Constants.MINIGRID:
            if config.reward_position is not None:
                reward_position = tuple(config.reward_position)
            else:
                reward_position = None
            if config.starting_position is not None:
                agent_starting_position = tuple(config.starting_position)
            else:
                agent_starting_position = None
            environment = minigrid.MiniGrid(
                size=tuple(config.size),
                starting_xy=agent_starting_position,
                reward_xy=reward_position,
                episode_timeout=config.episode_timeout,
            )

        return environment

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        if config.learner == constants.Constants.Q_LEARNER:
            learner = q_learner.TabularQLearner(
                action_space=self._environment.action_space,
                state_space=self._environment.state_space,
                policy=config.policy,
                epsilon=config.epsilon,
                alpha=config.alpha,
                gamma=config.gamma,
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
                action = self._learner.select_action(state)
                reward, new_state = self._environment.step(action)
                self._learner.step(state, action, reward, new_state)
                state = new_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            episode_lengths.append(self._environment.episode_step_count)

        return episode_rewards, episode_lengths
