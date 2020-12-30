from experiments import ach_config

from learners.tabular_learners import q_learner
from runners import base_runner

from typing import Tuple

import constants


class QLearningRunner(base_runner.BaseRunner):
    """Runner for vanilla Q-learning."""

    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        learner = q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target=config.target,
            initialisation_strategy=config.initialisation,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
        )
        return learner

    def _pre_episode_log(self, episode: int):
        if constants.Constants.VALUE_FUNCTION in self._plot_logging:
            self._plotter.plot_value_function(
                state_action_values=self._learner.state_action_values,
                extra_tag=f"{episode}_",
                walls=self._environment.walls,
            )
        if episode != 0:
            if constants.Constants.INDIVIDUAL_TRAIN_RUN in self._plot_logging:
                self._logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TRAIN_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

        if constants.Constants.CYCLE_COUNT in self._scalar_logging:
            num_cycles = cycle_counter.evaluate_loops_on_value_function(
                size=self._grid_size,
                state_action_values=self._learner.state_action_values,
            )
            self._logger.write_scalar_df(
                tag=constants.Constants.CYCLE_COUNT,
                step=episode,
                scalar=num_cycles,
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

        visitation_penalty = self._visitation_penalty(episode)

        state = self._environment.reset_environment(train=True)

        while self._environment.active:
            action = self._learner.select_behaviour_action(state)
            reward, new_state = self._environment.step(action)
            self._learner.step(
                state,
                action,
                reward,
                new_state,
                self._environment.active,
                visitation_penalty,
            )
            state = new_state
            episode_reward += reward

        return episode_reward, self._environment.episode_step_count
