from experiments import ach_config

from learners.tabular_learners import q_learner
from runners import base_runner

from typing import Tuple

import constants

import os


class QLearningRunner(base_runner.BaseRunner):
    """Runner for vanilla Q-learning."""

    def __init__(self, config: ach_config.AchConfig):
        super().__init__(config=config)

    def _setup_learner(self, config: ach_config.AchConfig):  # TODO: similar to envs
        """Initialise learner specified in configuration."""
        initialisation_strategy = self._get_initialisation_strategy(config)
        learner = q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            behaviour=config.behaviour,
            target=config.target,
            initialisation_strategy=initialisation_strategy,
            epsilon=self._epsilon_function,
            learning_rate=config.learning_rate,
            gamma=config.discount_factor,
            split_value_function=config.split_value_function
        )
        return learner

    def _get_initialisation_strategy(self, config: ach_config.AchConfig):
        if config.initialisation == constants.Constants.RANDOM_UNIFORM:
            initialisation_strategy = {
                constants.Constants.RANDOM_UNIFORM: {
                    constants.Constants.LOWER_BOUND: config.lower_bound,
                    constants.Constants.UPPER_BOUND: config.upper_bound
                }
            }
        elif config.initialisation == constants.Constants.RANDOM_NORMAL:
            initialisation_strategy = {
                constants.Constants.RANDOM_NORMAL: {
                    constants.Constants.MEAN: config.mean,
                    constants.Constants.VARIANCE: config.variance
                }
            }
        else:
            initialisation_strategy == {config.initialisation}
        return initialisation_strategy

    def _pre_episode_log(self, episode: int):
        visualisation_configurations = [
            (constants.Constants.MAX_VALUES_PDF, True, False),
            (constants.Constants.QUIVER_VALUES_PDF, False, True),
            (constants.Constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(constants.Constants.VALUE_FUNCTION, episode):
            for visualisation_configuration in visualisation_configurations:
                self._logger.info(
                    "Serial value function visualisation: "
                    f"{visualisation_configuration[0]}"
                )
                self._environment.plot_value_function(
                    values=self._learner.state_action_values,
                    save_path=os.path.join(
                        self._visualisations_folder_path,
                        f"{episode}_{visualisation_configuration[0]}",
                    ),
                    plot_max_values=visualisation_configuration[1],
                    quiver=visualisation_configuration[2],
                    over_actions=constants.Constants.MAX,
                )
        if self._visualisation_iteration(constants.Constants.VISITATION_COUNT_HEATMAP, episode):
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

        if self._scalar_log_iteration(constants.Constants.CYCLE_COUNT, episode):
            num_cycles = cycle_counter.evaluate_loops_on_value_function(
                size=self._grid_size,
                state_action_values=self._learner.state_action_values,
            )
            self._logger.write_scalar(
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

        state = self._environment.reset_environment(train=True)

        self._visitation_penalty.state_action_values = [self._learner.state_action_values]

        while self._environment.active:
            action = self._learner.select_behaviour_action(state)
            reward, new_state = self._environment.step(action)

            penalty, penalty_info = self._visitation_penalty(
                episode=episode, state=state, action=action, next_state=new_state
            )

            self._learner.step(
                state,
                action,
                reward,
                new_state,
                self._environment.active,
                penalty,
            )
            state = new_state
            episode_reward += reward

        return episode_reward, self._environment.episode_step_count

    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner."""
        pass

    def _post_visualisation(self):
        pass