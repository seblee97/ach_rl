import os
from typing import Tuple

from ach_rl import constants
from ach_rl.experiments import ach_config
from ach_rl.learners.tabular_learners import q_learner
from ach_rl.runners import base_runner


class QLearningRunner(base_runner.BaseRunner):
    """Runner for vanilla Q-learning."""

    def __init__(self, config, unique_id: str):
        super().__init__(config=config, unique_id=unique_id)

    def _get_data_columns(self):
        columns = [
            constants.TRAIN_EPISODE_REWARD,
            constants.TRAIN_EPISODE_LENGTH,
            constants.NEXT_STATE_POLICY_ENTROPY,
            constants.CURRENT_STATE_MEAN_UNCERTAINTY,
            constants.CURRENT_STATE_POLICY_ENTROPY,
            constants.NEXT_STATE_MEAN_UNCERTAINTY,
            constants.CURRENT_STATE_MAX_UNCERTAINTY,
            constants.NEXT_STATE_MAX_UNCERTAINTY,
            constants.CURRENT_STATE_SELECT_UNCERTAINTY,
            constants.MEAN_VISITATION_PENALTY,
        ]
        return columns

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
            split_value_function=config.split_value_function,
        )
        return learner

    def _get_initialisation_strategy(self, config: ach_config.AchConfig):
        if config.initialisation == constants.RANDOM_UNIFORM:
            initialisation_strategy = {
                constants.RANDOM_UNIFORM: {
                    constants.LOWER_BOUND: config.lower_bound,
                    constants.UPPER_BOUND: config.upper_bound,
                }
            }
        elif config.initialisation == constants.RANDOM_NORMAL:
            initialisation_strategy = {
                constants.RANDOM_NORMAL: {
                    constants.MEAN: config.mean,
                    constants.VARIANCE: config.variance,
                }
            }
        else:
            initialisation_strategy == {config.initialisation}
        return initialisation_strategy

    def _pre_episode_log(self, episode: int):
        visualisation_configurations = [
            (constants.MAX_VALUES_PDF, True, False),
            (constants.QUIVER_VALUES_PDF, False, True),
            (constants.QUIVER_MAX_VALUES_PDF, True, True),
        ]
        if self._visualisation_iteration(constants.VALUE_FUNCTION, episode):
            for visualisation_configuration in visualisation_configurations:
                self._logger.info(
                    "Serial value function visualisation: "
                    f"{visualisation_configuration[0]}"
                )
                averaged_values = (
                    self._environment.average_values_over_positional_states(
                        self._learner.state_action_values
                    )
                )
                averaged_max_values = {p: max(v) for p, v in averaged_values.items()}
                self._environment.plot_heatmap_over_env(
                    heatmap=averaged_max_values,
                    save_name=os.path.join(
                        self._visualisations_folder_path,
                        f"{episode}_{visualisation_configuration[0]}",
                    ),
                )
        if self._visualisation_iteration(constants.VISITATION_COUNT_HEATMAP, episode):
            self._environment.plot_value_function(
                values=self._learner.state_visitation_counts,
                save_path=os.path.join(
                    self._visualisations_folder_path,
                    f"{episode}_{constants.VISITATION_COUNT_HEATMAP_PDF}",
                ),
                plot_max_values=True,
                quiver=False,
                over_actions=constants.SELECT,
            )
        if episode != 0:
            if self._visualisation_iteration(constants.INDIVIDUAL_TRAIN_RUN, episode):

                self._environment.visualise_episode_history(
                    save_path=os.path.join(
                        self._checkpoint_path,
                        constants.ROLLOUTS,
                        f"{constants.INDIVIDUAL_TRAIN_RUN}_{episode}.mp4",
                    )
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

        if episode == 0:
            self._environment.render(
                save_path=os.path.join(self._visualisations_folder_path, "map.pdf")
            )

        self._visitation_penalty.state_action_values = [
            self._learner.state_action_values
        ]

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

        self._write_scalar(
            tag=constants.MEAN_VISITATION_PENALTY,
            episode=episode,
            scalar=penalty,
        )
        for penalty_info_name, info in penalty_info.items():
            self._write_scalar(
                tag=constants.MEAN_PENALTY_INFO,
                episode=episode,
                scalar=info,
                df_tag=penalty_info_name,
            )

        return episode_reward, self._environment.episode_step_count

    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner."""
        pass

    def _post_visualisation(self):
        pass
