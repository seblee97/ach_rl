import collections
import copy
import itertools
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import constants
from environments import base_environment


class MultiRoom(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms. Multiple rewards."""

    ACTION_SPACE = [0, 1, 2, 3]

    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN

    DELTAS = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(
        self,
        ascii_map_path: str,
        episode_timeout: Optional[int] = None,
    ):
        self._color_spectrum = "plasma"
        self._colormap = cm.get_cmap(self._color_spectrum)

        (
            self._map,
            self._starting_xy,
            self._key_positions,
            self._door_positions,
            reward_positions,
        ) = self._parse_map(ascii_map_path)

        state_indices = np.where(self._map == 0)
        wall_indices = np.where(self._map == 1)
        self._positional_state_space = list(zip(state_indices[1], state_indices[0]))
        self._key_possession_state_space = list(
            itertools.product([0, 1], repeat=len(self._key_positions))
        )
        self._state_space = [
            i[0] + i[1]
            for i in itertools.product(
                self._positional_state_space, self._key_possession_state_space
            )
        ]

        self._walls = list(zip(wall_indices[1], wall_indices[0]))

        self._rewards = {
            tuple(reward_position): 1.0 for reward_position in reward_positions
        }
        self._total_rewards = sum(self._rewards.values())

        self._episode_timeout = episode_timeout or np.inf
        self._visitation_counts = -1 * copy.deepcopy(self._map)

        self._episode_history: List[List[int]]
        self._agent_position: np.ndarray
        self._rewards_received: List[float]
        self._keys_state: np.ndarray
        self._key_collection_times: Dict[int, int]

    def _parse_map(
        self, map_file_path: str
    ) -> Tuple[np.ndarray, List, List, List, List]:

        start_positions = []
        key_positions = []
        door_positions = []
        reward_positions = []
        map_rows = []

        MAPPING = {
            constants.Constants.WALL_CHARACTER: 1,
            constants.Constants.START_CHARACTER: 0,
            constants.Constants.DOOR_CHARACTER: 0,
            constants.Constants.OPEN_CHARACTER: 0,
            constants.Constants.KEY_CHARACTER: 0,
            constants.Constants.REWARD_CHARACTER: 0,
        }

        with open(map_file_path) as f:
            map_lines = f.read().splitlines()

            # flip indices for x, y referencing
            for i, line in enumerate(map_lines[::-1]):
                map_row = [MAPPING[char] for char in line]
                map_rows.append(map_row)
                if constants.Constants.REWARD_CHARACTER in line:
                    reward_positions.append(
                        (line.index(constants.Constants.REWARD_CHARACTER), i)
                    )
                if constants.Constants.KEY_CHARACTER in line:
                    key_positions.append(
                        (line.index(constants.Constants.KEY_CHARACTER), i)
                    )
                if constants.Constants.START_CHARACTER in line:
                    start_positions.append(
                        (line.index(constants.Constants.START_CHARACTER), i)
                    )
                if constants.Constants.DOOR_CHARACTER in line:
                    door_positions.append(
                        (line.index(constants.Constants.DOOR_CHARACTER), i)
                    )

        assert all(
            len(i) == len(map_rows[0]) for i in map_rows
        ), "ASCII map must specify rectangular grid."

        assert (
            len(start_positions) == 1
        ), "maximally one start position 'S' should be specified in ASCII map."

        multi_room_grid = np.array(map_rows, dtype=float)

        return (
            multi_room_grid,
            start_positions[0],
            key_positions,
            door_positions,
            reward_positions,
        )

    @property
    def starting_xy(self) -> Tuple[int, int]:
        return self._starting_xy

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._state_space

    @property
    def walls(self) -> List[Tuple]:
        return self._walls

    @property
    def episode_step_count(self) -> int:
        return self._episode_step_count

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._visitation_counts

    @property
    def episode_history(self) -> np.ndarray:
        return np.array(self._episode_history)

    def _env_skeleton(
        self, show_rewards: bool = True, show_doors: bool = True, show_keys: bool = True
    ) -> np.ndarray:
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if show_rewards:
            # show reward in red
            for reward in self._rewards.keys():
                skeleton[reward[::-1]] = [1.0, 0.0, 0.0]

        if show_doors:
            # show door in maroon
            for door in self._door_positions:
                skeleton[tuple(door[::-1])] = [0.5, 0.0, 0.0]

        if show_keys:
            # show key in yellow
            for key_index, key_position in enumerate(self._key_positions):
                skeleton[tuple(key_position[::-1])] = [1.0, 1.0, 0.0]

        return skeleton

    def plot_episode_history(self) -> np.ndarray:
        heatmap = self._env_skeleton()

        state_rgb_images = []

        # show agent in silver
        for step, state in enumerate(self._episode_history):
            state_image = copy.deepcopy(heatmap)
            for key_index, key_position in enumerate(self._key_positions):
                if self._keys_state[key_index]:
                    if step > self._key_collection_times[key_index]:
                        heatmap[tuple(key_position[::-1])] = np.ones(3)
            state_image[tuple(state[::-1])] = 0.5 * np.ones(3)
            state_rgb_images.append(state_image)

        return state_rgb_images

    def _average_values_over_key_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        """For certain analyses (e.g. plotting value functions) we want to
        average the values for each position over all non-positional state information--
        in this case the key posessions.

        Args:
            values: full state-action value information

        Returns:
            averaged_values: (positional-state)-action values.
        """
        averaged_values = {}
        for state in self._positional_state_space:
            non_positional_set = [
                values[state + key_state]
                for key_state in self._key_possession_state_space
            ]
            non_positional_mean = np.mean(non_positional_set, axis=0)
            averaged_values[state] = non_positional_mean
        return averaged_values

    def _get_value_combinations(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], Dict[Tuple[int], float]]:
        value_combinations = {}
        for key_state in self._key_possession_state_space:
            value_combination = {}
            for state in self._positional_state_space:
                value_combination[state] = values[state + key_state]
            value_combinations[key_state] = value_combination

        return value_combinations

    def plot_value_function(
        self, values: Dict, plot_max_values: bool, quiver: bool, save_path: str
    ) -> None:
        """
        Plot value function over environment.

        The multiroom states include non-positional state information (key posession),
        so plots can be constructed with this information averaged over. Alternatively
        multiple plots for each combination can be made. For cases with up to 2 keys,
        we construct the combinations. Beyond this only the average is plotted.

        Args:
            values: state-action values.
            plot_max_values: whether or not to plot colormap of values.
            quiver: whether or not to plot arrows with implied best action.
            save_path: path to save graphs.
        """
        if len(self._key_positions) <= 2:
            value_combinations = self._get_value_combinations(values=values)
            fig, axes = plt.subplots(nrows=1 + 2 * len(self._key_positions), ncols=1)
        else:
            value_combinations = []
            fig, axes = plt.subplot(nrows=1, ncols=1)

        fig.subplots_adjust(hspace=0.5)

        averaged_values = self._average_values_over_key_states(values=values)

        self._value_plot(
            fig=fig,
            ax=axes[0],
            values=averaged_values,
            plot_max_values=plot_max_values,
            quiver=quiver,
            subtitle="Positional Average",
        )

        for i, (key_state, value_combination) in enumerate(value_combinations.items()):
            self._value_plot(
                fig=fig,
                ax=axes[i + 1],
                values=value_combination,
                plot_max_values=plot_max_values,
                quiver=quiver,
                subtitle=f"Key State: {key_state}",
            )

        fig.savefig(save_path, dpi=100)
        plt.close()

    def _value_plot(
        self, fig, ax, values: Dict, plot_max_values: bool, quiver: bool, subtitle: str
    ):
        if plot_max_values:
            image = self._get_value_heatmap(values=values)
            im = ax.imshow(image, origin="lower", cmap=self._colormap)
            fig.colorbar(im, ax=ax)
        if quiver:
            map_shape = self._env_skeleton().shape
            X, Y, arrow_x, arrow_y = self._get_quiver_data(
                map_shape=map_shape, values=values
            )
            ax.quiver(
                X,
                Y,
                arrow_x,
                arrow_y,
                color="red",
            )
        ax.title.set_text(subtitle)
        return ax

    def _get_value_heatmap(self, values: Dict) -> np.ndarray:
        environment_map = self._env_skeleton(
            show_rewards=False, show_doors=False, show_keys=False
        )

        all_values = np.concatenate(list(values.values()))
        current_max_value = np.max(all_values)
        current_min_value = np.min(all_values)

        for state, value in values.items():
            max_action_value = max(value)

            # remove alpha from rgba in colormap return
            # normalise value for color mapping
            environment_map[state[::-1]] = self._colormap(
                max_action_value
                - current_min_value / (current_max_value - current_min_value)
            )[:-1]

        return environment_map

    def _get_quiver_data(self, map_shape: Tuple[int], values: Dict) -> Tuple:
        action_arrow_mapping = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        X, Y = np.meshgrid(
            np.arange(map_shape[1]),
            np.arange(map_shape[0]),
            indexing="ij",
        )
        # x, y size of map (discard rgb from environment_map)
        arrow_x_directions = np.zeros(map_shape[:-1][::-1])
        arrow_y_directions = np.zeros(map_shape[:-1][::-1])

        for state, action_values in values.items():
            action_index = np.argmax(action_values)
            action = action_arrow_mapping[action_index]
            arrow_x_directions[state] = action[0]
            arrow_y_directions[state] = action[1]

        return X, Y, arrow_x_directions, arrow_y_directions

    def show_grid(self) -> np.ndarray:
        """Generate 2d array of current state of environment."""
        grid_state = copy.deepcopy(self._map)
        for reward in self._rewards.keys():
            grid_state[reward] = np.inf
        grid_state[tuple(self._agent_position)] = -1
        return grid_state

    def _move_agent(self, delta: np.ndarray) -> None:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._walls
        locked_door = tuple(provisional_new_position) in self._door_positions

        if locked_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._keys_state[door_index]:
                locked_door = False

        if not moving_into_wall and not locked_door:
            self._agent_position = provisional_new_position

        if tuple(self._agent_position) in self._key_positions:
            key_index = self._key_positions.index(tuple(self._agent_position))
            if not self._keys_state[key_index]:
                self._keys_state[key_index] = 1
                self._key_collection_times[key_index] = self._episode_step_count

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        self._move_agent(delta=self.DELTAS[action])

        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1

        reward = self._compute_reward()
        self._active = self._remain_active(reward=reward)

        self._episode_history.append(copy.deepcopy(tuple(self._agent_position)))

        new_state = tuple(self._agent_position) + tuple(self._keys_state)

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
        ):
            reward = self._rewards.get(tuple(self._agent_position))
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0
        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count == self._episode_timeout,
            reward == self._total_rewards,
        ]
        return not any(conditions)

    def reset_environment(self, train: bool) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        self._active = True
        self._episode_step_count = 0
        self._training = train
        self._agent_position = np.array(self._starting_xy)
        self._episode_history = [copy.deepcopy(tuple(self._agent_position))]
        self._rewards_received = []
        self._keys_state = np.zeros(len(self._key_positions), dtype=int)
        self._key_collection_times = {}

        initial_state = tuple(self._agent_position) + tuple(self._keys_state)

        return initial_state
