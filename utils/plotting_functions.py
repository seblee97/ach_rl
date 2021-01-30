import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants


def smooth_data(data: List[float], window_width: int) -> List[float]:
    """Calculates moving average of list of values

    Args:
        data: raw, un-smoothed data.
        window_width: width over which to take moving averags

    Returns:
        smoothed_values: averaged data
    """

    def _smooth(single_dataset):
        cumulative_sum = np.cumsum(single_dataset, dtype=np.float32)
        cumulative_sum[window_width:] = (
            cumulative_sum[window_width:] - cumulative_sum[:-window_width]
        )
        # explicitly forcing data type to 32 bits avoids floating point errors
        # with constant dat.
        smoothed_values = np.array(
            cumulative_sum[window_width - 1 :] / window_width, dtype=np.float32
        )
        return smoothed_values

    if all(isinstance(d, list) for d in data):
        smoothed_data = []
        for dataset in data:
            smoothed_data.append(_smooth(dataset))
    elif all(
        (isinstance(d, float) or isinstance(d, int) or isinstance(d, np.int64))
        for d in data
    ):
        smoothed_data = _smooth(data)

    return smoothed_data


def plot_all_multi_seed_multi_run(
    folder_path: str, exp_names: List[str], window_width: int
):
    """Expected structure of folder_path is:

    - folder_path
        |_ run_1
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ run_2
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ ...
        |_ ...
        |_ run_N
            |_ seed_0
            |_ seed_1
            |_ ...
            |_ seed_M

    with a file called data_logger.csv in each leaf folder.
    """
    experiment_folders = {
        exp_name: os.path.join(folder_path, exp_name) for exp_name in exp_names
    }

    tag_set = {}

    # arbitrarily select one seed's dataframe for each run to find set of column names
    for exp, exp_path in experiment_folders.items():
        ex_seed = os.listdir(exp_path)[0]
        ex_df = pd.read_csv(os.path.join(exp_path, ex_seed, "data_logger.csv"))
        tag_subset = list(ex_df.columns)
        for tag in tag_subset:
            if tag not in tag_set:
                tag_set[tag] = []
            tag_set[tag].append(exp)

    for tag, relevant_experiments in tag_set.items():
        fig = plt.figure()
        for exp in relevant_experiments:
            attribute_data = []
            seed_folders = [
                f for f in os.listdir(experiment_folders[exp]) if not f.startswith(".")
            ]
            for seed in seed_folders:
                df = pd.read_csv(
                    os.path.join(experiment_folders[exp], seed, "data_logger.csv")
                )
                tag_data = df[tag]
                attribute_data.append(tag_data)
            mean_attribute_data = np.mean(attribute_data, axis=0)
            std_attribute_data = np.std(attribute_data, axis=0)
            smooth_mean_data = smooth_data(
                mean_attribute_data, window_width=window_width
            )
            smooth_std_data = smooth_data(std_attribute_data, window_width=window_width)
            plt.plot(
                range(len(smooth_mean_data)),
                smooth_mean_data,
                label=exp,
            )
            plt.fill_between(
                range(len(smooth_mean_data)),
                smooth_mean_data - smooth_std_data,
                smooth_mean_data + smooth_std_data,
                alpha=0.3,
            )
            plt.legend()
            plt.xlabel(constants.Constants.EPISODE)
            plt.ylabel(tag)

        os.makedirs(os.path.join(folder_path, "figures"), exist_ok=True)
        fig.savefig(
            os.path.join(
                folder_path, "figures", f"{tag}_plot_multi_seed_multi_run.pdf"
            ),
            dpi=100,
        )
        plt.close()


def plot_multi_seed_multi_run(
    folder_path: str,
    tag: str,
    window_width: int,
    xlabel: str,
    ylabel: str,
    threshold_line: Optional[Union[float, int]] = None,
    show_deviations: bool = True,
    save: bool = True,
):
    """Plot average of multiple seeds for multiple different runs from saved csvs.

    Args:
        folder_path: full path to folder containing results
        tag: name of column in csvs to plot.
        window_width: width of window over which to smooth data (after averaging).
        xlabel: x axis label
        ylabel: y axis label
        threshold_line: whether to plot dotted line at some threshold
        show_deviations: whether to plot transluscent variance fill around mean.
        save: whether or not to save plot as image in folder path.
    """
    exp_names = [
        f for f in os.listdir(folder_path) if (f != "figures" and not f.startswith("."))
    ]

    def _is_number(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    sortable_exp_names = []
    non_sortable_exp_names = []
    for exp_name in exp_names:
        split_name = exp_name.split("_")
        try:
            if _is_number(split_name[1]):
                sortable_exp_names.append(exp_name)
            else:
                non_sortable_exp_names.append(exp_name)
        except IndexError:
            non_sortable_exp_names.append(exp_name)

    sorted_exp_names = (
        sorted(sortable_exp_names, key=lambda x: float(x.split("_")[1]))
        + non_sortable_exp_names
    )

    experiment_folders = [os.path.join(folder_path, f) for f in sorted_exp_names]
    fig = plt.figure()
    for i, exp in enumerate(experiment_folders):
        attribute_data = []
        seed_folders = [f for f in os.listdir(exp) if not f.startswith(".")]
        for seed in seed_folders:
            df = pd.read_csv(os.path.join(exp, seed, "data_logger.csv"))
            tag_data = df[tag]
            attribute_data.append(tag_data)
        mean_attribute_data = np.mean(attribute_data, axis=0)
        std_attribute_data = np.std(attribute_data, axis=0)
        smooth_mean_data = smooth_data(mean_attribute_data, window_width=window_width)
        smooth_std_data = smooth_data(std_attribute_data, window_width=window_width)
        plt.plot(
            range(len(smooth_mean_data)), smooth_mean_data, label=sorted_exp_names[i]
        )
        plt.fill_between(
            range(len(smooth_mean_data)),
            smooth_mean_data - smooth_std_data,
            smooth_mean_data + smooth_std_data,
            alpha=0.3,
        )
        plt.legend()
    if threshold_line is not None:
        plt.plot(
            [0, len(smooth_mean_data)],
            [threshold_line, threshold_line],
            linestyle="dashed",
            color="black",
        )
    if save:
        os.makedirs(os.path.join(folder_path, "figures"), exist_ok=True)
        fig.savefig(
            os.path.join(
                folder_path, "figures", f"{tag}_plot_multi_seed_multi_run.pdf"
            ),
            dpi=100,
        )
        plt.close()


def plot_value_function(
    state_action_values: Dict[Tuple[int, int], np.ndarray],
    save_path: str,
    plot_max_values: bool,
    quiver: bool,
    walls: Optional[Tuple] = None,
) -> None:
    fig = plt.figure()
    if quiver:
        action_arrow_mapping = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        X, Y = np.meshgrid(
            np.arange(grid_size[0]), np.arange(grid_size[1]), indexing="ij"
        )
        arrow_x_directions = np.zeros(grid_size)
        arrow_y_directions = np.zeros(grid_size)
        for state, action_values in state_action_values.items():
            action_index = np.argmax(action_values)
            arrow_x_directions[state] = action_arrow_mapping[action_index][0]
            arrow_y_directions[state] = action_arrow_mapping[action_index][1]
        plt.quiver(X, Y, arrow_x_directions, arrow_y_directions, color="red")
    if plot_max_values:
        # flip size so indexing is consistent with axis dimensions
        max_values = np.zeros(grid_size[::-1])
        for state, action_values in state_action_values.items():
            max_value = max(action_values)
            max_values[state[::-1]] = max_value
        plt.imshow(max_values, origin="lower")
        plt.colorbar()
    plt.xlim(-0.5, grid_size[0] - 0.5)
    plt.ylim(-0.5, grid_size[1] - 0.5)
    fig.savefig(save_path, dpi=100)
    plt.close()
