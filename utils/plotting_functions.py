import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth_data(data: List[float], window_width: int) -> List[float]:
    """Calculates moving average of list of values

    Args:
        data: raw, un-smoothed data.
        window_width: width over which to take moving averags

    Returns:
        smoothed_values: averaged data
    """

    def _smooth(single_dataset):
        cumulative_sum = np.cumsum(single_dataset, dtype=float)
        cumulative_sum[window_width:] = (
            cumulative_sum[window_width:] - cumulative_sum[:-window_width]
        )
        smoothed_values = cumulative_sum[window_width - 1 :] / window_width
        return smoothed_values

    if all(isinstance(d, list) for d in data):
        smoothed_data = []
        for dataset in data:
            smoothed_data.append(_smooth(dataset))
    elif all(isinstance(d, float) or isinstance(d, int) for d in data):
        smoothed_data = _smooth(data)

    return smoothed_data


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
    sorted_exp_names = sorted(exp_names, key=lambda x: float(x.split("_")[1]))
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


def plot_value_function(
    grid_size: Tuple[int, int],
    state_action_values: Dict[Tuple[int, int], np.ndarray],
    save_path: str,
    plot_max_values: bool,
    quiver: bool,
) -> None:
    fig = plt.figure()
    if quiver:
        action_arrow_mapping = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        X, Y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
        arrow_x_directions = np.zeros(grid_size)
        arrow_y_directions = np.zeros(grid_size)
        for state, action_values in state_action_values.items():
            action_index = np.argmax(action_values)
            arrow_x_directions[state] = action_arrow_mapping[action_index][0]
            arrow_y_directions[state] = action_arrow_mapping[action_index][1]
        plt.quiver(X, Y, arrow_x_directions, arrow_y_directions, color="red")
    if plot_max_values:
        max_values = np.zeros(grid_size)
        for state, action_values in state_action_values.items():
            max_value = max(action_values)
            max_values[state] = max_value
        plt.imshow(max_values)
        plt.colorbar()
    plt.xlim(-0.5, grid_size[0] - 0.5)
    plt.ylim(-0.5, grid_size[1] - 0.5)
    fig.savefig(save_path, dpi=100)
    plt.close()
