import os
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    show_deviations: bool = True,
    save: bool = True,
):
    """Plot average of multiple seeds for multiple different runs from saved csvs.

    Args:
        folder_path: full path to folder containing results
        tag: name of column in csvs to plot.
        window_width: width of window over which to smooth data (after averaging).
        show_deviations: whether to plot transluscent variance fill around mean.
        save: whether or not to save plot as image in folder path.
    """
    exp_names = [
        f for f in os.listdir(folder_path) if (f != "figures" and not f.startswith("."))
    ]
    experiment_folders = [os.path.join(folder_path, f) for f in exp_names]
    fig = plt.figure()
    for i, exp in enumerate(experiment_folders):
        attribute_data = []
        for seed in os.listdir(exp):
            df = pd.read_csv(os.path.join(exp, seed, "data_logger.csv"))
            tag_data = df[tag]
            attribute_data.append(tag_data)
        mean_attribute_data = np.mean(attribute_data, axis=0)
        std_attribute_data = np.std(attribute_data, axis=0)
        smooth_mean_data = smooth_data(mean_attribute_data, window_width=window_width)
        smooth_std_data = smooth_data(std_attribute_data, window_width=window_width)
        plt.plot(range(len(smooth_mean_data)), smooth_mean_data, label=exp_names[i])
        plt.fill_between(
            range(len(smooth_mean_data)),
            smooth_mean_data - smooth_std_data,
            smooth_mean_data + smooth_std_data,
            alpha=0.3,
        )
        plt.legend()
    if save:
        os.makedirs(os.path.join(folder_path, "figures"), exist_ok=True)
        fig.savefig(
            os.path.join(
                folder_path, "figures", f"{tag}_plot_multi_seed_multi_run.png"
            ),
            dpi=100,
        )