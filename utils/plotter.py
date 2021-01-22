import math
import os
from typing import List
from typing import Tuple
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants
from utils import plotting_functions


class Plotter:
    """Class for plotting scalar data."""

    def __init__(self, save_folder: str, logfile_path: str, smoothing: int):
        self._save_folder = save_folder
        self._logfile_path = logfile_path
        self._smoothing = smoothing

        self._plot_tags: List[str]

        self._log_df: pd.DataFrame
        self._scaling: int

    def load_data(self) -> None:
        """Read in data logged to path."""
        self._log_df = pd.read_csv(self._logfile_path)
        self._plot_tags = list(self._log_df.columns)
        self._scaling = len(self._log_df)

    @staticmethod
    def get_figure_skeleton(
        height: Union[int, float],
        width: Union[int, float],
        num_columns: int,
        num_rows: int,
    ) -> Tuple:

        fig = plt.figure(
            constrained_layout=False, figsize=(num_columns * width, num_rows * height)
        )

        heights = [height for _ in range(num_rows)]
        widths = [width for _ in range(num_columns)]

        spec = gridspec.GridSpec(
            nrows=num_rows,
            ncols=num_columns,
            width_ratios=widths,
            height_ratios=heights,
        )

        return fig, spec

    def plot_learning_curves(self) -> None:

        num_graphs = len(self._plot_tags)

        default_layout = (
            math.ceil(np.sqrt(num_graphs)),
            math.ceil(np.sqrt(num_graphs)),
        )
        graph_layout = constants.Constants.GRAPH_LAYOUTS.get(num_graphs, default_layout)

        num_rows = graph_layout[0]
        num_columns = graph_layout[1]

        self.fig, self.spec = self.get_figure_skeleton(
            height=4, width=5, num_columns=num_columns, num_rows=num_rows
        )

        for row in range(num_rows):
            for col in range(num_columns):

                graph_index = (row) * num_columns + col

                if graph_index < num_graphs:

                    print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))
                    self._plot_scalar(
                        row=row, col=col, data_tag=self._plot_tags[graph_index]
                    )

        save_path = os.path.join(self._save_folder, constants.Constants.PLOT_PDF)
        self.fig.savefig(save_path, dpi=100)
        plt.close()

    def _plot_scalar(
        self,
        row: int,
        col: int,
        data_tag: str,
    ):
        fig_sub = self.fig.add_subplot(self.spec[row, col])

        # labelling
        fig_sub.set_xlabel(constants.Constants.EPISODE)
        fig_sub.set_ylabel(data_tag)

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which="major", linestyle="-", linewidth="0.5", color="red", alpha=0.2
        )
        fig_sub.grid(
            which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.4
        )

        # plot data
        if data_tag in self._log_df.columns:
            sub_fig_tags = [data_tag]
            sub_fig_data = [self._log_df[data_tag].dropna()]
        else:
            sub_fig_tags = [tag for tag in self._log_df.columns if data_tag in tag]
            sub_fig_data = [self._log_df[tag].dropna() for tag in sub_fig_tags]

        smoothed_data = [
            plotting_functions.smooth_data(
                data=data.to_numpy(), window_width=min(len(data), self._smoothing)
            )
            for data in sub_fig_data
        ]

        x_data = [
            (self._scaling / len(data)) * np.arange(len(data)) for data in smoothed_data
        ]

        for x, y, label in zip(x_data, smoothed_data, sub_fig_tags):
            fig_sub.plot(x, y, label=label)

        fig_sub.legend()
