from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

import constants


class Plotter:
    """Class for plotting scalar data."""

    def __init__(self, save_path: str, logfile_path: str, plot_tags: List[str]):
        self._save_path = save_path
        self._logfile_path = logfile_path
        self._plot_tags = plot_tags

        self._log_df: pd.DataFrame

    def load_data(self) -> None:
        """Read in data logged to path."""
        self._log_df = pd.read_csv(self._logfile_path)

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

    def make_plots(self) -> None:

        graph_layout = (3, 3)
        num_graphs = len(self._plot_tags)
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

        self.fig.savefig(
            os.path.join(self._save_path, constants.Constants.PLOT_PDF), dpi=100
        )

    def _plot_scalar(
        self,
        row: int,
        col: int,
        data_tag: str,
    ):
        data = self._log_df[data_tag]

        fig_sub = self.fig.add_subplot(self.spec[row, col])

        fig_sub.plot(range(len(data)), data)

        # labelling
        fig_sub.set_xlabel(constants.Constants.EPISODE)
        fig_sub.set_ylabel(data_tag)
        fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which="major", linestyle="-", linewidth="0.5", color="red", alpha=0.2
        )
        fig_sub.grid(
            which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.4
        )
