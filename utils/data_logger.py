import os
from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments import ach_config
from utils import animator


class DataLogger:
    """Class for logging experimental data.

    Data can be stored in a csv. # TODO: add TensorBoard event file.
    Experiment monitoring can also be sent to a log file.
    """

    def __init__(self, config: ach_config.AchConfig):
        self._checkpoint_path = config.checkpoint_path
        self._logfile_path = config.logfile_path
        self._df_columns = []
        self._logger_data = {}
        self._plot_origin = config.plot_origin
        self._animation_library = config.animation_library
        self._file_format = config.animation_file_format

    def write_scalar(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dictionary.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.
        """
        if tag not in self._logger_data:
            self._logger_data[tag] = {}
            self._df_columns.append(tag)

        self._logger_data[tag][step] = scalar

    def checkpoint(self) -> None:
        """Construct dataframe from data and merge with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        assert (
            list(self._logger_data.keys()) == self._df_columns
        ), "Incorrect dataframe columns for merging"

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        pd.DataFrame(self._logger_data).to_csv(
            self._logfile_path, mode="a", header=header, index=False
        )

        # reset logger in memory to empty.
        self._df_columns = []
        self._logger_data = {}

    def plot_array_data(
        self, name: str, data: Union[List[np.ndarray], np.ndarray]
    ) -> None:
        """Plot array data to image file.

        Args:
            name: filename for save.
            data: data to save.
        """
        full_path = os.path.join(self._checkpoint_path, name)

        if isinstance(data, list):
            animator.animate(
                images=data,
                file_name=full_path,
                plot_origin=self._plot_origin,
                library=self._animation_library,
                file_format=self._file_format,
            )
        elif isinstance(data, np.ndarray):
            fig = plt.figure()
            plt.imshow(data, plot_origin=self._plot_origin)
            plt.colorbar()
            fig.savefig(fname=full_path)
            plt.close(fig)

    def write_array_data(self, name: str, data: np.ndarray) -> None:
        """Write array data to np save file.

        Args:
            name: filename for save.
            data: data to save.
        """
        full_path = os.path.join(self._checkpoint_path, name)
        np.save(file=full_path, arr=data)
