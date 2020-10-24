import os
from typing import List

import pandas as pd
from experiments import ach_config

from utils import decorators


class Logger:
    """Class for logging experimental data.

    Data can be stored in a csv. # TODO: add TensorBoard event file.
    Experiment monitoring can also be sent to a log file.
    """

    def __init__(self, config: ach_config.AchConfig):
        self._logfile_path = config.logfile_path
        self._df_columns = self._get_df_columns(config)
        self._logger_df = pd.DataFrame(columns=self._df_columns)

    def _get_df_columns(self, config: ach_config.AchConfig) -> List[str]:
        return config.columns

    def write_scalar_df(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dataframe.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.

        Raises:
            AssertionError: if tag provided is not previously defined as a column.
        """
        assert tag in self._df_columns, (
            f"Scalar tag {tag} not in list of recognised columns"
            f"for DataFrame provided: {self._df_columns}"
        )
        self._logger_df.at[step, tag] = scalar

    @decorators.timer
    def checkpoint_df(self) -> None:
        """Merge dataframe with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        assert all(
            self._logger_df.columns == self._df_columns
        ), "Incorrect dataframe columns for merging"

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        self._logger_df.to_csv(self._logfile_path, mode="a", header=header, index=False)

        # reset logger in memory to empty.
        self._logger_df = pd.DataFrame(columns=self._df_columns)
