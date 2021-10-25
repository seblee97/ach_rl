"""
This script is for running a set of experiments in series.

It can either be called from the command line with the following required arguments:

    - path pointing to the configuration yaml file
    - path pointing to the firectory in which to output results of experiment
    - path pointing to a json file describing changes to be made  to the config.

Else, the individual method 'serial_run' can be imported for use in other workflows.
"""
import argparse
import os
from typing import List

import constants
from experiments.run_modes import single_run
from utils import experiment_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_path", metavar="-C", type=str, help="path to yaml file.", required=True
)
parser.add_argument(
    "--config_changes_paths",
    metavar="-CC",
    type=str,
    help="path to json config changes file.",
    required=False,
)
parser.add_argument(
    "--checkpoint_paths",
    metavar="-CP",
    type=str,
    help="path to dir in which to output results.",
    required=True,
)

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def serial_run(config_path: str, checkpoint_paths: List[str]) -> None:
    """Set of experiments run in series.

    Args:
        config_path: path to yaml file.
        checkpoint_paths: list of paths to directories to output results.
    """
    for checkpoint_path in checkpoint_paths:
        changes = experiment_utils.json_to_config_changes(
            os.path.join(checkpoint_path, constants.Constants.CONFIG_CHANGES_JSON)
        )
        single_run.single_run(
            config_path=config_path, checkpoint_path=checkpoint_path, changes=changes
        )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config_changes_path is not None:
        config_changes = experiment_utils.json_to_config_changes(
            args.config_changes_path
        )
    else:
        config_changes = []
    single_run(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        changes=config_changes,
    )
