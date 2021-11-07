"""
This script is for running a set of experiments in parallel.

It can either be called from the command line with the following required arguments:

    - path pointing to the configuration yaml file
    - path pointing to the firectory in which to output results of experiment
    - path pointing to a json file describing changes to be made  to the config.

Else, the individual method 'parallel_run' can be imported for use in other workflows.
"""
import argparse
import os
from typing import List

import constants
import torch.multiprocessing as mp
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
    "--experiment_path",
    metavar="-EP",
    type=str,
    help=("path to dir containing subdirs of checkpoint paths" "for each experiment."),
    required=True,
)

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def parallel_run(config_path: str, checkpoint_paths: List[str]) -> None:
    """Set of experiments run in parallel.

    Args:
        config_path: path to yaml file.
        checkpoint_paths: list of paths to directories to output results.
    """
    processes = []

    for checkpoint_path in checkpoint_paths:
        changes = experiment_utils.json_to_config_changes(
            os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
        )
        process = mp.Process(
            target=single_run.single_run, args=(config_path, checkpoint_path, changes)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def _get_checkpoint_paths_from_experiment_path(experiment_path: str):
    checkpoint_paths = []
    for run_path in os.listdir(experiment_path):
        full_run_path = os.path.join(experiment_path, run_path)
        if os.path.isdir(full_run_path):
            for seed_path in os.listdir(full_run_path):
                checkpoint_path = os.path.join(full_run_path, seed_path)
                if os.path.isdir(checkpoint_path):
                    checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_paths = _get_checkpoint_paths_from_experiment_path(args.experiment_path)
    parallel_run(config_path=args.config_path, checkpoint_paths=checkpoint_paths)
