import argparse
import importlib
import os
from typing import Dict
from typing import List
from typing import Union

import constants
from experiments.run_modes import cluster_array_run
from experiments.run_modes import cluster_run
from experiments.run_modes import parallel_run
from experiments.run_modes import serial_run
from experiments.run_modes import single_run
from utils import experiment_logger
from utils import experiment_utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--mode",
                    metavar="-M",
                    required=True,
                    help="run experiment.")
parser.add_argument("--config_path",
                    metavar="-C",
                    default="config.yaml",
                    help="path to base configuration file.")
parser.add_argument("--seeds",
                    metavar="-S",
                    default="[0]",
                    help="list of seeds to run.")
parser.add_argument("--config_changes",
                    metavar="-CC",
                    default="config_changes.py")
parser.add_argument(
    "--memory",
    metavar="-MEM",
    default=64,
    help="amount of memory to assign to each node when working on cluster")


def _process_seed_arguments(seeds: Union[str, List[int]]):
    if isinstance(seeds, list):
        return seeds
    elif isinstance(seeds, str):
        try:
            seeds = [int(seeds.strip("[").strip("]"))]
        except ValueError:
            seeds = [int(s) for s in seeds.strip("[").strip("]").split(",")]
    return seeds


def _organise_config_changes_and_checkpoint_dirs(experiment_path,
                                                 config_changes: Dict[
                                                     str, List[Dict]],
                                                 seeds: List[int]) -> List[str]:
    checkpoint_paths = []
    for run_name, changes in config_changes.items():
        for seed in seeds:
            checkpoint_path = os.path.join(experiment_path, run_name, str(seed))
            os.makedirs(name=checkpoint_path, exist_ok=True)
            changes.append({constants.Constants.SEED: seed})
            experiment_utils.config_changes_to_json(
                config_changes=changes,
                json_path=os.path.join(checkpoint_path,
                                       constants.Constants.CONFIG_CHANGES_JSON))
            checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    os.makedirs(name=experiment_path, exist_ok=True)

    # logger at root of experiment i.e. not individual runs or seeds
    logger = experiment_logger.get_logger(experiment_path=experiment_path,
                                          name=__name__)

    if args.mode == constants.Constants.SINGLE:
        single_checkpoint_path = os.path.join(experiment_path,
                                              constants.Constants.SINGLE)
        os.makedirs(name=single_checkpoint_path, exist_ok=True)
        single_run.single_run(config_path=args.config_path,
                              checkpoint_path=single_checkpoint_path)
    else:
        seeds = _process_seed_arguments(args.seeds)
        logger.info(f"Seeds after processing: {seeds}")
        config_changes = importlib.import_module(
            name=f"{args.config_changes.strip('.py')}").CONFIG_CHANGES
        experiment_utils.config_changes_to_json(
            config_changes=config_changes,
            json_path=os.path.join(
                experiment_path,
                f"all_{constants.Constants.CONFIG_CHANGES_JSON}"))
        checkpoint_paths = _organise_config_changes_and_checkpoint_dirs(
            experiment_path=experiment_path,
            config_changes=config_changes,
            seeds=seeds)
        if args.mode == constants.Constants.PARALLEL:
            parallel_run.parallel_run(config_path=args.config_path,
                                      checkpoint_paths=checkpoint_paths)
        elif args.mode == constants.Constants.SERIAL:
            serial_run.serial_run(config_path=args.config_path,
                                  checkpoint_paths=checkpoint_paths)
        elif args.mode == constants.Constants.CLUSTER:
            if len(checkpoint_paths) <= 8:
                num_cpus = len(checkpoint_paths)
            elif len(checkpoint_paths) <= 32:
                num_cpus = 32
            elif len(checkpoint_paths) <= 64:
                num_cpus = 64
            else:
                raise ValueError(
                    f"{len(checkpoint_paths)} is too many run combinations.")
            cluster_run.cluster_run(experiment_path=experiment_path,
                                    config_path=args.config_path,
                                    num_cpus=num_cpus,
                                    memory_per_node=args.memory)
        elif args.mode == constants.Constants.CLUSTER_ARRAY:
            raise NotADirectoryError
        else:
            raise ValueError(f"Mode '{args.mode}' not recognised.")
