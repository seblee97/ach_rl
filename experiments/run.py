import argparse
import collections.abc
import copy
import importlib
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Process
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import constants
import torch
import yaml
from experiments import ach_config
from experiments import cluster_run
from experiments import parallel_run
from experiments import run_methods
from experiments import serial_run
from experiments import single_run
from runners import dqn_runner
from runners import q_learning_ensemble_runner
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import cluster_methods
from utils import experiment_logger
from utils import experiment_utils
from utils import plotting_functions

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    metavar="-M",
    default="parallel",
    help="run in 'parallel' or 'serial', or 'single' (no changes), or 'cluster'",
)
parser.add_argument("--config_path", metavar="-C", default="config.yaml")
parser.add_argument("--seeds", metavar="-S", default=list(range(1)))
parser.add_argument("--config_changes",
                    metavar="-CC",
                    default="config_changes.py")
parser.add_argument("--num_cpus", metavar="-NC", default=8)
parser.add_argument("--memory", metavar="-MEM", default=64)
parser.add_argument("--cluster_mode", metavar="-CM", default="individual")

args = parser.parse_args()

# def parallel_run(config_path: str, results_folder: str, timestamp: str,
#                  config_changes: Dict[str,
#                                       List[Dict]], seeds: List[int]) -> None:
#     """Run experiments in parallel.

#     Args:
#         base_configuration: config object.
#         seeds: list of seeds to run experiment over.
#         config_changes: changes to make to base configuration
#             for each separate experiment.
#         experiment_path: path to experiment.
#         results_folder: path to results.
#         timestamp: experiment timestamp.
#     """
#     procs = []

#     # macOS + python 3.8 change in multiprocessing defaults.
#     # workaround: https://github.com/pytest-dev/pytest-flask/issues/104
#     multiprocessing.set_start_method("spawn")

#     for run_name, changes in config_changes.items():
#         logger.info(f"{run_name}")
#         for seed in seeds:
#             logger.info(f"Seed: {seed}")
#             p = Process(
#                 target=run_methods.single_run,
#                 args=(config_path, results_folder, timestamp, run_name, changes,
#                       seed),
#             )
#             p.start()
#             procs.append(p)

#     for p in procs:
# #         p.join()

# def serial_run(config_path: str, results_folder: str, timestamp: str,
#                config_changes: Dict[str, List[Dict]], seeds: List[int]) -> None:
#     """Run experiments in serial.

#     Args:
#         base_configuration: config object.
#         seeds: list of seeds to run experiment over.
#         config_changes: changes to make to base configuration
#             for each separate experiment.
#         experiment_path: path to experiment.
#         results_folder: path to results.
#         timestamp: experiment timestamp.
#     """
#     for run_name, changes in config_changes.items():
#         logger.info(f"{run_name}")
#         for seed in seeds:
#             logger.info(f"Seed: {seed}")
#             run_methods.single_run(config_path=config_path,
#                                    results_folder=results_folder,
#                                    timestamp=timestamp,
#                                    run_name=run_name,
#                                    changes=changes,
# seed=seed)


def _submit_job(config_path: str,
                results_folder: str,
                run_name: str,
                timestamp: str,
                num_cpus: int,
                memory: int,
                changes: List[Dict],
                seeds: Union[int, List[int]],
                cluster_mode: str = ""):
    checkpoint_path = os.path.join(results_folder, timestamp, run_name)
    config_changes_path = os.path.join(checkpoint_path, "config_changes.json")
    job_script_path = os.path.join(checkpoint_path, "job_script")
    error_path = os.path.join(checkpoint_path,
                              constants.Constants.ERROR_FILE_NAME)
    output_path = os.path.join(checkpoint_path,
                               constants.Constants.OUTPUT_FILE_NAME)

    os.makedirs(name=checkpoint_path, exist_ok=True)

    experiment_utils.config_changes_to_json(config_changes=changes,
                                            json_path=config_changes_path)

    run_command = (f"python cluster_run.py --config_path {config_path} "
                   f"--seed '{seeds}' --config_changes {config_changes_path} "
                   f"--results_folder {results_folder} --timestamp {timestamp} "
                   f"--run_name {run_name}")

    if cluster_mode:
        run_command = f"{run_command} --mode {cluster_mode}"

    cluster_methods.create_job_script(run_command=run_command,
                                      save_path=job_script_path,
                                      num_cpus=num_cpus,
                                      conda_env_name="ach",
                                      memory=memory,
                                      error_path=error_path,
                                      output_path=output_path)

    # subprocess.call(run_command, shell=True)
    subprocess.call(f"qsub {job_script_path}", shell=True)


# def cluster_run(config_path: str, results_folder: str, timestamp: str,
#                 config_changes: Dict[str, List[Dict]], seeds: List[int],
#                 num_cpus: int, memory: int, cluster_mode: str) -> None:

#     for run_name, changes in config_changes.items():
#         if cluster_mode == constants.Constants.INDIVIDUAL:
#             for seed in seeds:
#                 logger.info(f"Run name: {run_name}, seed: {seed}")
#                 checkpoint_path = os.path.join(results_folder, timestamp,
#                                                run_name, str(seed))
#                 logger.info(f"Creating checkpoint path at {checkpoint_path}...")
#                 os.makedirs(name=checkpoint_path, exist_ok=True)

#                 logger.info(f"Run name: {run_name}, seed: {seed}")
#                 _submit_job(config_path=config_path,
#                             results_folder=results_folder,
#                             run_name=run_name,
#                             timestamp=timestamp,
#                             num_cpus=num_cpus,
#                             memory=memory,
#                             changes=changes,
#                             seeds=seed)
#         else:
#             logger.info(f"Run name: {run_name}, seed: {seeds}")
#             _submit_job(config_path=config_path,
#                         results_folder=results_folder,
#                         run_name=run_name,
#                         timestamp=timestamp,
#                         num_cpus=num_cpus,
#                         memory=memory,
#                         changes=changes,
#                         seeds=seeds,
#                         cluster_mode=cluster_mode)


def cluster_array_run(config_path: str, results_folder: str, timestamp: str,
                      config_changes: Dict[str, List[Dict]], seeds: List[int],
                      num_cpus: int, memory: int) -> None:

    config_changes_dir = os.path.join(
        results_folder, timestamp, constants.Constants.CONFIG_CHANGES_SYM_PATH)
    # error_files_dir = os.path.join(results_folder, timestamp,
    #                                constants.Constants.ERROR_FILES_SYM_PATH)
    # output_files_dir = os.path.join(results_folder, timestamp,
    # constants.Constants.OUTPUT_FILES_SYM_PATH)
    checkpoint_paths_dir = os.path.join(
        results_folder, timestamp, constants.Constants.CHECKPOINTS_SYM_PATH)

    os.makedirs(config_changes_dir, exist_ok=True)
    # os.makedirs(error_files_dir, exist_ok=True)
    # os.makedirs(output_files_dir, exist_ok=True)
    os.makedirs(checkpoint_paths_dir, exist_ok=True)

    job_script_path = os.path.join(results_folder, timestamp, "job_script")

    num_configurations = len(config_changes)
    num_seeds = len(seeds)

    for i, (run_name, changes) in enumerate(config_changes.items()):
        for j, seed in enumerate(seeds):

            array_job_index = i * num_seeds + j + 1

            logger.info(
                f"Run name: {run_name}, seed: {seed}, array_job_index: {array_job_index}"
            )

            checkpoint_path = os.path.join(results_folder, timestamp, run_name,
                                           str(seed))
            checkpoint_sym_path = os.path.join(checkpoint_paths_dir,
                                               str(array_job_index))
            os.makedirs(name=checkpoint_path, exist_ok=True)

            config_changes_path = os.path.join(checkpoint_path,
                                               "config_changes.json")
            config_changes_sym_path = os.path.join(
                config_changes_dir, f"config_changes_{array_job_index}.json")

            # error_path = os.path.join(checkpoint_path,
            #                           constants.Constants.ERROR_FILE_NAME)
            # with open(error_path, "w") as empty:
            #     pass
            # error_sym_path = os.path.join(error_files_dir,
            #                               f"error_{array_job_index}.txt")
            # output_path = os.path.join(checkpoint_path,
            #                            constants.Constants.OUTPUT_FILE_NAME)
            # with open(output_path, "w") as empty:
            #     pass
            # output_sym_path = os.path.join(output_files_dir,
            # f"output_{array_job_index}.txt")

            os.symlink(config_changes_path, config_changes_sym_path)
            os.symlink(checkpoint_path,
                       checkpoint_sym_path,
                       target_is_directory=True)
            # os.symlink(error_path, error_sym_path)
            # os.symlink(output_path, output_sym_path)

            # add seed to config changes
            changes.append({constants.Constants.SEED: seed})

            experiment_utils.config_changes_to_json(
                config_changes=changes, json_path=config_changes_path)

    pbs_array_index = "$PBS_ARRAY_INDEX/"
    # error_pbs_array_index = "error_$PBS_ARRAY_INDEX.txt"
    # output_pbs_array_index = "output_$PBS_ARRAY_INDEX.txt"
    config_changes_pbs_array_index = "config_changes_$PBS_ARRAY_INDEX.json"
    # error_path_string = f'"{os.path.join(error_files_dir, error_pbs_array_index)}"'
    # output_path_string = f'"{os.path.join(output_files_dir, output_pbs_array_index)}"'

    run_command = (
        f'python cluster_array_run.py --config_path {config_path} '
        f'--config_changes "{os.path.join(config_changes_dir, config_changes_pbs_array_index)}" '
        f'--checkpoint_path "{os.path.join(checkpoint_paths_dir, pbs_array_index)}" '
    )

    cluster_methods.create_job_script(
        run_command=run_command,
        save_path=job_script_path,
        num_cpus=num_cpus,
        conda_env_name="ach",
        memory=memory,
        error_path="",
        output_path="",
        checkpoint_path=os.path.join(checkpoint_paths_dir, pbs_array_index),
        array_job_length=num_configurations * num_seeds)

    subprocess.call(f"qsub {job_script_path}", shell=True)


def _create_experiment_directories(experiment_path: str, run_names: List[str],
                                   seeds: List[int]) -> None:
    for run_name in run_names:
        for seed in seeds:
            checkpoint_path = os.path.join(experiment_path, run_name, str(seed))
            os.makedirs(name=checkpoint_path, exist_ok=True)


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
            seeds=args.seeds)
        if args.mode == constants.Constants.PARALLEL:
            parallel_run.parallel_run(config_path=args.config_path,
                                      checkpoint_paths=checkpoint_paths)
        elif args.mode == constants.Constants.SERIAL:
            serial_run.serial_run(config_path=args.config_path,
                                  checkpoint_paths=checkpoint_paths)
        elif args.mode == constants.Constants.CLUSTER:
            cluster_run.cluster_run(experiment_path=experiment_path,
                                    config_path=args.config_path,
                                    num_cpus=len(checkpoint_paths),
                                    memory_per_node=args.memory)

        elif args.mode == constants.Constants.CLUSTER_ARRAY:
            cluster_array_run(config_path=args.config_path,
                              results_folder=results_folder,
                              timestamp=timestamp,
                              config_changes=args.config_changes,
                              seeds=args.seeds,
                              num_cpus=args.num_cpus,
                              memory=args.memory)
