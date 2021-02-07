import argparse
import collections.abc
import copy
import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
from multiprocessing import Process
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import constants
import torch
import yaml
from experiments import ach_config
from experiments import run_methods
from experiments.config_changes import ConfigChange
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
    help=
    "run in 'parallel' or 'serial', or 'single' (no changes), or 'cluster'",
)
parser.add_argument("--config_path", metavar="-C", default="config.yaml")
parser.add_argument("--seeds", metavar="-S", default=list(range(3)))
parser.add_argument("--config_changes",
                    metavar="-CC",
                    default=ConfigChange.config_changes)

args = parser.parse_args()


def parallel_run(config_path: str, results_folder: str, timestamp: str,
                 config_changes: Dict[str,
                                      List[Dict]], seeds: List[int]) -> None:
    """Run experiments in parallel.

    Args:
        base_configuration: config object.
        seeds: list of seeds to run experiment over.
        config_changes: changes to make to base configuration
            for each separate experiment.
        experiment_path: path to experiment.
        results_folder: path to results.
        timestamp: experiment timestamp.
    """
    procs = []

    # macOS + python 3.8 change in multiprocessing defaults.
    # workaround: https://github.com/pytest-dev/pytest-flask/issues/104
    multiprocessing.set_start_method("spawn")

    for run_name, changes in config_changes.items():
        logger.info(f"{run_name}")
        for seed in seeds:
            logger.info(f"Seed: {seed}")
            p = Process(
                target=run_methods.single_run,
                args=(config_path, results_folder, timestamp, run_name,
                      changes, seed),
            )
            p.start()
            procs.append(p)

    for p in procs:
        p.join()


def serial_run(config_path: str, results_folder: str, timestamp: str,
               config_changes: Dict[str,
                                    List[Dict]], seeds: List[int]) -> None:
    """Run experiments in serial.

    Args:
        base_configuration: config object.
        seeds: list of seeds to run experiment over.
        config_changes: changes to make to base configuration
            for each separate experiment.
        experiment_path: path to experiment.
        results_folder: path to results.
        timestamp: experiment timestamp.
    """
    for run_name, changes in config_changes.items():
        logger.info(f"{run_name}")
        for seed in seeds:
            logger.info(f"Seed: {seed}")
            run_methods.single_run(config_path=config_path,
                                   results_folder=results_folder,
                                   timestamp=timestamp,
                                   run_name=run_name,
                                   changes=changes,
                                   seed=seed)


def cluster_run(config_path: str, results_folder: str, timestamp: str,
                config_changes: Dict[str,
                                     List[Dict]], seeds: List[int]) -> None:
    for run_name, changes in config_changes.items():
        logger.info(f"Run name: {run_name}, seed: {seeds}")

        # if seeds a list, do x; if seeds an int, do y; if seeds nothing, do z;
        # no tmpdir, save job script and config changes json in experiment_path
        checkpoint_path = os.path.join(results_folder, timestamp, run_name)
        config_changes_path = os.path.join(checkpoint_path,
                                           "config_changes.json")
        job_script_path = os.path.join(checkpoint_path, "job_script")
        error_path = os.path.join(checkpoint_path,
                                  constants.Constants.ERROR_FILE_NAME)
        output_path = os.path.join(checkpoint_path,
                                   constants.Constants.OUTPUT_FILE_NAME)

        os.makedirs(name=checkpoint_path, exist_ok=True)

        experiment_utils.config_changes_to_json(config_changes=changes,
                                                json_path=config_changes_path)

        run_command = (
            f"python cluster_run.py --config_path {config_path} "
            f"--seed '{seeds}' --config_changes {config_changes_path} "
            f"--results_folder {results_folder} --timestamp {timestamp} "
            f"--run_name {run_name} --mode parallel")

        cluster_methods.create_job_script(run_command=run_command,
                                          save_path=job_script_path,
                                          num_cpus=8,
                                          conda_env_name="ach",
                                          memory=60,
                                          error_path=error_path,
                                          output_path=output_path)

        subprocess.call(run_command, shell=True)
        # subprocess.call(f"qsub {job_script_path}", shell=True)


if __name__ == "__main__":

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    os.makedirs(name=experiment_path, exist_ok=True)

    # logger at root of experiment i.e. not individual runs or seeds
    logger = experiment_logger.get_logger(experiment_path=experiment_path,
                                          name=__name__)

    if args.mode == constants.Constants.SINGLE:
        single_run(config_path=args.config_path,
                   results_folder=results_folder,
                   timestamp=timestamp,
                   run_name=constants.Constants.SINGLE)
    else:
        experiment_utils.save_config_changes(
            config_changes=args.config_changes,
            file_name=os.path.join(
                experiment_path,
                f"{constants.Constants.CONFIG_CHANGES}.json",
            ),
        )
        if args.mode == constants.Constants.PARALLEL:
            parallel_run(config_path=args.config_path,
                         results_folder=results_folder,
                         timestamp=timestamp,
                         config_changes=args.config_changes,
                         seeds=args.seeds)
        elif args.mode == constants.Constants.CLUSTER:
            cluster_run(config_path=args.config_path,
                        results_folder=results_folder,
                        timestamp=timestamp,
                        config_changes=args.config_changes,
                        seeds=args.seeds)
        elif args.mode == constants.Constants.SERIAL:
            serial_run(config_path=args.config_path,
                       results_folder=results_folder,
                       timestamp=timestamp,
                       config_changes=args.config_changes,
                       seeds=args.seeds)

# def summary_plot(
#     smoothng: int, exp_names: List[str], experiment_path: str
# ):
#     """Plot summary of experiment."""
#     plotting_functions.plot_all_multi_seed_multi_run(
#         folder_path=experiment_path, exp_names=exp_names, window_width=smoothng
#     )

# def single_run(
#     config_path: str,
#     run_name: str,
#     seed: int,
#     results_folder: str,
#     experiment_path: str,
#     timestamp: str,
#     config_change: List[Tuple[str, Any, bool]],
# ):
#     """Run single experiment.

#     Args:
#         base_configuration: config object.
#         seeds: list of seeds to run experiment over.
#         run_name: name of single experiment.
#         experiment_path: path to experiment.
#         results_folder: path to results.
#         timestamp: experiment timestamp.
#         config_change: change to make to base configuration.
#     """
#     config = copy.deepcopy(base_configuration)

#     experiment_utils.set_random_seeds(seed)
#     checkpoint_path = experiment_utils.get_checkpoint_path(
#         results_folder, timestamp, run_name, str(seed)
#     )

#     os.makedirs(name=checkpoint_path, exist_ok=True)

#     config.amend_property(
#         property_name=constants.Constants.SEED, new_property_value=seed
#     )

#     for change in config_change:
#         try:
#             config.amend_property(
#                 property_name=change[0],
#                 new_property_value=change[1],
#             )
#         except AssertionError:
#             if change[2]:
#                 config.add_property(
#                     property_name=change[0],
#                     property_value=change[1],
#                 )

#     config = experiment_utils.set_device(config)

#     config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, timestamp)
#     config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
#     config.add_property(
#         constants.Constants.LOGFILE_PATH,
#         os.path.join(checkpoint_path, "data_logger.csv"),
#     )

#     if config.type == constants.Constants.SARSA_LAMBDA:
#         r = sarsa_lambda_runner.SARSALambdaRunner(config=config)
#     elif config.type == constants.Constants.Q_LEARNING:
#         r = q_learning_runner.QLearningRunner(config=config)
#     elif config.type == constants.Constants.DQN:
#         r = dqn_runner.DQNRunner(config=config)
#     elif config.type == constants.Constants.ENSEMBLE_Q_LEARNING:
#         r = q_learning_ensemble_runner.EnsembleQLearningRunner(config=config)
#     else:
#         raise ValueError(f"Learner type {type} not recognised.")

#     r.train()
#     r.post_process()

# base_configuration = ach_config.AchConfig(config=args.config_path)

# base_configuration.add_property(constants.Constants.RUN_PATH, MAIN_FILE_PATH)

# timestamp = experiment_utils.get_experiment_timestamp()
# results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
# experiment_path = os.path.join(results_folder, timestamp)

# os.makedirs(name=experiment_path, exist_ok=True)

# if args.mode != constants.Constants.SINGLE:
#     save_config_changes(
#         config_changes=args.config_changes,
#         file_name=os.path.join(
#             experiment_path,
#             f"{constants.Constants.CONFIG_CHANGES}.json",
#         ),
#     )

# if args.mode == constants.Constants.PARALLEL:
#     if base_configuration.use_gpu:
#         _distribute_over_gpus(args.config_changes)

#     parallel_run(
#         base_configuration=base_configuration,
#         config_changes=args.config_changes,
#         seeds=args.seeds,
#         experiment_path=experiment_path,
#         results_folder=results_folder,
#         timestamp=timestamp,
#     )
#     summary_plot(
#         config=base_configuration,
#         experiment_path=experiment_path,
#         exp_names=list(args.config_changes.keys()),
#     )
# if args.mode == constants.Constants.CLUSTER:
#     cluster_run(
#         configuration=args.config,
#         config_changes=args.config_changes,
#         seeds=args.seeds,
#         experiment_path=experiment_path,
#         results_folder=results_folder,
#         timestamp=timestamp,
#     )
#     summary_plot(
#         config=base_configuration,
#         experiment_path=experiment_path,
#         exp_names=list(args.config_changes.keys()),
#     )
# elif args.mode == constants.Constants.SERIAL:
#     serial_run(
#         base_configuration=base_configuration,
#         config_changes=args.config_changes,
#         seeds=args.seeds,
#         experiment_path=experiment_path,
#         results_folder=results_folder,
#         timestamp=timestamp,
#     )
#     summary_plot(
#         config=base_configuration,
#         experiment_path=experiment_path,
#         exp_names=list(args.config_changes.keys()),
#     )
# elif args.mode == constants.Constants.SINGLE:
#     single_run(
#         base_configuration=base_configuration,
#         run_name=constants.Constants.SINGLE,
#         seed=base_configuration.seed,
#         results_folder=results_folder,
#         experiment_path=experiment_path,
#         timestamp=timestamp,
#         config_change=[],
#     )
