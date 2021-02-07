import argparse
import multiprocessing
import os
from multiprocessing import Process
from typing import Dict
from typing import List

import constants
from experiments import ach_config
from experiments import run_methods
from runners import dqn_runner
from runners import q_learning_ensemble_runner
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_utils

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", metavar="-C")
parser.add_argument("--seed", metavar="-S", default=0)
parser.add_argument("--results_folder")
parser.add_argument("--config_changes")
parser.add_argument("--timestamp")
parser.add_argument("--run_name")
parser.add_argument("--mode")

args = parser.parse_args()


def parallel_run(config_path: str, seeds: List[int], results_folder: str,
                 timestamp: str, run_name: str, changes: List[Dict]):
    procs = []

    # macOS + python 3.8 change in multiprocessing defaults.
    # workaround: https://github.com/pytest-dev/pytest-flask/issues/104
    multiprocessing.set_start_method("spawn")

    for seed in seeds:
        p = Process(
            target=run_methods.single_run,
            args=(
                config_path,
                results_folder,
                timestamp,
                run_name,
                changes,
                seed,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def serial_run(config_path: str, seeds: List[int], results_folder: str,
               timestamp: str, run_name: str, changes: List[Dict]):
    for seed in seeds:
        run_methods.single_run(config_path=config_path,
                               seed=seed,
                               results_folder=results_folder,
                               timestamp=timestamp,
                               run_name=run_name)


if __name__ == '__main__':
    try:
        seeds = int(args.seed)
    except ValueError:
        seeds = [int(i) for i in args.seed.strip("[]").split(",")]

    config_changes = experiment_utils.json_to_config_changes(
        args.config_changes)

    print(config_changes)

    if isinstance(seeds, int):
        run_methods.single_run(config_path=args.config_path,
                               seed=seeds,
                               results_folder=args.results_folder,
                               timestamp=args.timestamp,
                               run_name=args.run_name,
                               changes=config_changes)
    elif isinstance(seeds, list):
        if args.mode == constants.Constants.PARALLEL:
            parallel_run(config_path=args.config_path,
                         seeds=seeds,
                         results_folder=args.results_folder,
                         timestamp=args.timestamp,
                         run_name=args.run_name,
                         changes=config_changes)
        elif args.mode == constants.Constants.SERIAL:
            serial_run(config_path=args.config_path,
                       seeds=args.seed,
                       results_folder=args.results_folder,
                       timestamp=args.timestamp,
                       run_name=args.run_name,
                       changes=config_changes)

# def single_run(config_path: str, seed: int, results_folder: str, timestamp: str, run_name: str, changes: List[Dict]):

#     config = ach_config.AchConfig(config=config_path, changes=changes)
#     checkpoint_path = experiment_utils.get_checkpoint_path(
#         results_folder, timestamp, run_name, str(seed)
#     )

#     experiment_utils.set_random_seeds(seed)
#     config = experiment_utils.set_device(config)
#     config.amend_property(
#         property_name=constants.Constants.SEED, new_property_value=seed
#     )
#     config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, timestamp)
#     config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
#     config.add_property(
#         constants.Constants.LOGFILE_PATH,
#         os.path.join(checkpoint_path, "data_logger.csv"),
#     )

#     os.makedirs(name=checkpoint_path, exist_ok=True)

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
