import argparse
import copy
import os
import multiprocessing
from multiprocessing import Process
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import constants
from experiments import ach_config
from experiments.config_changes import ConfigChange
from runners import dqn_runner
from runners import q_learning_ensemble_runner
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_utils
from utils import plotting_functions

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    metavar="-M",
    default="parallel",
    help="run in 'parallel' or 'serial', or 'single' (no changes)",
)
parser.add_argument("--config", metavar="-C", default="config.yaml")
parser.add_argument("--seeds", metavar="-S", default=range(1))
parser.add_argument(
    "--config_changes", metavar="-CC", default=ConfigChange.config_changes
)

args = parser.parse_args()


def parallel_run(
    base_configuration: ach_config.AchConfig,
    seeds: List[int],
    config_changes: Dict[str, List[Tuple[str, Any]]],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
) -> None:
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
    multiprocessing.set_start_method("fork")

    for run_name, changes in config_changes.items():
        for seed in seeds:
            p = Process(
                target=single_run,
                args=(
                    base_configuration,
                    run_name,
                    seed,
                    results_folder,
                    experiment_path,
                    timestamp,
                    changes,
                ),
            )
            p.start()
            procs.append(p)

    for p in procs:
        p.join()


def serial_run(
    base_configuration: ach_config.AchConfig,
    seeds: List[int],
    config_changes: Dict[str, List[Tuple[str, Any]]],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
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
        print(f"{run_name}")
        for seed in seeds:
            print(f"Seed: {seed}")
            single_run(
                base_configuration=base_configuration,
                seed=seed,
                results_folder=results_folder,
                experiment_path=experiment_path,
                timestamp=timestamp,
                run_name=run_name,
                config_change=changes,
            )


def single_run(
    base_configuration: ach_config.AchConfig,
    run_name: str,
    seed: int,
    results_folder: str,
    experiment_path: str,
    timestamp: str,
    config_change: List[Tuple[str, Any]],
):
    """Run single experiment.

    Args:
        base_configuration: config object.
        seeds: list of seeds to run experiment over.
        run_name: name of single experiment.
        experiment_path: path to experiment.
        results_folder: path to results.
        timestamp: experiment timestamp.
        config_change: change to make to base configuration.
    """
    config = copy.deepcopy(base_configuration)

    config = experiment_utils.set_device(config)

    experiment_utils.set_random_seeds(seed)
    checkpoint_path = experiment_utils.get_checkpoint_path(
        results_folder, timestamp, run_name, str(seed)
    )

    config.amend_property(
        property_name=constants.Constants.SEED, new_property_value=seed
    )

    for change in config_change:
        config.amend_property(
            property_name=change[0],
            new_property_value=change[1],
        )

    config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, timestamp)
    config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(
        constants.Constants.LOGFILE_PATH,
        os.path.join(checkpoint_path, "data_logger.csv"),
    )

    if config.type == constants.Constants.SARSA_LAMBDA:
        r = sarsa_lambda_runner.SARSALambdaRunner(config=config)
    elif config.type == constants.Constants.Q_LEARNING:
        r = q_learning_runner.QLearningRunner(config=config)
    elif config.type == constants.Constants.DQN:
        r = dqn_runner.DQNRunner(config=config)
    elif config.type == constants.Constants.ENSEMBLE_Q_LEARNING:
        r = q_learning_ensemble_runner.EnsembleQLearningRunner(config=config)
    else:
        raise ValueError(f"Learner type {type} not recognised.")

    r.train()
    r.post_process()


def summary_plot(config: ach_config.AchConfig, experiment_path: str):
    """Plot summary of experiment."""
    for tag in config.scalars:
        if isinstance(tag[0], str):
            plotting_functions.plot_multi_seed_multi_run(
                folder_path=experiment_path,
                tag=tag,
                window_width=config.smoothing,
                xlabel=constants.Constants.EPISODE,
                ylabel=tag,
            )


if __name__ == "__main__":

    base_configuration = ach_config.AchConfig(config=args.config)

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    if args.mode == constants.Constants.PARALLEL:
        parallel_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
        summary_plot(config=base_configuration, experiment_path=experiment_path)
    elif args.mode == constants.Constants.SERIAL:
        serial_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
        summary_plot(config=base_configuration, experiment_path=experiment_path)
    elif args.mode == constants.Constants.SINGLE:
        single_run(
            base_configuration=base_configuration,
            run_name=constants.Constants.SINGLE,
            seed=base_configuration.seed,
            results_folder=results_folder,
            experiment_path=experiment_path,
            timestamp=timestamp,
            config_change=[],
        )
