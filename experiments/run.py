import argparse
import os

import constants
from experiments import ach_config
from run_modes import cluster_run
from run_modes import parallel_run
from run_modes import serial_run
from run_modes import single_run
from run_modes import utils
from runners import q_learning_ensemble_runner
from runners import q_learning_runner

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--mode", metavar="-M", required=True, help="run experiment.")
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="config.yaml",
    help="path to base configuration file.",
)
parser.add_argument(
    "--seeds", metavar="-S", default="[0]", help="list of seeds to run."
)
parser.add_argument("--config_changes", metavar="-CC", default="config_changes.py")


if __name__ == "__main__":

    args = parser.parse_args()

    results_folder = os.path.join(MAIN_FILE_PATH, constants.RESULTS)

    config_class_name = "AChConfig"
    config_module_name = "ach_config"
    config_module_path = os.path.join(MAIN_FILE_PATH, "ach_config.py")
    config_class = ach_config.AchConfig
    config = config_class(args.config_path)

    if config.type == constants.Q_LEARNING:
        runner_class = q_learning_runner.QLearningRunner
        runner_class_name = "QLearningRunner"
        runner_module_name = "q_learning_runner"
        runner_module_path = os.path.join(MAIN_FILE_PATH, "q_learning_runner.py")
    elif config.type == constants.ENSEMBLE_Q_LEARNING:
        runner_class = q_learning_ensemble_runner.EnsembleQLearningRunner
        runner_class_name = "EnsembleQLearningRunner"
        runner_module_name = "q_learning_ensemble_runner"
        runner_module_path = os.path.join(
            MAIN_FILE_PATH, "q_learning_ensemble_runner.py"
        )

    if args.mode == constants.SINGLE:

        single_checkpoint_path = utils.setup_experiment(
            mode="single", results_folder=results_folder, config_path=args.config_path
        )

        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=args.config_path,
            checkpoint_path=single_checkpoint_path,
            run_methods=["train", "post_process"],
        )

    elif args.mode in [constants.PARALLEL, constants.SERIAL, constants.CLUSTER]:

        seeds = utils.process_seed_arguments(args.seeds)

        checkpoint_paths = utils.setup_experiment(
            mode="multi",
            results_folder=results_folder,
            config_path=args.config_path,
            config_changes_path=args.config_changes,
            seeds=seeds,
        )

        if args.mode == constants.PARALLEL:

            parallel_run.parallel_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=["train", "post_process"],
                stochastic_packages=["numpy", "torch", "random"],
            )

        elif args.mode == constants.SERIAL:

            serial_run.serial_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=["train", "post_process"],
                stochastic_packages=["numpy", "torch", "random"],
            )

        elif args.mode == constants.CLUSTER:

            cluster_run.cluster_run(
                runner_class_name=runner_class_name,
                runner_module_name=runner_module_name,
                runner_module_path=runner_module_path,
                config_class_name=config_class_name,
                config_module_name=config_module_name,
                config_module_path=config_module_path,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=["train", "post_process"],
                stochastic_packages=["numpy", "torch", "random"],
                env_name="rama",
                scheduler="univa",
            )

    else:
        raise ValueError(f"run mode {args.mode} not recognised.")
