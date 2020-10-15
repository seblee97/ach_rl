import argparse
import datetime
import time
import os

import ach_config
import constants

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args() -> argparse.Namespace:
    """Get args from command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-config",
        type=str,
        help="path to configuration file for student teacher experiment",
        default="base_config.yaml",
    )

    args = parser.parse_args()

    return args


def get_config_object(args: argparse.Namespace) -> ach_config.AchConfig:
    """Read config path into configuration object.

    Args:
        args: argparser namespace object with arguments specifying configuration.

    Returns:
        configuration: configuration object.
    """
    full_config_path = os.path.join(MAIN_FILE_PATH, args.config)
    configuration = ach_config.AchConfig(config=full_config_path)
    return configuration


def set_random_seeds(seed: int) -> None:
    # import packages with non-deterministic behaviour
    import random
    import numpy as np

    # set random seeds for these packages
    random.seed(seed)
    np.random.seed(seed)


def set_experiment_metadata(config: ach_config.AchConfig) -> ach_config.AchConfig:
    """Set metadata (time, date, names) etc. for experiment."""
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = config.experiment_name or ""

    results_folder_base = constants.Constants.RESULTS

    checkpoint_path = (
        f"{MAIN_FILE_PATH}/{results_folder_base}/{exp_timestamp}/{experiment_name}/"
    )

    config.set_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
    config.set_property(constants.Constants.EXPERIMENT_TIMESTAMP, exp_timestamp)

    return config


def run(config: ach_config.AchConfig):
    


if __name__ == "___main__":
    args = get_args()
    config = get_config_object(args)
    config = set_experiment_metadata(config=config)
    set_random_seeds(seed=config.seed)
    run(config)
