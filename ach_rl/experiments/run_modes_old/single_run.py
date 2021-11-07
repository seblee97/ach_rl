"""
This script is for running a single experiment.

It can either be called from the command line with the following required arguments:

    - path pointing to the configuration yaml file
    - path pointing to the firectory in which to output results of experiment

Optionally, the following arguments can be provided:

    - path pointing to a json file describing changes to be made  to the config.

Else, the individual method 'single_run' can be imported for use in other workflows,
e.g. to use the multiprocessing module.
"""
import argparse
import os
from typing import Dict
from typing import List

import constants
from experiments import ach_config
from runners import dqn_ensemble_independent_runner
from runners import dqn_runner
from runners import q_learning_ensemble_runner
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_logger
from utils import experiment_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_path", metavar="-C", type=str, help="path to yaml file.", required=True
)
parser.add_argument(
    "--config_changes_path",
    metavar="-CC",
    type=str,
    help="path to json config changes file.",
    required=False,
)
parser.add_argument(
    "--checkpoint_path",
    metavar="-CP",
    type=str,
    help="path to dir in which to output results.",
    required=True,
)

MAIN_FILE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def single_run(
    config_path: str, checkpoint_path: str, changes: List[Dict] = []
) -> None:
    """Single experiment run.

    Args:
        config_path: path to yaml file.
        checkpoint_path: path to directory to output results.
        changes: specification of changes to be made to config.
    """

    logger = experiment_logger.get_logger(
        experiment_path=checkpoint_path, name=__name__
    )

    config = ach_config.AchConfig(config=config_path, changes=changes)

    seed = config.seed

    experiment_utils.set_random_seeds(seed)
    config = experiment_utils.set_device(config, logger=logger)

    config.amend_property(property_name=constants.SEED, new_property_value=seed)

    config.add_property(constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(
        constants.LOGFILE_PATH,
        os.path.join(checkpoint_path, "data_logger.csv"),
    )
    config.add_property(constants.RUN_PATH, MAIN_FILE_PATH)

    # os.makedirs(name=checkpoint_path, exist_ok=True)

    if config.type == constants.SARSA_LAMBDA:
        r = sarsa_lambda_runner.SARSALambdaRunner(config=config)
    elif config.type == constants.Q_LEARNING:
        r = q_learning_runner.QLearningRunner(config=config)
    elif config.type == constants.VANILLA_DQN:
        r = dqn_runner.DQNRunner(config=config)
    elif config.type == constants.ENSEMBLE_Q_LEARNING:
        r = q_learning_ensemble_runner.EnsembleQLearningRunner(config=config)
    elif config.type == constants.BOOTSTRAPPED_ENSEMBLE_DQN:
        r = dqn_runner.DQNRunner(config=config)
    elif config.type == constants.INDEPENDENT_ENSEMBLE_DQN:
        r = dqn_ensemble_independent_runner.EnsembleDQNIndependentRunner(config=config)
    else:
        raise ValueError(f"Learner type {type} not recognised.")

    r.train()
    r.post_process()


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
