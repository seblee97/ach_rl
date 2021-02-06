import os
from typing import Dict
from typing import List

import constants
from experiments import ach_config
from runners import dqn_runner
from runners import q_learning_ensemble_runner
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def single_run(config_path: str, results_folder: str, timestamp: str, run_name: str, changes: List[Dict] = [], seed: int = None):

    config = ach_config.AchConfig(config=config_path, changes=changes)
    
    seed = seed or config.seed
    
    checkpoint_path = experiment_utils.get_checkpoint_path(
        results_folder, timestamp, run_name, str(seed)
    )

    experiment_utils.set_random_seeds(seed)
    config = experiment_utils.set_device(config)
    
    config.amend_property(
        property_name=constants.Constants.SEED, new_property_value=seed
    )
    config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, timestamp)
    config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(
        constants.Constants.LOGFILE_PATH,
        os.path.join(checkpoint_path, "data_logger.csv"),
    )
    config.add_property(constants.Constants.RUN_PATH, MAIN_FILE_PATH)

    os.makedirs(name=checkpoint_path, exist_ok=True)

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
