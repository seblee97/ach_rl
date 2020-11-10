import copy
import itertools
import os
from multiprocessing import Process

import constants
from experiments import ach_config
from experiments import config_template
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_utils
from utils import plotting_functions

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

EXPERIMENT_DESCRIPTION = ""

BASE_CONFIGURATION = ach_config.AchConfig(
    config="config.yaml", template=config_template.ConfigTemplate.base_template
)

VISITATION_PENALTIES = [0, 0.001, 0.002]


def run(experiment_path: str, results_folder: str, timestamp: str):
    penalty_seed_tuples = itertools.product(*[VISITATION_PENALTIES, range(3)])

    procs = []

    for penalty_seed in penalty_seed_tuples:
        p = Process(
            target=single_run,
            args=(
                penalty_seed[1],
                results_folder,
                experiment_path,
                timestamp,
                penalty_seed[0],
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def single_run(
    seed: int,
    results_folder: str,
    experiment_path: str,
    timestamp: str,
    visitation_penalty: float,
):
    config = copy.deepcopy(BASE_CONFIGURATION)
    experiment_utils.set_random_seeds(seed)
    checkpoint_path = experiment_utils.get_checkpoint_path(
        results_folder, timestamp, f"vp_{visitation_penalty}", str(seed)
    )

    config.amend_property(
        property_name=constants.Constants.SEED, new_property_value=seed
    )
    config.amend_property(
        property_name=constants.Constants.VISITATION_PENALTY,
        new_property_value=visitation_penalty,
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

    r.train()
    r.post_process()


def summary_plot(experiment_path: str):

    config = BASE_CONFIGURATION

    if config.num_rewards == 1:
        threshold = sum(config.reward_positions[0])
    else:
        threshold = None

    if constants.Constants.PLOT_EPISODE_LENGTHS in config.plots:
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.TEST_EPISODE_LENGTH,
            window_width=40,
            threshold_line=threshold,
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TEST_EPISODE_LENGTH,
        )
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.TRAIN_EPISODE_LENGTH,
            window_width=40,
            threshold_line=threshold,
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TRAIN_EPISODE_LENGTH,
        )

    if constants.Constants.PLOT_EPISODE_REWARDS in config.plots:
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.TEST_EPISODE_REWARD,
            window_width=40,
            threshold_line=sum(config.reward_magnitudes),
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TEST_EPISODE_REWARD,
        )
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.TRAIN_EPISODE_REWARD,
            window_width=40,
            threshold_line=sum(config.reward_magnitudes),
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TRAIN_EPISODE_REWARD,
        )

    if constants.Constants.NO_REPEAT_TEST_EPISODE_LENGTH in config.plots:
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_LENGTH,
            window_width=40,
            threshold_line=threshold,
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TEST_EPISODE_LENGTH,
        )
    if constants.Constants.NO_REPEAT_TEST_EPISODE_REWARD in config.plots:
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_REWARD,
            window_width=40,
            threshold_line=threshold,
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.TRAIN_EPISODE_LENGTH,
        )


if __name__ == "__main__":
    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)
    run(
        experiment_path=experiment_path,
        results_folder=results_folder,
        timestamp=timestamp,
    )
    summary_plot(experiment_path=experiment_path)
