import argparse
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

from typing import List

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser()

# arg_parser.add_argument(name="--experiment_description", default="")
parser.add_argument(
    "--mode",
    metavar="-M",
    default="parallel",
    help="run in 'parallel' or 'serial'",
)
parser.add_argument("--config", metavar="-C", default="config.yaml")
parser.add_argument(
    "--penalties", metavar="-P", default=[0, 0.001, 0.002, 0.003, 0.004, 0.005]
)
parser.add_argument("--seeds", metavar="-S", default=range(5))

args = parser.parse_args()


def parallel_run(
    base_configuration: ach_config.AchConfig,
    visitation_penalties: List[float],
    seeds: List[int],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    penalty_seed_tuples = itertools.product(*[visitation_penalties, seeds])

    procs = []

    for penalty_seed in penalty_seed_tuples:
        p = Process(
            target=single_run,
            args=(
                base_configuration,
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


def serial_run(
    base_configuration: ach_config.AchConfig,
    visitation_penalties: List[float],
    seeds: List[int],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    for visitation_penalty in visitation_penalties:
        print(f"Visitation Penalty: {visitation_penalty}")
        for seed in seeds:
            print(f"Seed: {seed}")
            single_run(
                base_configuration=base_configuration,
                seed=seed,
                results_folder=results_folder,
                experiment_path=experiment_path,
                timestamp=timestamp,
                visitation_penalty=visitation_penalty,
            )


def single_run(
    base_configuration: ach_config.AchConfig,
    seed: int,
    results_folder: str,
    experiment_path: str,
    timestamp: str,
    visitation_penalty: float,
):
    config = copy.deepcopy(base_configuration)
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


def summary_plot(config: ach_config.AchConfig, experiment_path: str):

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

    if constants.Constants.CYCLE_COUNT in config.plots:
        plotting_functions.plot_multi_seed_multi_run(
            folder_path=experiment_path,
            tag=constants.Constants.CYCLE_COUNT,
            window_width=40,
            xlabel=constants.Constants.EPISODE,
            ylabel=constants.Constants.CYCE_COUNT,
        )


if __name__ == "__main__":

    base_configuration = ach_config.AchConfig(
        config=args.config, template=config_template.ConfigTemplate.base_template
    )

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    if args.mode == constants.Constants.PARALLEL:
        parallel_run(
            base_configuration=base_configuration,
            visitation_penalties=args.penalties,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    elif args.mode == constants.Constants.SERIAL:
        serial_run(
            base_configuration=base_configuration,
            visitation_penalties=args.penalties,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    summary_plot(config=base_configuration, experiment_path=experiment_path)
