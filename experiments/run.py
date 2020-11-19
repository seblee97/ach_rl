import argparse
import copy
import itertools
import os
from multiprocessing import Process
from typing import Any
from typing import List
from typing import Tuple

import constants
from experiments import ach_config
from experiments import config_template
from runners import q_learning_runner
from runners import sarsa_lambda_runner
from utils import experiment_utils
from utils import plotting_functions

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
    "--config_changes",
    metavar="-CC",
    default=(
        "[(vp_schedule, 0) | (vp_schedule, 0.001) | "
        "(vp_schedule, 0.002) | (vp_schedule, 0.003) | "
        "(vp_schedule, 0.004) | (vp_schedule, 0.005)]"
    ),
)
parser.add_argument("--seeds", metavar="-S", default=range(5))

args = parser.parse_args()


def parse_config_changes(config_changes: str) -> List[Tuple[str, Any]]:
    """Parser takes strings. Split this string into config changes.

    Format of string must be [X | Y | ...] where each X, Y etc. are tuples
    with following format: (str, Any), where requirements for type of second element
    in tuple is set by string in first.

    Args:
        config_changes: str encoding changes to be made to config for each run

    Returns:
        decoded_config_changes: interpretable config changes.
    """
    import pdb

    pdb.set_trace()
    split_by_element = config_changes.strip("[").strip("]").split(" | ")

    decoded_config_changes = []

    for element in split_by_element:
        key, change = element.strip("(").strip(")").split(", ")
        if key == constants.Constants.VP_SCHEDULE:
            # change must by float
            decoded_change = float(change)
        elif key == constants.Constants.INITIALISATION:
            # change must either by str or float/int
            if change.isdigit():
                decoded_change = float(change)
            else:
                decoded_change = change
        decoded_config_changes.append((key, decoded_change))

    return decoded_config_changes


def parallel_run(
    base_configuration: ach_config.AchConfig,
    config_changes: List[Tuple[str, Any]],
    seeds: List[int],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    config_change_seed_tuples = itertools.product(*[config_changes, seeds])

    procs = []

    for config_change_tuple in config_change_seed_tuples:
        p = Process(
            target=single_run,
            args=(
                base_configuration,
                config_change_tuple[1],
                results_folder,
                experiment_path,
                timestamp,
                config_change_tuple[0],
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def serial_run(
    base_configuration: ach_config.AchConfig,
    config_changes: List[Tuple[str, Any]],
    seeds: List[int],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    for change in config_changes:
        print(f"{change[0]}: {change[1]}")
        for seed in seeds:
            print(f"Seed: {seed}")
            single_run(
                base_configuration=base_configuration,
                seed=seed,
                results_folder=results_folder,
                experiment_path=experiment_path,
                timestamp=timestamp,
                config_change=change,
            )


def single_run(
    base_configuration: ach_config.AchConfig,
    seed: int,
    results_folder: str,
    experiment_path: str,
    timestamp: str,
    config_change: Tuple[str, Any],
):
    config = copy.deepcopy(base_configuration)
    experiment_utils.set_random_seeds(seed)
    checkpoint_path = experiment_utils.get_checkpoint_path(
        results_folder, timestamp, f"{config_change[0]}_{config_change[1]}", str(seed)
    )

    config.amend_property(
        property_name=constants.Constants.SEED, new_property_value=seed
    )

    config.amend_property(
        property_name=config_change[0],
        new_property_value=config_change[1],
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
            ylabel=constants.Constants.CYCLE_COUNT,
        )


if __name__ == "__main__":

    base_configuration = ach_config.AchConfig(
        config=args.config, template=config_template.ConfigTemplate.base_template
    )

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    config_changes = parse_config_changes(args.config_changes)
    # penalties: str = args.penalties
    # parsed_penalties = [float(i) for i in penalties.strip("[").strip("]").split(",")]

    if args.mode == constants.Constants.PARALLEL:
        parallel_run(
            base_configuration=base_configuration,
            config_changes=config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    elif args.mode == constants.Constants.SERIAL:
        serial_run(
            base_configuration=base_configuration,
            config_changes=config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    summary_plot(config=base_configuration, experiment_path=experiment_path)
