import argparse
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

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", metavar="-C")
parser.add_argument("--config_changes")
parser.add_argument("--checkpoint_path")

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def cluster_array_run(
    config_path: str,
    results_folder: str,
    timestamp: str,
    config_changes: Dict[str, List[Dict]],
    seeds: List[int],
    num_cpus: int,
    memory: int,
) -> None:

    config_changes_dir = os.path.join(
        results_folder, timestamp, constants.CONFIG_CHANGES_SYM_PATH
    )
    # error_files_dir = os.path.join(results_folder, timestamp,
    #                                constants.ERROR_FILES_SYM_PATH)
    # output_files_dir = os.path.join(results_folder, timestamp,
    # constants.OUTPUT_FILES_SYM_PATH)
    checkpoint_paths_dir = os.path.join(
        results_folder, timestamp, constants.CHECKPOINTS_SYM_PATH
    )

    os.makedirs(config_changes_dir, exist_ok=True)
    # os.makedirs(error_files_dir, exist_ok=True)
    # os.makedirs(output_files_dir, exist_ok=True)
    os.makedirs(checkpoint_paths_dir, exist_ok=True)

    job_script_path = os.path.join(results_folder, timestamp, "job_script")

    num_configurations = len(config_changes)
    num_seeds = len(seeds)

    for i, (run_name, changes) in enumerate(config_changes.items()):
        for j, seed in enumerate(seeds):

            array_job_index = i * num_seeds + j + 1

            logger.info(
                f"Run name: {run_name}, seed: {seed}, array_job_index: {array_job_index}"
            )

            checkpoint_path = os.path.join(
                results_folder, timestamp, run_name, str(seed)
            )
            checkpoint_sym_path = os.path.join(
                checkpoint_paths_dir, str(array_job_index)
            )
            os.makedirs(name=checkpoint_path, exist_ok=True)

            config_changes_path = os.path.join(checkpoint_path, "config_changes.json")
            config_changes_sym_path = os.path.join(
                config_changes_dir, f"config_changes_{array_job_index}.json"
            )

            # error_path = os.path.join(checkpoint_path,
            #                           constants.ERROR_FILE_NAME)
            # with open(error_path, "w") as empty:
            #     pass
            # error_sym_path = os.path.join(error_files_dir,
            #                               f"error_{array_job_index}.txt")
            # output_path = os.path.join(checkpoint_path,
            #                            constants.OUTPUT_FILE_NAME)
            # with open(output_path, "w") as empty:
            #     pass
            # output_sym_path = os.path.join(output_files_dir,
            # f"output_{array_job_index}.txt")

            os.symlink(config_changes_path, config_changes_sym_path)
            os.symlink(checkpoint_path, checkpoint_sym_path, target_is_directory=True)
            # os.symlink(error_path, error_sym_path)
            # os.symlink(output_path, output_sym_path)

            # add seed to config changes
            changes.append({constants.SEED: seed})

            experiment_utils.config_changes_to_json(
                config_changes=changes, json_path=config_changes_path
            )

    pbs_array_index = "$PBS_ARRAY_INDEX/"
    # error_pbs_array_index = "error_$PBS_ARRAY_INDEX.txt"
    # output_pbs_array_index = "output_$PBS_ARRAY_INDEX.txt"
    config_changes_pbs_array_index = "config_changes_$PBS_ARRAY_INDEX.json"
    # error_path_string = f'"{os.path.join(error_files_dir, error_pbs_array_index)}"'
    # output_path_string = f'"{os.path.join(output_files_dir, output_pbs_array_index)}"'

    run_command = (
        f"python cluster_array_run.py --config_path {config_path} "
        f'--config_changes "{os.path.join(config_changes_dir, config_changes_pbs_array_index)}" '
        f'--checkpoint_path "{os.path.join(checkpoint_paths_dir, pbs_array_index)}" '
    )

    cluster_methods.create_job_script(
        run_command=run_command,
        save_path=job_script_path,
        num_cpus=num_cpus,
        conda_env_name="ach",
        memory=memory,
        error_path="",
        output_path="",
        checkpoint_path=os.path.join(checkpoint_paths_dir, pbs_array_index),
        array_job_length=num_configurations * num_seeds,
    )

    subprocess.call(f"qsub {job_script_path}", shell=True)


def single_run(config_path: str, checkpoint_path: str, changes: List[Dict] = []):

    config = ach_config.AchConfig(config=config_path, changes=changes)

    seed = config.seed

    experiment_utils.set_random_seeds(seed)
    config = experiment_utils.set_device(config)

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
    elif config.type == constants.DQN:
        r = dqn_runner.DQNRunner(config=config)
    elif config.type == constants.ENSEMBLE_Q_LEARNING:
        r = q_learning_ensemble_runner.EnsembleQLearningRunner(config=config)
    else:
        raise ValueError(f"Learner type {type} not recognised.")

    r.train()
    r.post_process()


if __name__ == "__main__":
    args = parser.parse_args()
    config_changes = experiment_utils.json_to_config_changes(args.config_changes)
    single_run(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        changes=config_changes,
    )
