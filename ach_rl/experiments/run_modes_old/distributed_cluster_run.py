"""
This script is for runs being executed on the cluster.

While the cluter_run.py deals with submitting one job and parallelising within that job,
this script is for submitting a range of jobs.
"""
import os
import subprocess
from typing import List

import constants
from utils import cluster_methods
from utils import experiment_utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def distributed_cluster_run(
    experiment_path: str,
    config_path: str,
    checkpoint_paths: List[str],
    num_cpus: int,
    memory_per_node: int,
    num_gpus: int,
    gpu_type: str,
    cluster_debug: bool,
) -> None:
    """Set of experiments run in parallel.

    Args:
        config_path: path to yaml file.
        checkpoint_paths: list of paths to directories to output results.
    """
    for checkpoint_path in checkpoint_paths:
        changes_path = os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
        changes = experiment_utils.json_to_config_changes(changes_path)
        job_path = os.path.join(experiment_path, checkpoint_path)
        job_script_path = os.path.join(job_path, constants.JOB_SCRIPT)
        env = constants.ACH
        error_path = os.path.join(job_path, constants.ERROR_FILE_NAME)
        output_path = os.path.join(job_path, constants.OUTPUT_FILE_NAME)

        run_command = (
            f"python run_modes/single_run.py --config_path {config_path} "
            f"--config_changes_path {changes_path} "
            f"--checkpoint_path {job_path}"
        )

        cluster_methods.create_job_script(
            run_command=run_command,
            save_path=job_script_path,
            num_cpus=num_cpus,
            conda_env_name=env,
            memory=memory_per_node,
            num_gpus=num_gpus,
            gpu_type=gpu_type,
            error_path=error_path,
            output_path=output_path,
        )

        if cluster_debug:
            subprocess.call(run_command, shell=True)
        else:
            subprocess.call(f"qsub {job_script_path}", shell=True)
