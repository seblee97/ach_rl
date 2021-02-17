"""
This script is for runs being executed on the cluster.

It first creates a job script with the relevant specifications, 
and then excutes it in a subprocess.
"""
import os
import subprocess

import constants
from utils import cluster_methods

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def cluster_run(experiment_path: str,
                config_path: str,
                num_cpus: int,
                memory_per_node: int,
                num_gpus: int,
                gpu_type: str,
                cluster_debug: bool = False) -> None:
    """Set of experiments run in parallel.

    Args:
        config_path: path to yaml file.
        checkpoint_paths: list of paths to directories to output results.
    """

    job_script_path = os.path.join(experiment_path,
                                   constants.Constants.JOB_SCRIPT)
    env = constants.Constants.ACH
    error_path = os.path.join(experiment_path,
                              constants.Constants.ERROR_FILE_NAME)
    output_path = os.path.join(experiment_path,
                               constants.Constants.OUTPUT_FILE_NAME)

    run_command = (
        f"python run_modes/parallel_run.py --config_path {config_path} "
        f"--experiment_path {experiment_path}")

    cluster_methods.create_job_script(run_command=run_command,
                                      save_path=job_script_path,
                                      num_cpus=num_cpus,
                                      conda_env_name=env,
                                      memory=memory_per_node,
                                      num_gpus=num_gpus,
                                      gpu_type=gpu_type,
                                      error_path=error_path,
                                      output_path=output_path)

    if cluster_debug:
        subprocess.call(run_command, shell=True)
    else:
        subprocess.call(f"qsub {job_script_path}", shell=True)
