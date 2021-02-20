import datetime
import json
import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import constants
import torch
from experiments import ach_config


def set_random_seeds(seed: int) -> None:
    # import packages with non-deterministic behaviour
    import random

    import numpy as np

    # set random seeds for these packages
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(config: ach_config.AchConfig,
               logger: Optional = None) -> ach_config.AchConfig:
    """Establish availability of GPU."""
    if logger is not None:
        print_fn = logger.info
    else:
        print_fn = print
    if config.use_gpu:
        print_fn("Attempting to find GPU...")
        if torch.cuda.is_available():
            print_fn("GPU found, using the GPU...")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            config.add_property(constants.Constants.USING_GPU, True)
            experiment_device = torch.device("cuda:{}".format(config.gpu_id))
            print_fn(f"Device in use: {experiment_device}")
        else:
            print_fn("GPU not found, reverting to CPU")
            config.add_property(constants.Constants.USING_GPU, False)
            experiment_device = torch.device("cpu")
    else:
        print_fn("Using the CPU")
        experiment_device = torch.device("cpu")
    config.add_property(constants.Constants.EXPERIMENT_DEVICE,
                        experiment_device)
    return config


def save_config_changes(config_changes: Dict[str, List[Tuple[str, Any, bool]]],
                        file_name: str) -> None:
    with open(file_name, "w") as fp:
        json.dump(config_changes, fp, indent=4)


def _distribute_over_gpus(config_changes: Dict[str, List[Tuple[Any]]]):
    """If performing a parallel run with GPUs we want to split our processes evenly
    over the available GPUs. This method allocates tasks (as evenly as possible)
    over the GPUs.

    Method changes config_changes object in-place.
    """
    num_runs = len(args.seeds) * len(config_changes)
    num_gpus_available = torch.cuda.device_count()

    if num_gpus_available > 0:
        even_runs_per_gpu = num_runs // num_gpus_available
        excess_runs = num_runs % num_gpus_available

        gpu_ids = {}

        for i in range(num_gpus_available):
            jobs_with_id_i = {
                j: i for j in range(i * even_runs_per_gpu, (i + 1) *
                                    even_runs_per_gpu)
            }
            gpu_ids.update(jobs_with_id_i)

        for i in range(excess_runs):
            gpu_ids[num_runs - excess_runs + i] = i

        for i, config_change in enumerate(config_changes.values()):
            config_change.append((constants.Constants.GPU_ID, gpu_ids[i]))


def config_changes_to_json(config_changes: List[Dict], json_path: str) -> None:
    """Write a list of dictionaries to json file."""
    with open(json_path, "w") as json_file:
        json.dump(config_changes, json_file)


def json_to_config_changes(json_path: str) -> List[Dict]:
    """Read a list of dictionaries from json file."""
    with open(json_path, "r") as json_file:
        config_changes = json.load(json_file)
    return config_changes


def get_experiment_timestamp() -> str:
    """Get a timestamp in YY-MM-DD-HH-MM-SS format."""
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return exp_timestamp


def get_checkpoint_path(
    folder: str,
    timestamp: str,
    experiment_name: str,
    subfolder_name: Optional[str] = "",
) -> str:
    """Get full checkpoint path for experiment logs etc."""
    checkpoint_path = os.path.join(folder, timestamp, experiment_name,
                                   subfolder_name)
    return checkpoint_path
