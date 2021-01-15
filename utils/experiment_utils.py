import datetime
import os
import time
from typing import Optional

import torch

import constants
from experiments import ach_config


def set_random_seeds(seed: int) -> None:
    # import packages with non-deterministic behaviour
    import random

    import numpy as np

    # set random seeds for these packages
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(
    config: ach_config.AchConfig,
) -> ach_config.AchConfig:
    """Establish availability of GPU."""
    if config.use_gpu:
        print("Attempting to find GPU...")
        if torch.cuda.is_available():
            print("GPU found, using the GPU...")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            config.add_property(constants.Constants.USING_GPU, True)
            experiment_device = torch.device("cuda:{}".format(config.gpu_id))
            print(f"Device in use: {experiment_device}")
        else:
            print("GPU not found, reverting to CPU")
            config.add_property(constants.Constants.USING_GPU, False)
            experiment_device = torch.device("cpu")
    else:
        print("Using the CPU")
        experiment_device = torch.device("cpu")
    config.add_property(constants.Constants.EXPERIMENT_DEVICE, experiment_device)
    return config


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
    checkpoint_path = os.path.join(folder, timestamp, experiment_name, subfolder_name)
    return checkpoint_path
