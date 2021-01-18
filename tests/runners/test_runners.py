import datetime
import os
import tempfile
import time
from typing import Any
from typing import List
from typing import Tuple

import constants
from experiments import ach_config
from runners import base_runner
from runners import q_learning_ensemble_runner
from utils import experiment_utils

RUNNER_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_CONFIG_FILE_PATH = os.path.join(RUNNER_TEST_DIR, "test_config.yaml")
ATARI_TEST_CONFIG_FILE_PATH = os.path.join(RUNNER_TEST_DIR, "atari_test_config.yaml")


def _get_base_config(config_path: str) -> ach_config.AchConfig:
    """
    Generate base configuration object from test yaml file.

    Use temporary folder location to write data of tests.

    Returns:
        config: Configuration object
    """
    config = ach_config.AchConfig(
        config=config_path,
    )

    config = experiment_utils.set_device(config)
    experiment_utils.set_random_seeds(config.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_datetime = datetime.datetime.fromtimestamp(time.time())
        experiment_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        checkpoint_path = os.path.join(tmpdir, "test_results", experiment_timestamp)

    config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, experiment_timestamp)
    config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(
        constants.Constants.LOGFILE_PATH,
        os.path.join(checkpoint_path, "data_logger.csv"),
    )

    return config


class TestRunners:
    """Integration test for runners."""

    def _test_runner(
        self,
        runner: base_runner.BaseRunner,
        config_changes: List[Tuple[str, Any]],
        base_config: ach_config.AchConfig,
    ):

        for change in config_changes:
            base_config.amend_property(
                property_name=change[0],
                new_property_value=change[1],
            )

        runner_instance = runner(base_config)
        runner_instance.train()
        runner_instance.post_process()

    def _test_ensemble_q_runner(self):
        config_changes = [
            ("environment", "multiroom"),
            ("apply_curriculum", False),
            ("ascii_map_path", os.path.join(RUNNER_TEST_DIR, "test_ascii.txt")),
            ("type", "ensemble_q_learning"),
            ("visitation_penalty_type", "adaptive_uncertainty"),
        ]
        base_config = _get_base_config(config_path=TEST_CONFIG_FILE_PATH)
        self._test_runner(
            runner=q_learning_ensemble_runner.EnsembleQLearningRunner,
            config_changes=config_changes,
            base_config=base_config,
        )

    def test_dqn_runner(self):
        config_changes = [
            ("apply_curriculum", False),
            ("type", "dqn"),
            ("visitation_penalty_type", "hard_coded"),
        ]
        base_config = _get_base_config(config_path=ATARI_TEST_CONFIG_FILE_PATH)
        self._test_runner(
            runner=q_learning_ensemble_runner.EnsembleQLearningRunner,
            config_changes=config_changes,
            base_config=base_config,
        )
