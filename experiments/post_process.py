import argparse
import json
import os

import constants
from utils import plotting_functions

parser = argparse.ArgumentParser()

parser.add_argument("--results_folder",
                    type=str,
                    help="path to results folder to post-process.")
parser.add_argument("--smoothing",
                    type=int,
                    help="window width for moving averaging.",
                    default=40)

if __name__ == "__main__":
    args = parser.parse_args()
    config_changes_json_path = os.path.join(args.results_folder,
                                            "all_config_changes.json")

    with open(config_changes_json_path, "r") as f:
        changes = json.load(f)
        exp_names = list(changes.keys())
    plotting_functions.plot_all_multi_seed_multi_run(
        folder_path=args.results_folder,
        exp_names=exp_names,
        window_width=args.smoothing)
