import logging
import os

import constants


def get_logger(experiment_path: str, name: str):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(constants.Constants.LOG_FORMAT)

    file_handler = logging.FileHandler(os.path.join(
        experiment_path, constants.Constants.LOG_FILE_NAME))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# logging.basicConfig(level=logging.INFO, format=constants.Constants.LOG_FORMAT,
#                     filename=, filemode='w')

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(logging.Formatter(constants.Constants.LOG_FORMAT))

# logging.getLogger(name).addHandler(console)

# return
