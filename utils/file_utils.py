import torch
import shutil
import logging
import coloredlogs
import os


def get_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path)
    # log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger