"""
Main.py
Author: marco Hidalgo
"""

from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from utils_ import logging
from churn_library import CustomerChurn


logging.getLogger(__name__)


def run(config: DictConfig):
    """

    :param config: config to run pipeline
    :return: Ran pipeline
    """

    try:
        prediction = CustomerChurn(config)
        prediction.predict()
        logging.info('Script executed')
    except BaseException as err:
        logging.error('Error while starting execution {}'. format(err))


if __name__ == "__main__":
    logging.info("Initializing script")
    project_path = Path.cwd()
    config_path = (project_path / 'config' / 'config.yaml')
    config = OmegaConf.load(config_path)
    config['DIRECTORIES']['PROJECT_DIR'] = str(project_path)
    run(config)
