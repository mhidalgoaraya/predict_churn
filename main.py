from pathlib import Path
from omegaconf import DictConfig
from utils.utils import logging
from preprocessing.make_model import CustomerChurn

project_path = Path(__file__).parent.parent
config_path = str((project_path / 'config').as_posix())


def run(config:DictConfig):
    try:
        logging.error(f'Error: Data path does not exist')

        prediction = CustomerChurn(config:DictConfig)
        prediction.compute()
        logging.info('INFO: Script executed')

    except:
        logging.error(f'Error: Error while starting execution')


if __name__ == "__main__":
    config = DictConfig(config_path)
    config['PROJECT_DIR'] = str(project_path).as_posix()
    run(config)
