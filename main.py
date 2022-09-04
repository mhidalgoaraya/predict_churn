from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from utils.utils import logging
from preprocessing.make_model import CustomerChurn
import logging

logging.getLogger(__name__)


def run(config: DictConfig):
    try:
        prediction = CustomerChurn(config)
        prediction.predict()
        logging.info('Script executed')
    except BaseException as e:
        logging.error(f'{e} Error while starting execution')


if __name__ == "__main__":
    logging.info("nitializing script")
    project_path = Path.cwd()
    config_path = (project_path / 'config'/'config.yaml')
    config = OmegaConf.load(config_path)
    config['DIRECTORIES']['PROJECT_DIR'] = str(project_path)
    run(config)
